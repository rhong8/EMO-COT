import argparse
import json
import os
import random
import time
from functools import partial
import torch
import requests

from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, f1_score
import soundfile as sf

#from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
#from qwen_omni_utils import process_mm_info
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 数据集配置
ds_collections = {
    'meld': {'path': 'ser/meld_eval.jsonl'},
    'iemocap': {'path': 'ser/iemocap_eval.jsonl'},
    'merr_test1': {'path': 'ser/merr_eval_test1.jsonl'},
    'merr_test2': {'path': 'ser/merr_eval_test2.jsonl'}
}

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        path = ds['path']
        self.datas = open(path).readlines()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        audio = data['audio']
        source = data['source']
        prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>" + data['prompt']
        gt = data['gt']
        return {
            'audio': audio,
            'prompt': prompt,
            'source': source,
            'gt': gt
        }

def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs

def collate_fn(inputs, processor):
    input_texts = [_['prompt'] for _ in inputs]
    source = [_['source'] for _ in inputs]
    gt = [_['gt'] for _ in inputs]
    audio_path = [_['audio'] for _ in inputs]
    input_audios = [ffmpeg_read(read_audio(_['audio']), sampling_rate=processor.feature_extractor.sampling_rate) for _ in inputs]
    inputs = processor(text=input_texts, audios=input_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    return inputs, audio_path, source, gt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B')
    #parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2.5-Omni-7B')
    parser.add_argument('--dataset', type=str, default='meld')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # 加载模型和处理器
    if args.checkpoint == 'Qwen/Qwen2-Audio-7B':
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.checkpoint, device_map='cuda', trust_remote_code=True, torch_dtype='auto').eval()
        processor = AutoProcessor.from_pretrained(args.checkpoint)
    else:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        
    #processor = AutoProcessor.from_pretrained(args.checkpoint)
    processor.tokenizer.padding_side = 'left'

    random.seed(args.seed)
    dataset = AudioDataset(
        ds=ds_collections[args.dataset],
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
    )

    # 推理循环
    gts = []
    sources = []
    rets = []
    audio_paths = []
    for _, (inputs, audio_path, source, gt) in tqdm(enumerate(data_loader)):
        inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        output_ids = model.generate(**inputs, max_new_tokens=256, min_new_tokens=1, do_sample=False)
        output_ids = output_ids[:, inputs['input_ids'].size(1):]
        output = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        gts.extend(gt)
        rets.extend(output)
        sources.extend(source)
        audio_paths.extend(audio_path)

    # 评估结果
    print(f"Evaluating {args.dataset} ...")
    results = []
    for gt, response, source, audio_path in zip(gts, rets, sources, audio_paths):
        results.append({
            'gt': gt,
            'response': response,
            'source': source,
            'audio_path': audio_path,
        })
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file = f'{args.dataset}_{time_prefix}.json'
    json.dump(results, open(results_file, 'w'))
    results_dict = {}
    for item in tqdm(results):
        source = item["source"]
        results_dict.setdefault(source, []).append(item)

    # for source in results_dict:
    #     refs, hyps = [], []
    #     results_list = results_dict[source]
    #     for result in results_list:
    #         gt = result["gt"]
    #         response = result["response"].lstrip()
    #         refs.append(gt)
    #         hyps.append(response)
    #     score = accuracy_score(refs, hyps)
    #     print(f"{source} ACC_score:", score, len(hyps))
    for source in results_dict:
        refs, hyps = [], []
        results_list = results_dict[source]
        for result in results_list:
            gt = result["gt"]
            response = result["response"].lstrip()
            refs.append(gt)
            hyps.append(response)

        # Label list for calculating WA / UA
        labels = sorted(list(set(refs + hyps)))

        acc = accuracy_score(refs, hyps)
        wa = recall_score(refs, hyps, average='weighted', labels=labels, zero_division=0)
        ua = recall_score(refs, hyps, average='macro', labels=labels, zero_division=0)
        f1 = f1_score(refs, hyps, average='macro', labels=labels, zero_division=0)

        print(f"{source} ACC: {acc:.4f} | WA: {wa:.4f} | UA: {ua:.4f} | F1: {f1:.4f} | Count: {len(hyps)}")
