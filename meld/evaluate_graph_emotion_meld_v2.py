'''This file assesses the final results of the classification


'''


import argparse
import json
import os
import random
import time
from functools import partial
import torch
import requests

from build_jsonl import parse_response

from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Dataset configuration
# ds_collections = {
#     'meld': {'path': 'ser/meld_eval.jsonl'}
# }
ds_collections = {
    'meld': {'path': 'ser/meld_eval.jsonl'},
    'iemocap': {'path': 'ser/iemocap_eval.jsonl'},
    'merr_test1': {'path': 'ser/merr_eval_test1.jsonl'},
    'merr_test2': {'path': 'ser/merr_eval_test2.jsonl'}
}


wav_path = 'content/drive/MyDrive/MELD.Raw/output_repeated_splits_test_wav' #YOUR WAV PATH HERE
emotion_path = 'content/drive/MyDrive/MELD.Raw/emotion-graph'


# Dataset class
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
        prompt = data['prompt']  # Original prompt
        gt = data['gt']
        return {
            'audio': audio,
            'prompt': prompt,
            'source': source,
            'gt': gt
        }

# Read audio file
def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs

# Data collation function
def collate_fn(inputs, processor):
    input_texts = []
    source = [_['source'] for _ in inputs]
    gt = [_['gt'] for _ in inputs]
    audio_path = [_['audio'] for _ in inputs]
    input_audios = [ffmpeg_read(read_audio(_['audio']), sampling_rate=processor.feature_extractor.sampling_rate) for _ in inputs]
    
    for i, item in enumerate(inputs):
        # Extract identifier from audio path, e.g., dia0_utt0
        audio_filename = os.path.basename(audio_path[i])  # e.g., dia0_utt0.wav
        identifier = audio_filename.replace('.wav', '')  # e.g., dia0_utt0
        # For iemocap
        if args.dataset == "iemocap":
            emotion_graph_path = os.path.join('iemocap/emotion_graph', f'emotion_graph_{identifier}.json')
        elif args.dataset == "meld":
            emotion_graph_path = os.path.join('meld/emotion_graph', f'emotion_graph_{identifier}.json')
        elif args.dataset == "merr_test1":
            emotion_graph_path = os.path.join('MERR_toolbox/EmotionGraph/test1', f'{identifier}.json')
        else:
            emotion_graph_path = os.path.join('MERR_toolbox/EmotionGraph/test2', f'{identifier}.json')
        # Load Emotion Graph
        try:
            with open(emotion_graph_path, 'r', encoding='utf-8') as f:
                emotion_graph = json.load(f)
            emotion_graph_str = json.dumps(emotion_graph, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to load Emotion Graph: {emotion_graph_path}, error: {e}")
            emotion_graph_str = "{}"  # If it fails, default to an empty dictionary
        
        # Build new prompt
        original_prompt = item['prompt']
        new_prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>Based on Emotion Graph: {emotion_graph_str} {original_prompt}"
        input_texts.append(new_prompt)
    
    # Use processor to process inputs
    inputs = processor(text=input_texts, audios=input_audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    return inputs, audio_path, source, gt

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B')
    #parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2.5-Omni-7B')
    parser.add_argument('--dataset', type=str, default='meld')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Load model and processor
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint, device_map='cuda', trust_remote_code=True, torch_dtype='auto').eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    processor.tokenizer.padding_side = 'left'

    # Set random seed
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

    # Inference loop
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

    # Evaluation results
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
    json.dump(results, open(results_file, 'w'), ensure_ascii=False)

    # Save as txt file
    txt_file = os.path.join(f"{args.dataset}_{time_prefix}_results.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        for item in results:
            audio = item["audio_path"]
            gt = item["gt"]
            pred = parse_response(item["response"]) or item["response"].strip().lower()
            f.write(f"AUDIO: {audio}\nGT: {gt}\nPREDICT: {pred}\n\n")
    print(f"Written to {txt_file}")
    
    # Calculate accuracy
    results_dict = {}
    for item in tqdm(results):
        source = item["source"]
        results_dict.setdefault(source, []).append(item)

    for source in results_dict:
        refs, hyps = [], []
        results_list = results_dict[source]
        for result in results_list:
            gt = result["gt"]
            response = parse_response(result["response"]) or result["response"].strip().lower()
            refs.append(gt)
            hyps.append(response)

        # Label list for calculating WA / UA
        labels = sorted(list(set(refs + hyps)))

        acc = accuracy_score(refs, hyps)
        wa = recall_score(refs, hyps, average='weighted', labels=labels, zero_division=0)
        ua = recall_score(refs, hyps, average='macro', labels=labels, zero_division=0)
        f1 = f1_score(refs, hyps, average='macro', labels=labels, zero_division=0)
        f1_weighted = f1_score(refs, hyps, average='weighted', labels=labels, zero_division=0)


        #print(f"{source} ACC: {acc:.4f} | WA: {wa:.4f} | UA: {ua:.4f} | F1: {f1:.4f} | Count: {len(hyps)}")
        print(f"{source} ACC: {acc:.4f} | WA: {wa:.4f} | UA: {ua:.4f} | F1: {f1:.4f} | WF1: {f1_weighted:.4f} | Count: {len(hyps)}")