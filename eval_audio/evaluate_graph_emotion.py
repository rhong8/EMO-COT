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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Maps dataset names to their .jsonl file paths.
# Each .jsonl contains one utterance per line with audio path, prompt, source, and ground truth label.
ds_collections = {
    'meld': {'path': '/content/drive/MyDrive/MELD.Raw/meld_eval.jsonl'},
    #'iemocap': {'path': 'ser/iemocap_eval.jsonl'},
    #'merr_test1': {'path': 'ser/merr_eval_test1.jsonl'},
    #'merr_test2': {'path': 'ser/merr_eval_test2.jsonl'}
}

# PyTorch Dataset class that reads the .jsonl file line by line.
# Each line is parsed into audio path, source, prompt (with emotion graph embedded), and ground truth label.
# The prompt gets the Qwen2-Audio audio token prepended so the model knows where to inject the audio encoding.
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

# Reads raw audio bytes from either a local file path or a URL.
def read_audio(audio_path):
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        inputs = requests.get(audio_path).content
    else:
        with open(audio_path, "rb") as f:
            inputs = f.read()
    return inputs

# Collation function called by DataLoader for each batch.
# Reads and decodes raw audio for each sample using ffmpeg, then passes both
# the text prompts and audio waveforms through the Qwen2-Audio processor
# to produce tokenized tensors ready for GPU inference.
def collate_fn(inputs, processor):
    input_texts = [_['prompt'] for _ in inputs]
    source = [_['source'] for _ in inputs]
    gt = [_['gt'] for _ in inputs]
    audio_path = [_['audio'] for _ in inputs]
    input_audios = [ffmpeg_read(read_audio(_['audio']), sampling_rate=processor.feature_extractor.sampling_rate) for _ in inputs]
    inputs = processor(text=input_texts, audio=input_audios, return_tensors="pt", padding=True)
    return inputs, audio_path, source, gt

if __name__ == '__main__':
    # Parse command line arguments for model checkpoint, dataset, batch size, workers, and random seed.
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Qwen/Qwen2-Audio-7B')
    parser.add_argument('--dataset', type=str, default='meld')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Load Qwen2-Audio-7B model and its processor onto GPU in half precision.
    # The processor handles both text tokenization and audio feature extraction.
    if args.checkpoint == 'Qwen/Qwen2-Audio-7B':
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.checkpoint, device_map='cuda', torch_dtype='auto').eval()
        processor = AutoProcessor.from_pretrained(args.checkpoint)
    else:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")
        processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    # Left-pad tokenized inputs so that all sequences in a batch end at the same position,
    # which is required for correct autoregressive generation with batch size > 1.
    processor.tokenizer.padding_side = 'left'

    # Seed for reproducibility, then build the dataset and DataLoader.
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

    # Inference loop: for each batch, move tensors to GPU, run greedy decoding,
    # strip the prompt tokens from the output, and decode predicted token IDs back to text.
    # Collects predictions, ground truths, sources, and audio paths across all batches.
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

    # Assemble all predictions and ground truths into a results list and save to a timestamped JSON file.
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

    # Group results by source dataset for per-dataset metric calculation.
    results_dict = {}
    for item in tqdm(results):
        source = item["source"]
        results_dict.setdefault(source, []).append(item)

    # For each source, compute ACC, WA (weighted recall), UA (macro recall), and F1.
    # Labels are derived from the union of ground truth and predicted labels to handle any unseen predictions.
    for source in results_dict:
        refs, hyps = [], []
        results_list = results_dict[source]
        for result in results_list:
            gt = result["gt"]
            response = result["response"].lstrip()
            refs.append(gt)
            hyps.append(response)

        labels = sorted(list(set(refs + hyps)))

        acc = accuracy_score(refs, hyps)
        wa = recall_score(refs, hyps, average='weighted', labels=labels, zero_division=0)
        ua = recall_score(refs, hyps, average='macro', labels=labels, zero_division=0)
        f1 = f1_score(refs, hyps, average='macro', labels=labels, zero_division=0)

        print(f"{source} ACC: {acc:.4f} | WA: {wa:.4f} | UA: {ua:.4f} | F1: {f1:.4f} | Count: {len(hyps)}")