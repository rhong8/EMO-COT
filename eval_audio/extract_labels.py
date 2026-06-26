import json
import re
from sklearn.metrics import accuracy_score, recall_score, f1_score

results_file = '/content/meld_260626054936.json'  # replace with your actual filename

def extract_label(response):
    match = re.search(r'\b([A-E])\b', response)
    return match.group(1) if match else response.strip()

with open(results_file) as f:
    results = json.load(f)

refs, hyps = [], []
for result in results:
    gt = result['gt']
    response = extract_label(result['response'].lstrip())
    refs.append(gt)
    hyps.append(response)

# print a few to verify extraction is working
for i in range(5):
    print(f"GT: {repr(refs[i])} | Extracted: {repr(hyps[i])} | Raw: {repr(results[i]['response'])}")

labels = sorted(list(set(refs + hyps)))

acc = accuracy_score(refs, hyps)
wa = recall_score(refs, hyps, average='weighted', labels=labels, zero_division=0)
ua = recall_score(refs, hyps, average='macro', labels=labels, zero_division=0)
f1 = f1_score(refs, hyps, average='macro', labels=labels, zero_division=0)

print(f"\nmeld ACC: {acc:.4f} | WA: {wa:.4f} | UA: {ua:.4f} | F1: {f1:.4f} | Count: {len(hyps)}")