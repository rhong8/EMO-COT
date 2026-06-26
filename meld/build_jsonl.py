import json
import re
from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, f1_score

results_file = '/content/drive/MyDrive/MELD.Raw/meld_260626054936.json'  # replace with your actual filename

emotion_to_letter = {
    'neutral': 'A',
    'happy': 'B',
    'joy': 'B',
    'sad': 'C',
    'sadness': 'C',
    'surprised': 'D',
    'surprise': 'D',
    'angry': 'E',
    'anger': 'E',
}

def extract_label(response):
    response_clean = response.strip().lower()
    match = re.search(r'\b([a-e])\b', response_clean)
    if match:
        return match.group(1).upper()
    for emotion, letter in emotion_to_letter.items():
        if emotion in response_clean:
            return letter
    return response_clean.strip()

with open(results_file) as f:
    results = json.load(f)

refs, hyps = [], []
for result in results:
    refs.append(result['gt'])
    hyps.append(extract_label(result['response']))

print("Prediction distribution:", Counter(hyps))
print("Ground truth distribution:", Counter(refs))

labels = sorted(list(set(refs + hyps)))
acc = accuracy_score(refs, hyps)
wa = recall_score(refs, hyps, average='weighted', labels=labels, zero_division=0)
ua = recall_score(refs, hyps, average='macro', labels=labels, zero_division=0)
f1 = f1_score(refs, hyps, average='macro', labels=labels, zero_division=0)

print(f"\nmeld ACC: {acc:.4f} | WA: {wa:.4f} | UA: {ua:.4f} | F1: {f1:.4f} | Count: {len(hyps)}")