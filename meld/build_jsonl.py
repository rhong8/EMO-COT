import os
import json
import re
import pandas as pd

wav_dir = '/content/drive/MyDrive/MELD.Raw/output_repeated_splits_test_wav'
emotion_graph_dir = '/content/drive/MyDrive/MELD.Raw/emotion-graph'
ground_truth = pd.read_csv('/content/drive/MyDrive/MELD.Raw/test_sent_emo.csv')
jsonl_path = '/content/drive/MyDrive/MELD.Raw/meld_eval.jsonl'

# Maps both emotion words and option letters to the canonical lowercase label letter.
_RESPONSE_MAP = {
    'neutral':   'a', 'a': 'a',
    'happy':     'b', 'joy': 'b',       'b': 'b',
    'sad':       'c', 'sadness': 'c',   'c': 'c',
    'surprised': 'd', 'surprise': 'd',  'd': 'd',
    'angry':     'e', 'anger': 'e',     'e': 'e',
}

def parse_response(response: str) -> str | None:
    """Return a lowercase label letter (a-e) from an LLM response.

    Accepts a bare letter (A-E), an emotion word (e.g. 'Neutral', 'Happy'),
    or a longer reply containing either.  Returns None when no recognisable
    token is found.
    """
    text = response.strip().lower()
    # Prefer an explicit option letter so "The answer is B" maps to 'b'.
    letter_match = re.search(r'\b([a-e])\b', text)
    if letter_match:
        return letter_match.group(1)
    # Fall back to an emotion word anywhere in the response.
    for token, label in _RESPONSE_MAP.items():
        if re.search(r'\b' + token + r'\b', text):
            return label
    return None


def construct_jsonl():
    i = 0
    with open(jsonl_path, 'w') as f:
        for filename in os.listdir(wav_dir):
            try:
                numbers = re.findall(r'\d+', filename)
                dia_id = int(numbers[0])
                utt_id = int(numbers[1])

                emotion_graph_filename = f"emotion_graph_{filename.replace('.wav', '')}.json"
                emotion_graph_path = os.path.join(emotion_graph_dir, emotion_graph_filename)
                wav_path = os.path.join(wav_dir, filename)

                with open(emotion_graph_path) as ef:
                    graph = json.load(ef)
                graph_str = json.dumps(graph)

                prompt_string = (
                    f"Emotion Graph:\n{graph_str}\n"
                    f"Use the audio and emotion graph as context and answer the following question.\n"
                    f"Task: Recognize the emotion with keywords in English: (A) Neutral (B) Happy (C) Sad (D) Surprised (E) Angry\n"
                    f"Answer only with the option letter and nothing else (A, B, C, D, or E)"
                )

                row = ground_truth[
                    (ground_truth['Dialogue_ID'] == dia_id) &
                    (ground_truth['Utterance_ID'] == utt_id)
                ].iloc[0]
                emotion = row['Emotion']

                match emotion:
                    case "neutral":
                        label_letter = 'a'
                    case "joy":
                        label_letter = 'b'
                    case "sadness":
                        label_letter = 'c'
                    case "surprise":
                        label_letter = 'd'
                    case "anger":
                        label_letter = 'e'
                    case _:
                        i += 1
                        continue

                item = {
                    "audio": wav_path,
                    "prompt": prompt_string,
                    "source": "meld",
                    "gt": label_letter
                }
                f.write(json.dumps(item) + '\n')

                if i % 100 == 99:
                    print(f"Processed file {i + 1}")

                i += 1

            except Exception as e:
                print(f"Exception occured at file {filename}, file no {i + 1} : {e} ")
                continue


if __name__ == "__main__":
    construct_jsonl()
