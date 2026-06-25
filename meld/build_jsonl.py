import os
import json
import re
import pandas as pd

wav_dir = '/content/drive/MyDrive/MELD.Raw/output_repeated_splits_test_wav'
emotion_graph_dir = '/content/drive/MyDrive/MELD.Raw/emotion-graph'
ground_truth = pd.read_csv('/content/drive/MyDrive/MELD.Raw/test_sent_emo.csv')
jsonl_path = '/content/drive/MyDrive/MELD.Raw/meld_eval.jsonl'


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
                    f"Select the option letter from the provided choices to answer."
                )

                row = ground_truth[(ground_truth['Dialogue_ID'] == dia_id) & (ground_truth['Utterance_ID'] == utt_id)].iloc[0]
                emotion = row['Emotion']

                match emotion:
                    case "neutral":
                        label_letter = 'A'
                    case "joy":
                        label_letter = 'B'
                    case "sadness":
                        label_letter = 'C'
                    case "surprise":
                        label_letter = 'D'
                    case "anger":
                        label_letter = 'E'
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




            except Execption as e:
                print(f"Exception occured at file {filename}, file no {i + 1} : {e} ")
                continue
            

if __name__ == "__main__":
    construct_jsonl()