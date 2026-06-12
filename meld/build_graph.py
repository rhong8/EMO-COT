import os
import json
import pandas as pd
import librosa
import spacy
from transformers import pipeline
from sklearn.metrics import accuracy_score
import opensmile

# 加载 spaCy 模型用于关键词提取
nlp = spacy.load('en_core_web_sm')

# 加载情感分析模型
sentiment_analyzer = pipeline("sentiment-analysis")

smile = opensmile.Smile(
    features = opensmile.FeatureSet.eGeMAPSv02,
    feature_level = opensmile.FeatureLevel.Functionals,
)
# 提取并分类音频特征的函数
def extract_audio_features(audio_path, utterance):
    try:
        audio, sr = librosa.load(audio_path)
        pitch, _ = librosa.piptrack(y=audio, sr=sr)
        pitch_mean = pitch.mean()
        pitch_label = "high" if pitch_mean > 100 else "low"

        duration = librosa.get_duration(y=audio, sr=sr)
        word_count = len(utterance.split())
        rate = word_count / duration if duration > 0 else 0
        rate_label = "fast" if rate > 2 else "slow"

        volume = librosa.feature.rms(y=audio).mean()
        volume_label = "loud" if volume > 0.1 else "soft"

        return [
            {"id": "1", "feature": "pitch", "value": pitch_label},
            {"id": "2", "feature": "rate", "value": rate_label},
            {"id": "3", "feature": "volume", "value": volume_label}
        ]
    except Exception as e:
        print(f"处理音频文件 {audio_path} 时出错: {e}")
        return [
            {"id": "1", "feature": "pitch", "value": "unknown"},
            {"id": "2", "feature": "rate", "value": "unknown"},
            {"id": "3", "feature": "volume", "value": "unknown"}
        ]

# 从话语中提取关键词的函数
def extract_keyword(utterance):
    doc = nlp(utterance)
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
    return keywords[0] if keywords else "unknown"

# 使用 Transformers 模型预测情感极性的函数
def get_sentiment(utterance):
    result = sentiment_analyzer(utterance)[0]
    label = result['label'].lower()
    if label == 'positive':
        return "positive"
    elif label == 'negative':
        return "negative"
    else:
        return "objective" 

def get_relation(feature_label, sentiment):
    if sentiment == "objective":
        return "objective"  
    feature_sentiment = "positive" if feature_label in ["high", "fast", "loud"] else "negative"
    return "supports" if feature_sentiment == sentiment else "conflicts"

def build_emotion_graph(csv_path, audio_dir, emotion_graph_dir="meld/emotion_graph"):
    if not os.path.exists(emotion_graph_dir):
        os.makedirs(emotion_graph_dir)

    df = pd.read_csv(csv_path)
    predictions = []
    ground_truths = []

    for idx, row in df.iterrows():
        utterance = row['Utterance']
        ground_truth_sentiment = row['Sentiment'].lower() if pd.notna(row['Sentiment']) else "objective"
        if ground_truth_sentiment == "neutral":
            ground_truth_sentiment = "objective"  # 修改

        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']

        predicted_sentiment = get_sentiment(utterance)
        predictions.append(predicted_sentiment)
        ground_truths.append(ground_truth_sentiment)

        audio_file = f"dia{dialogue_id}_utt{utterance_id}.wav"
        audio_path = os.path.join(audio_dir, audio_file)

        if not os.path.exists(audio_path):
            print(f"未找到音频文件: {audio_path}")
            continue

        audio_features = extract_audio_features(audio_path, utterance)
        keyword = extract_keyword(utterance)

        text_data = [
            {
                "id": "4",
                "content": utterance,
                "keyword": keyword,
                "sentiment": ground_truth_sentiment
            }
        ]

        relationships = [
            {"from": "1", "to": "4", "relation": get_relation(audio_features[0]['value'], ground_truth_sentiment)},
            {"from": "2", "to": "4", "relation": get_relation(audio_features[1]['value'], ground_truth_sentiment)},
            {"from": "3", "to": "4", "relation": get_relation(audio_features[2]['value'], ground_truth_sentiment)}
        ]

        emotion_graph = {
            "audio": audio_features,
            "text": text_data,
            "relationships": relationships
        }

        filename = f"emotion_graph_dia{dialogue_id}_utt{utterance_id}.json"
        output_path = os.path.join(emotion_graph_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(emotion_graph, f, indent=4, ensure_ascii=False)

        print(f"已保存 Emotion Graph 到: {output_path}")

    if ground_truths and predictions:
        accuracy = accuracy_score(ground_truths, predictions)
        print(f"Transformers 模型预测的 sentiment 准确率: {accuracy:.2f}")
    else:
        print("无法计算准确率：没有可用的 ground truth 或预测数据")

if __name__ == "__main__":
    csv_path = "YOUR_GRAPH_PATH"
    audio_dir = "YOUR_AUDIO_PATH"
    build_emotion_graph(csv_path, audio_dir)
