import os
import json
import re
import librosa
import spacy
from transformers import pipeline

# Load spaCy model for keyword extraction
nlp = spacy.load('en_core_web_sm')

# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")
#sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
# label_map = {
#     "LABEL_0": "negative",
#     "LABEL_1": "neutral",
#     "LABEL_2": "positive"
# }

# Function to extract and classify audio features
def extract_audio_features(audio_path, utterance):
    try:
        audio, sr = librosa.load(audio_path)
        
        # Extract Pitch
        pitch, _ = librosa.piptrack(y=audio, sr=sr)
        pitch_mean = pitch.mean()
        pitch_label = "high" if pitch_mean > 100 else "low"
        
        # Extract Rate
        duration = librosa.get_duration(y=audio, sr=sr)
        word_count = len(utterance.split())
        rate = word_count / duration if duration > 0 else 0
        rate_label = "fast" if rate > 2 else "slow"
        
        # Extract Volume
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

# Function to extract keywords from utterance
def extract_keyword(utterance):
    doc = nlp(utterance)
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
    return keywords[0] if keywords else "unknown"

# Function to predict sentiment using Transformers model
def get_sentiment(utterance):
    result = sentiment_analyzer(utterance)[0]
    label = result['label'].lower()
    if label == 'positive':
        return "positive"
    elif label == 'negative':
        return "negative"
    else:
        return "neutral"
# def get_sentiment(utterance):
#     result = sentiment_analyzer(utterance)[0]
#     raw_label = result['label']  # e.g., "LABEL_0"
#     return label_map.get(raw_label, "neutral")  # 默认兜底为 neutral

# Function to define relationships based on feature and sentiment
def get_relation(feature_label, sentiment):
    if sentiment == "neutral":
        return "neutral"
    feature_sentiment = "positive" if feature_label in ["high", "fast", "loud"] else "negative"
    if feature_sentiment == sentiment:
        return "supports"
    else:
        return "conflicts"
# def get_relation(feature_label, sentiment):
#     # 定义高唤醒（激烈）和低唤醒（平静）特征
#     high_arousal = ["high", "fast", "loud"]
#     low_arousal = ["low", "slow", "soft"]

#     if sentiment == "neutral":
#         # 平静语音特征支持中性情绪，激烈语音特征冲突
#         if feature_label in low_arousal:
#             return "supports"
#         else:
#             return "conflicts"
#     else:
#         # 原本逻辑：正面情绪偏高激烈，负面偏低也可能激烈
#         feature_sentiment = "positive" if feature_label in high_arousal else "negative"
#         if feature_sentiment == sentiment:
#             return "supports"
#         else:
#             return "conflicts"


# Function to build transcription dictionary from IEMOCAP transcription files
def build_transcription_dict(transcription_dirs):
    """
    Build a dictionary mapping utterance identifiers to text captions from transcription files.
    
    Parameters:
        transcription_dirs (list): List of transcription directory paths (e.g., Session1, Session2, etc.)
    
    Returns:
        dict: Dictionary with keys as identifiers (e.g., 'Ses01F_impro01_F000') and values as text captions
    """
    transcription_dict = {}
    for session_dir in transcription_dirs:
        if not os.path.exists(session_dir):
            print(f"目录 {session_dir} 不存在，跳过")
            continue
        for txt_file in os.listdir(session_dir):
            if txt_file.endswith('.txt'):
                file_path = os.path.join(session_dir, txt_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Match pattern: Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
                        match = re.match(r"(\w+) \[\d+\.\d+-\d+\.\d+\]: (.+)", line.strip())
                        if match:
                            identifier = match.group(1)  # e.g., Ses01F_impro01_F000
                            text_caption = match.group(2)  # e.g., Excuse me.
                            transcription_dict[identifier] = text_caption
                        else:
                            print(f"警告: 文件 {txt_file} 中的行格式不正确: {line.strip()}")
    print(f"已构建 transcription 字典，包含 {len(transcription_dict)} 个条目")
    return transcription_dict

# Main function to build Emotion Graph for IEMOCAP
def build_emotion_graph_iemocap(audio_dir, transcription_dirs, json_dir, emotion_graph_dir="iemocap/emotion_graph"):
    """
    Build Emotion Graphs for IEMOCAP dataset and save them as JSON files.
    
    Parameters:
        audio_dir (str): Directory containing .wav audio files (e.g., 'iemocap/test_wav')
        transcription_dirs (list): List of transcription directory paths
        json_dir (str): Directory containing .json files with emotion labels (e.g., 'iemocap/test')
        emotion_graph_dir (str): Directory to save Emotion Graph JSON files
    """
    # Build transcription dictionary
    transcription_dict = build_transcription_dict(transcription_dirs)
    
    # Ensure emotion_graph directory exists
    os.makedirs(emotion_graph_dir, exist_ok=True)
    
    # List all .wav files in audio_dir
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if not wav_files:
        print(f"错误: 在 {audio_dir} 中未找到任何 .wav 文件")
        return
    
    for wav_file in wav_files:
        identifier = wav_file.replace('.wav', '')  # e.g., Ses01F_impro01_F000
        
        # Get text caption from transcription dictionary
        #import pdb; pdb.set_trace()
        if identifier in transcription_dict:
            utterance = transcription_dict[identifier]
        else:
            print(f"警告: 未找到 {identifier} 对应的转录文本，跳过")
            continue
        
        # # Load emotion label from corresponding .json file
        # json_path = os.path.join(json_dir, f"{identifier}.json")
        # if os.path.exists(json_path):
        #     try:
        #         with open(json_path, 'r', encoding='utf-8') as f:
        #             data = json.load(f)
        #             #gt_emotion = data.get('tag', 'unknown')  # Assume 'tag' field contains emotion label
        #     except Exception as e:
        #         print(f"错误: 加载 {json_path} 时出错: {e}")
        #         continue
        # else:
        #     print(f"警告: 未找到 {json_path}，跳过")
        #     continue
        
        # Full audio path
        audio_path = os.path.join(audio_dir, wav_file)
        if not os.path.exists(audio_path):
            print(f"未找到音频文件: {audio_path}")
            continue
        
        # Extract audio features
        audio_features = extract_audio_features(audio_path, utterance)
        
        # Extract keyword
        keyword = extract_keyword(utterance)
        
        # Predict sentiment (since IEMOCAP has no ground truth sentiment, use predicted sentiment)
        predicted_sentiment = get_sentiment(utterance)
        
        # Build text data using predicted sentiment (consistent with MELD logic, adapted for no ground truth sentiment)
        text_data = [
            {"id": "4", "content": utterance, "keyword": keyword, "sentiment": predicted_sentiment}
        ]
        
        # Define relationships using predicted sentiment
        relationships = [
            {"from": "1", "to": "4", "relation": get_relation(audio_features[0]['value'], predicted_sentiment)},
            {"from": "2", "to": "4", "relation": get_relation(audio_features[1]['value'], predicted_sentiment)},
            {"from": "3", "to": "4", "relation": get_relation(audio_features[2]['value'], predicted_sentiment)}
        ]
        
        # Build Emotion Graph
        emotion_graph = {
            "audio": audio_features,
            "text": text_data,
            "relationships": relationships,
            #"gt_emotion": gt_emotion  # Include ground truth emotion for reference
        }
        
        # Save to JSON file
        filename = f"emotion_graph_{identifier}.json"
        output_path = os.path.join(emotion_graph_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(emotion_graph, f, indent=4, ensure_ascii=False)
        
        print(f"已保存 Emotion Graph 到: {output_path}")

# Usage example
if __name__ == "__main__":
    transcription_dirs = [f"iemocap/transcriptions/Session{i}/transcriptions" for i in range(1, 6)]
    audio_dir = "iemocap/test_wav"
    json_dir = "iemocap/test"
    build_emotion_graph_iemocap(audio_dir, transcription_dirs, json_dir)