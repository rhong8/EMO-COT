import os
import json
import pandas as pd
import librosa
import spacy
import openSMILE
from transformers import pipeline

# 加载 spaCy 模型用于关键词提取
nlp = spacy.load('en_core_web_sm')

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)



# 加载 LLM 模型
llm = pipeline("text-generation", model="facebook/bart-large-mnli")
#llm.config.pad_token_id = llm.config.eos_token_id
# 提取并分类音频特征的函数
def extract_audio_features(audio_path, utterance):
    try:
        features = smile.process_file(audio_path)
        audio, sr = librosa.load(audio_path)
        
        # 提取音高（Pitch）
        pitch, _ = librosa.piptrack(y=audio, sr=sr)
        pitch_mean = pitch.mean()
        pitch_label = "high" if pitch_mean > 100 else "low"
        
        # 提取语速（Rate）
        duration = librosa.get_duration(y=audio, sr=sr)
        word_count = len(utterance.split())
        rate = word_count / duration if duration > 0 else 0
        rate_label = "fast" if rate > 2 else "slow"
        
        # 提取音量（Volume）
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

# 使用 LLM 生成关系的函数
def generate_relation_with_llm(audio_feature, feature_value, text_content):
    prompt = (
        f"You are tasked with determining how an audio feature interacts with the emotion in a text. Follow these steps to analyze:\n\n"
        f"1. **Understand the text emotion**: Read the text '{text_content}' and identify its main emotion (e.g., happy, sad, angry, calm).\n"
        f"2. **Interpret the audio feature**: The audio feature is '{audio_feature}' with value '{feature_value}'. Use this guide:\n"
        f"   - Pitch: High = energetic or happy, Low = calm or sad.\n"
        f"   - Rate: Fast = excited or urgent, Slow = calm or thoughtful.\n"
        f"   - Volume: Loud = intense or strong, Soft = gentle or weak.\n"
        f"3. **Compare and reason**: Does the audio feature match the text’s emotion, contradict it, or have no clear effect? For example:\n"
        f"   - Happy text + High pitch = Matches (supports).\n"
        f"   - Sad text + High pitch = Contradicts (conflicts).\n"
        f"   - Calm text + Medium pitch = No strong effect (neutral).\n"
        f"4. **Answer**: Based on your reasoning, choose one word: supports, conflicts, or neutral.\n\n"
        f"Provide your answer as a single word."
    )
    # prompt = (
    #     f"Given the audio feature '{audio_feature}' with value '{feature_value}' "
    #     f"and the text '{text_content}', does the audio feature support, conflict with, or remain neutral to the emotion in the text? "
    #     "Answer with one word: supports, conflicts, or neutral."
    # )
    response = llm(prompt, max_length=300, num_return_sequences=1, temperature=0.7, top_p=0.9)
    generated_text = response[0]['generated_text'].strip().lower()
    
    if "supports" in generated_text:
        return "supports"
    elif "conflicts" in generated_text:
        return "conflicts"
    elif "neutral" in generated_text:
        return "neutral"
    else:
        return "unknown"

# 主函数：构建并保存 Emotion Graph
def build_emotion_graph(csv_path, audio_dir, emotion_graph_dir="emotion_graph_llm_wo_sent"):
    """
    根据 CSV 文件和音频文件构建 Emotion Graph，并保存到 emotion_graph 文件夹。
    
    参数:
        csv_path (str): CSV 文件路径
        audio_dir (str): 音频文件目录路径
        emotion_graph_dir (str): Emotion Graph JSON 文件保存目录，默认为 "emotion_graph"
    """
    # 确保 emotion_graph 文件夹存在
    if not os.path.exists(emotion_graph_dir):
        os.makedirs(emotion_graph_dir)
    
    # 加载 CSV 文件
    df = pd.read_csv(csv_path)
    
    # 遍历 CSV 的每一行
    for idx, row in df.iterrows():
        utterance = row['Utterance']
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        
        # 构建音频文件名
        audio_file = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        audio_path = os.path.join(audio_dir, audio_file)
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            print(f"未找到音频文件: {audio_path}")
            continue
        
        # 提取音频特征
        audio_features = extract_audio_features(audio_path, utterance)
        
        # 提取关键词
        keyword = extract_keyword(utterance)
        
        # 构建文本数据（去掉 sentiment）
        text_data = [
            {"id": "4", "content": utterance, "keyword": keyword}
        ]
        
        # 使用 LLM 定义关系
        relationships = [
            {"from": "1", "to": "4", "relation": generate_relation_with_llm("pitch", audio_features[0]['value'], utterance)},
            {"from": "2", "to": "4", "relation": generate_relation_with_llm("rate", audio_features[1]['value'], utterance)},
            {"from": "3", "to": "4", "relation": generate_relation_with_llm("volume", audio_features[2]['value'], utterance)}
        ]
        
        # 构建 Emotion Graph
        emotion_graph = {
            "audio": audio_features,
            "text": text_data,
            "relationships": relationships
        }
        
        # 保存为 JSON 文件到 emotion_graph 文件夹
        filename = f"emotion_graph_dia{dialogue_id}_utt{utterance_id}.json"
        output_path = os.path.join(emotion_graph_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(emotion_graph, f, indent=4, ensure_ascii=False)
        
        print(f"已保存 Emotion Graph 到: {output_path}")

# 使用示例
if __name__ == "__main__":
    csv_path = "/data/jiacheng/Qwen2-Audio/meld/MELD.Raw/test_sent_emo.csv"
    audio_dir = "/data/jiacheng/Qwen2-Audio/meld/MELD.Raw/output_repeated_splits_test_wav"
    build_emotion_graph(csv_path, audio_dir)