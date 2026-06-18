import os
import json
import pandas as pd
import librosa
import spacy
import openSMILE
from transformers import pipeline

# Load spaCy model for keyword extraction
nlp = spacy.load('en_core_web_sm')

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)



# Load LLM model
llm = pipeline("text-generation", model="facebook/bart-large-mnli")
#llm.config.pad_token_id = llm.config.eos_token_id
# Function to extract and classify audio features
def extract_audio_features(audio_path, utterance):
    try:
        features = smile.process_file(audio_path)
        audio, sr = librosa.load(audio_path)
        
        # Extract pitch
        pitch, _ = librosa.piptrack(y=audio, sr=sr)
        pitch_mean = pitch.mean()
        pitch_label = "high" if pitch_mean > 100 else "low"
        
        # Extract speech rate
        duration = librosa.get_duration(y=audio, sr=sr)
        word_count = len(utterance.split())
        rate = word_count / duration if duration > 0 else 0
        rate_label = "fast" if rate > 2 else "slow"
        
        # Extract volume
        volume = librosa.feature.rms(y=audio).mean()
        volume_label = "loud" if volume > 0.1 else "soft"
        
        return [
            {"id": "1", "feature": "pitch", "value": pitch_label},
            {"id": "2", "feature": "rate", "value": rate_label},
            {"id": "3", "feature": "volume", "value": volume_label}
        ]
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
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

# Function to generate relationships using LLM
def generate_relation_with_llm(audio_feature, feature_value, text_content):
    prompt = (
        f"You are tasked with determining how an audio feature interacts with the emotion in a text. Follow these steps to analyze:\n\n"
        f"1. **Understand the text emotion**: Read the text '{text_content}' and identify its main emotion (e.g., happy, sad, angry, calm).\n"
        f"2. **Interpret the audio feature**: The audio feature is '{audio_feature}' with value '{feature_value}'. Use this guide:\n"
        f"   - Pitch: High = energetic or happy, Low = calm or sad.\n"
        f"   - Rate: Fast = excited or urgent, Slow = calm or thoughtful.\n"
        f"   - Volume: Loud = intense or strong, Soft = gentle or weak.\n"
        f"3. **Compare and reason**: Does the audio feature match the text's emotion, contradict it, or have no clear effect? For example:\n"
        f"   - Happy text + High pitch = Matches (supports).\n"
        f"   - Sad text + High pitch = Contradicts (conflicts).\n"
        f"   - Calm text + Medium pitch = No strong effect (neutral).\n"
        f"4. **Answer**: Based on your reasoning, choose one word: supports, conflicts, or neutral.\n\n"
        f"Provide your answer as a single word."
    )
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

# Main function: build and save Emotion Graph
def build_emotion_graph(csv_path, audio_dir, emotion_graph_dir="emotion_graph_llm_wo_sent"):
    """
    Build Emotion Graph from CSV and audio files and save to emotion_graph folder.
    
    Args:
        csv_path (str): Path to CSV file
        audio_dir (str): Path to audio files directory
        emotion_graph_dir (str): Directory to save Emotion Graph JSON files, default is "emotion_graph"
    """
    # Ensure emotion_graph folder exists
    if not os.path.exists(emotion_graph_dir):
        os.makedirs(emotion_graph_dir)
    
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Iterate over each row in CSV
    for idx, row in df.iterrows():
        utterance = row['Utterance']
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        
        # Build audio filename
        audio_file = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        audio_path = os.path.join(audio_dir, audio_file)
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue
        
        # Extract audio features
        audio_features = extract_audio_features(audio_path, utterance)
        
        # Extract keyword
        keyword = extract_keyword(utterance)
        
        # Build text data (without sentiment)
        text_data = [
            {"id": "4", "content": utterance, "keyword": keyword}
        ]
        
        # Define relationships using LLM
        relationships = [
            {"from": "1", "to": "4", "relation": generate_relation_with_llm("pitch", audio_features[0]['value'], utterance)},
            {"from": "2", "to": "4", "relation": generate_relation_with_llm("rate", audio_features[1]['value'], utterance)},
            {"from": "3", "to": "4", "relation": generate_relation_with_llm("volume", audio_features[2]['value'], utterance)}
        ]
        
        # Build Emotion Graph
        emotion_graph = {
            "audio": audio_features,
            "text": text_data,
            "relationships": relationships
        }
        
        # Save as JSON file to emotion_graph folder
        filename = f"emotion_graph_dia{dialogue_id}_utt{utterance_id}.json"
        output_path = os.path.join(emotion_graph_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(emotion_graph, f, indent=4, ensure_ascii=False)
        
        print(f"Saved Emotion Graph to: {output_path}")

# Example usage
if __name__ == "__main__":
    csv_path = "/data/jiacheng/Qwen2-Audio/meld/MELD.Raw/test_sent_emo.csv"
    audio_dir = "/data/jiacheng/Qwen2-Audio/meld/MELD.Raw/output_repeated_splits_test_wav"
    build_emotion_graph(csv_path, audio_dir)