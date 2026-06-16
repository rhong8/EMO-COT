import os
import json
import pandas as pd
import librosa
import spacy
from transformers import pipeline
from sklearn.metrics import accuracy_score
import opensmile
pip install keybert

# Load spaCy model for keyword extraction
nlp = spacy.load('en_core_web_sm')

# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

smile = opensmile.Smile(
    features = opensmile.FeatureSet.eGeMAPSv02,
    feature_level = opensmile.FeatureLevel.Functionals,
)

features_df = pd.read_csv('/content/drive/MyDrive/MELD.Raw/acoustic_features.csv')
row_features = features_df[features_df['filename'] == file].iloc[0]

# Function to extract and classify audio features
def extract_audio_features(audio_path, utterance):
    try:
        #this stores the row. Access the attributes through column
        row_features = features_df[features_df['filename'] == audio_path].iloc[0]
        #features = smile.process_file(audio_path)
        #get volume, jitter, shimmer, and intensity from smile


        #grab mean data from pre-loaded csv instead of extracting again
        pitch = row_features['pitch']
        volume = row_features['volume']
        jitter = row_features['jitter']
        shimmer = row_features['shimmer']
        intensity = row_features['intensity']
        syllables_rate = row_features['syllables_rate']
        speech_rate = row_features['speech_rate']
        duration = row_features['duration']
        
        pitch_label = "high" if pitch > 39.7 else "normal" pitch_mean >= 32.0 else "low"

        #speech rate label
        speech_rate_label = "fast" if speech_rate > 4.965 else "normal" if speech_rate >= 1.2 else "slow"
        

        volume_label = "loud" if volume > 0.607 else "normal" if rate >= 0.429 else "soft"
        return [
            {"id": "1", "feature": "pitch", "value": pitch_label},
            {"id": "2", "feature": "speech_rate", "value": speech_rate_label},
            {"id": "3", "feature": "volume", "value": volume_label}
        ]
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return [
            {"id": "1", "feature": "pitch", "value": "unknown"},
            {"id": "2", "feature": "speech_rate", "value": "unknown"},
            {"id": "3", "feature": "volume", "value": "unknown"}
        ]

# Function to extract keywords from utterance
def extract_keyword(utterance):
    doc = nlp(utterance)
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
    return keywords[0] if keywords else "unknown"

# Function to predict sentiment polarity using Transformers model
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
            ground_truth_sentiment = "objective"  # modified

        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']

        predicted_sentiment = get_sentiment(utterance)
        predictions.append(predicted_sentiment)
        ground_truths.append(ground_truth_sentiment)

        audio_file = f"dia{dialogue_id}_utt{utterance_id}.wav"
        audio_path = os.path.join(audio_dir, audio_file)

        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
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

        print(f"Saved Emotion Graph to: {output_path}")

    if ground_truths and predictions:
        accuracy = accuracy_score(ground_truths, predictions)
        print(f"Transformer model sentiment prediction accuracy: {accuracy:.2f}")
    else:
        print("Cannot calculate accuracy: no ground truth or prediction data available")

if __name__ == "__main__":
    csv_path = "YOUR_GRAPH_PATH"
    audio_dir = "YOUR_AUDIO_PATH"
    build_emotion_graph(csv_path, audio_dir)