import os
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from google.colab import userdata

'''
This file takes from an existing features .csv file, and creates the emotion graph based on that.
It does not extract data in the loop. It also uses Groq API Llama LLM (free) to
infer cross-modal relations, because its free and efficient. If you don't have a Groq API, you need to create a .env file and put it in there.
'''


model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

pipe = pipeline("text-generation", model="Qwen/Qwen3-8B", torch_dtype="auto", device_map="cuda")







#Global var for all features, specifically for the MELD dataset.
features_df = pd.read_csv('/content/drive/MyDrive/MELD.Raw/all_features.csv')
torch.random.manual_seed(0) 




# Function to extract and classify audio features to labels on filename alone, assuming that the folders
#Are in the right place.  Returns a list of dictionary of the features, labelled from ID 1 to 7 for downstream
# Emotion Graph evaluation.
def extract_audio_features(filename):
    try:
        #this stores the row. Access the attributes through column
        row_features = features_df[features_df['filename'] == filename].iloc[0]
        #features = smile.process_file(audio_path)
        #get volume, jitter, shimmer, and intensity from smile


        #grab data from pre-loaded csv instead of extracting again
        pitch = row_features['pitch']
        loudness = row_features['loudness']
        jitter = row_features['jitter']
        shimmer = row_features['shimmer']
        intensity = row_features['intensity']
        syllables_rate = row_features['syllables_rate']
        speech_rate = row_features['speech_rate']
        
        
        #assign labels "fast", "normal", "slow", or appropiately , "high", "normal", "low" for features.
        pitch_label = "high" if pitch > 39.7 else "normal" if pitch >= 32.0 else "low"
        speech_rate_label = "fast" if speech_rate > 4.965 else "normal" if speech_rate >= 1.2 else "slow"
        jitter_label = "high" if jitter > 0.030 else "normal" if jitter >= 0.018 else "low"
        shimmer_label = "high" if shimmer > 1.306 else "normal" if shimmer >= 1.025 else "low"
        intensity_label = "high" if intensity > 0.0017 else "normal" if intensity >= 0.0007 else "low"
        syllables_label = "high" if syllables_rate > 6.287 else "normal" if syllables_rate >= 1.294 else "low"
        loudness_label = "loud" if loudness > 0.607 else "normal" if loudness >= 0.429 else "soft"


        return [
            {"id": "1", "feature": "pitch", "value": pitch_label},
            {"id": "2", "feature": "speech_rate", "value": speech_rate_label},
            {"id": "3", "feature": "jitter", "value": jitter_label},
            {"id": "4", "feature": "shimmer", "value": shimmer_label},
            {"id": "5", "feature": "intensity", "value": intensity_label},
            {"id": "6", "feature": "syllables", "value": syllables_label},
            {"id": "7", "feature": "loudness", "value": loudness_label},
        ]
    

    except Exception as e:
        print(f"Error processing audio file {filename}: {e}")
        return [
            {"id": "1", "feature": "pitch", "value": "unknown"},
            {"id": "2", "feature": "speech_rate", "value": "unknown"},
            {"id": "3", "feature": "jitter", "value": "unknown"},
            {"id": "4", "feature": "shimmer", "value": "unknown"},
            {"id": "5", "feature": "intensity", "value": "unknown"},
            {"id": "6", "feature": "syllables", "value": "unknown"},
            {"id": "7", "feature": "loudness", "value": "unknown"},
        ]

# Extract it from our all_features.csv file directly without having to recompute.
def extract_keyword(filename):
    row_features = features_df[features_df['filename'] == filename].iloc[0]

    keyword = row_features['keyword']

    #pandas doesn't have None, it has NaN
    return keyword if pd.notna(keyword) else "unknown"

# Extract it directly from the .csv file.
def get_sentiment(filename):
    row_features = features_df[features_df['filename'] == filename].iloc[0]

    return row_features['sentiment']


# Function to generate relationships using LLM, using Groq API. Need to generate a .json of 7 values
#audio_feature: pitch, volume, intensity, etc
#feature_label: low, normal, high / slow, normal, fast
def get_relation_with_llm(utterance, audio_features, sentiment):

    # prepare the model input
    prompt = f"""
    You are analyzing speech emotion recognition features.

    Utterance: "{utterance}"
    Predicted sentiment: {sentiment}

    Acoustic features:
    - pitch: {audio_features[0]['value']}
    - speech_rate: {audio_features[1]['value']}
    - jitter: {audio_features[2]['value']}
    - shimmer: {audio_features[3]['value']}
    - intensity: {audio_features[4]['value']}
    - syllables: {audio_features[5]['value']}
    - loudness: {audio_features[6]['value']}

    For each feature, does it support, contradict, or is neutral to the predicted sentiment?
    Respond ONLY with valid JSON, no explanation, one answer per feature:
    An example response:
    {{"pitch": "supports/contradicts/neutral", "speech_rate": "supports/contradicts/neutral", "jitter": "supports/contradicts/neutral", "shimmer": "supports/contradicts/neutral", "intensity": "supports/contradicts/neutral", "syllables": "supports/contradicts/neutral", "loudness": "supports/contradicts/neutral"}}
    """

    
    messages = [
        {"role": "user", "content": prompt}
    ]


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # False because we want relatively quick responses
    )
    result = pipe(text, max_new_tokens=100, do_sample=False, return_full_text=False)

    result = result[0]['generated_text'].strip() #strip hallucination wrappers
    result = result = result.replace('```json', '').replace('```', '').strip()
    #return a string, the build_emotion_graph method converts it to a .json


    return result


#Builds the full emotion graph based on files from all_features.csv and saves it to emotion_graph_dir
def build_emotion_graph(emotion_graph_dir):
    print("Building the emotion graph...")
    existing_files = os.listdir(emotion_graph_dir)
    i = 1
    pipe.model.generation_config.max_length = None #surpress the warning that max_new_tokens take precedence
    for idx, row in features_df.iterrows():
        filename = row['filename']


        #if a file is already in the emotion-graph folder, simply just skip it

        output_filename = f"emotion_graph_{filename.replace('.wav', '')}.json"
        
        output_path = os.path.join(emotion_graph_dir, output_filename)
        
        if output_filename in existing_files:
            print(f"{output_filename} has already been processed. skipping")
            continue
        


        utterance = row['Utterance']
        
        predicted_sentiment = get_sentiment(filename)
        

        audio_features = extract_audio_features(filename)
        keyword = extract_keyword(filename)


        #the text node of the paper
        text_data = [
            {
                "id": "8",
                "utterance": utterance,
                "keyword": keyword,
                "sentiment": predicted_sentiment,
            }
        ]

        #create a singular cross-modal .json
        cross_modal = get_relation_with_llm(utterance, audio_features, predicted_sentiment)

        cross_modal = cross_modal.strip()
        cross_modal = cross_modal.replace('```json', '').replace('```', '').strip() #clean it for common wrappers

        try:
            cross_modal = json.loads(cross_modal)
        except json.JSONDecodeError:
            print(f"Converting to .json failed. At {filename} ")
            cross_modal = {}

        
        #create cross-modal graph of feature to sentiment
        relationships = [
        {"from": "1", "to": "8", "relation": cross_modal.get('pitch', 'unknown')},
        {"from": "2", "to": "8", "relation": cross_modal.get('speech_rate', 'unknown')},
        {"from": "3", "to": "8", "relation": cross_modal.get('jitter', 'unknown')},
        {"from": "4", "to": "8", "relation": cross_modal.get('shimmer', 'unknown')},
        {"from": "5", "to": "8", "relation": cross_modal.get('intensity', 'unknown')},
        {"from": "6", "to": "8", "relation": cross_modal.get('syllables', 'unknown')},
        {"from": "7", "to": "8", "relation": cross_modal.get('loudness', 'unknown')},
        ]
        emotion_graph = {
            "audio": audio_features,
            "text": text_data,
            "relationships": relationships
        }
        #Replacing the .wav with .json

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(emotion_graph, f, indent=4, ensure_ascii=False)

        print(f"Saved Emotion Graph {i} to: {output_path}")
        i += 1


if __name__ == "__main__":
    csv_path = "YOUR_GRAPH_PATH"
    audio_dir = "YOUR_AUDIO_PATH"
    emotion_graph_dir = '/content/drive/MyDrive/MELD.Raw/emotion-graph-2'
    
    build_emotion_graph(emotion_graph_dir)