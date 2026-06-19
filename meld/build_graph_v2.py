import os
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline



'''
This file takes from an existing features .csv file, and creates the emotion graph based on that.
It does not extract data in the loop. It also uses Groq API Llama LLM (free) to
infer cross-modal relations, because its free and efficient. If you don't have a Groq API, you need to create a .env file and put it in there.
'''


#Intended for GROQ API Usage
'''

try:
    os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
except:
    pass  # Running locally, .env handles it
'''






#Global var for all features, specifically for the MELD dataset.
features_df = pd.read_csv('/content/drive/MyDrive/MELD.Raw/all_features.csv')
torch.random.manual_seed(0) 


model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

#load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)

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


# Function to generate relationships using LLM
#audio_feature: pitch, volume, intensity, etc.
#feature_label: low, normal, high / slow, normal, fast
def get_relation_with_llm(utterance, audio_feature, feature_label, sentiment):

    prompt = (
        
        f"<|user|>\nYou are tasked with determining how an audio feature interacts with the emotion in a text. Follow these steps to analyze:\n\n"
        f"1. **Understand the text emotion**: Read the text '{utterance}'. Its sentiment is {sentiment}. \n"
        f"2. **Interpret the audio feature**: The audio feature is '{audio_feature}' with value '{feature_label}'. Use this guide:\n"
        f"   - Pitch: High = energetic or happy, Low = calm or sad.\n"
        f"   - Rate: Fast = excited or urgent, Slow = calm or thoughtful.\n"
        f"   - Volume: Loud = intense or strong, Soft = gentle or weak.\n"
        f"3. **Compare and reason**: Does the audio feature match the text's emotion, contradict it, or have no clear effect? For example:\n"
        f"   - Happy text + High pitch = Matches (supports).\n"
        f"   - Sad text + High pitch = Contradicts (conflicts).\n"
        f"   - Calm text + Medium pitch = No strong effect (neutral).\n"
        f"4. **Answer**: Based on your reasoning, choose one word: supports, conflicts, or neutral.\n\n"
        f"Provide your answer as a single word. <|end|>\n<|assistant|>\n"
    )

    #deterministic outcome, always pick the highest likelihood answer
    response = pipe(prompt, max_new_tokens = 40, num_return_sequences=1, do_sample = False, truncation = True)
    generated_text = response[0]['generated_text'][len(prompt):].strip().lower() #grab the text after the prompt
    
    if "supports" in generated_text:
        return "supports"
    elif "conflicts" in generated_text:
        return "conflicts"
    elif "neutral" in generated_text:
        return "neutral"
    else:
        return "unknown"


#Builds the full emotion graph based on files from all_features.csv and saves it to emotion_graph_dir
def build_emotion_graph(emotion_graph_dir):
    print("Building the emotion graph...")


    for idx, row in features_df.iterrows():
        filename = row['filename']
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

        #create cross-modal graph of feature to sentiment
        relationships = [
            {"from": "1", "to": "8", "relation": get_relation_with_llm(utterance, audio_features[0]['feature'], audio_features[0]['value'], predicted_sentiment)},
            {"from": "2", "to": "8", "relation": get_relation_with_llm(utterance, audio_features[1]['feature'], audio_features[1]['value'], predicted_sentiment)},
            {"from": "3", "to": "8", "relation": get_relation_with_llm(utterance, audio_features[2]['feature'], audio_features[2]['value'], predicted_sentiment)},
            {"from": "4", "to": "8", "relation": get_relation_with_llm(utterance, audio_features[3]['feature'], audio_features[3]['value'], predicted_sentiment)},
            {"from": "5", "to": "8", "relation": get_relation_with_llm(utterance, audio_features[4]['feature'], audio_features[4]['value'], predicted_sentiment)},
            {"from": "6", "to": "8", "relation": get_relation_with_llm(utterance, audio_features[5]['feature'], audio_features[5]['value'], predicted_sentiment)},
            {"from": "7", "to": "8", "relation": get_relation_with_llm(utterance, audio_features[6]['feature'], audio_features[6]['value'], predicted_sentiment)},
        ]

        emotion_graph = {
            "audio": audio_features,
            "text": text_data,
            "relationships": relationships
        }
        #Replacing the .wav with .json
        output_filename = f"emotion_graph_{filename.replace('.wav', '')}.json"
        
        output_path = os.path.join(emotion_graph_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(emotion_graph, f, indent=4, ensure_ascii=False)

        print(f"Saved Emotion Graph to: {output_path}")


if __name__ == "__main__":
    csv_path = "YOUR_GRAPH_PATH"
    audio_dir = "YOUR_AUDIO_PATH"
    emotion_graph_dir = '/content/drive/MyDrive/MELD.Raw/emotion-graph'
    
    build_emotion_graph(emotion_graph_dir)