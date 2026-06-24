
import opensmile
import os
import librosa
import numpy as np
import pandas as pd
import textstat
import re
import subprocess
import json


from keybert import KeyBERT
from transformers import pipeline


sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

kw_model = KeyBERT()

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

#YOUR GROUND TRUTH DATA HERE
ground_truth = 'MELD/data/MELD/test_sent_emo.csv'

#YOUR ALL FEATURES FILE HERE (ONLY FOR DEBUGGING PURPOSES)
AF = '/content/drive/MyDrive/MELD.Raw/all_features.csv'


df = pd.read_csv(ground_truth)

all_features = pd.read_csv(AF)


#Get the sentiment from the utterance
def get_sentiment(utterance):
    result = sentiment_analyzer(utterance)[0]
    label = result['label'].lower()
    if label == 'label_0':
        return "negative"
    elif label == 'label_1':
        return "objective"
    elif label == 'label_2':
        return "positive"


def get_keyword(utterance):

    keywords = kw_model.extract_keywords(utterance, keyphrase_ngram_range=(1, 1), stop_words=None)
    keyword = keywords[0][0]

    return keyword if keywords else "unknown"


def convert_to_wav(audio_dir, wav_dir):
    os.makedirs(wav_dir, exist_ok=True)

    for filename in os.listdir(audio_dir):
        if filename.startswith('._'):
            continue

        if filename.endswith('.mp4'):
            mp4_path = os.path.join(audio_dir, filename)
            wav_path = os.path.join(wav_dir, filename.replace('.mp4', '.wav'))

            if not os.path.exists(wav_path):
                subprocess.run(
                    ['ffmpeg', '-i', mp4_path, '-ar', '16000', '-ac', '1', wav_path, '-y', '-loglevel', 'error'],
                    check=True
                )

    print("Done converting files.")


#Returns a set of valid files to later access. Excludes files with the wrong naming convention or not .wav files
def clean_files(wav_dir):
    valid_files = set()
    
    for file in os.listdir(wav_dir):     
        if not file.endswith('wav') or not re.match(r'^dia\d+_utt\d+\.wav$', file):
            continue
        
        numbers = re.findall(r'\d+', file)
        dia_id = int(numbers[0])
        utt_id = int(numbers[1])
        
        match = df[(df['Dialogue_ID'] == dia_id) & (df['Utterance_ID'] == utt_id)]
        if match.empty:
            continue
        
        valid_files.add(file)
    
    return valid_files


#Checks files with the valid format, but are not in the validation table
def extraneous_files(wav_dir):
    final_files = []
    extraneous_files = []
    csv_files_dict = dict()

    for idx, row in df.iterrows():
        file_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
        csv_files_dict[file_name] = row

    count_missing = 0

    for f in os.listdir(wav_dir):
        if f.startswith('final'):
            final_files.append(f)
        elif f.startswith('dia'):
            if f not in csv_files_dict:
                count_missing += 1
                extraneous_files.append(f)
        else:
            extraneous_files.append(f)

    print(len(final_files))
    print(final_files)
    print(extraneous_files)
    print("Files that were found in the wav directory but not in the .csv: ", count_missing)
    #print(csv_files_dict)


def get_utterance(wav_path):
    if "dia" not in wav_path or "utt" not in wav_path:
        return None
    
    numbers = re.findall(r'\d+', wav_path)
    
    dia_id = int(numbers[0])
    utt_id = int(numbers[1])
    
    row = df[(df['Dialogue_ID'] == dia_id) & (df['Utterance_ID'] == utt_id)].iloc[0]

    return row['Utterance']

#Calculates the 7 acoustic features for each audio file and saves it to cache_path.
#Calculates the mean and std dev of the entire corpus and saves it to .json path
def calculate_corpus_stats(wav_dir, cache_path, json_path):
    
    valid_files = clean_files(wav_dir)
    
    pitch_total = 0
    loudness_total = 0
    jitter_total = 0
    shimmer_total = 0
    syllables_rate_total = 0
    speech_rate_total = 0
    intensity_total = 0
    file_count = 0
    results = []

    for file in valid_files:
        if not file.endswith('.wav'):
            continue
        
        file_path = os.path.join(wav_dir, file)
        utterance = get_utterance(file_path)

        audio, sr = librosa.load(file_path, sr=None)
        features = smile.process_file(file_path)
        duration = librosa.get_duration(y=audio, sr=sr)

        pitch = features['F0semitoneFrom27.5Hz_sma3nz_amean'].values[0]
        loudness = features['loudness_sma3_amean'].values[0]
        jitter = features['jitterLocal_sma3nz_amean'].values[0]
        shimmer = features['shimmerLocaldB_sma3nz_amean'].values[0]
        intensity = np.mean(audio**2)
        syllables_rate = textstat.syllable_count(utterance) / duration
        speech_rate = textstat.lexicon_count(utterance) / duration

        pitch_total += pitch
        loudness_total += loudness
        jitter_total += jitter
        shimmer_total += shimmer
        intensity_total += intensity
        syllables_rate_total += syllables_rate
        speech_rate_total += speech_rate


        results.append({
        'filename': file,
        'pitch': pitch,
        'loudness': loudness,
        'jitter': jitter,
        'shimmer': shimmer,
        'intensity': intensity,
        'syllables_rate': syllables_rate,
        'speech_rate': speech_rate,
        'duration': duration
        })


        if file_count % 100 == 99:
            print(f"Processing file {file_count + 1}")
        file_count += 1

    results_df = pd.DataFrame(results)
    results_df.to_csv(cache_path, index=False)
    print(f"Saved {file_count} files to {cache_path}")

    pitch_mean = pitch_total / file_count  
    loudness_mean = loudness_total / file_count
    jitter_mean = jitter_total / file_count
    shimmer_mean = shimmer_total / file_count
    intensity_mean = intensity_total / file_count
    syllables_rate_mean = syllables_rate_total / file_count
    speech_rate_mean = speech_rate_total / file_count

    pitch_sqdiff = 0
    loudness_sqdiff = 0
    jitter_sqdiff = 0
    shimmer_sqdiff = 0
    intensity_sqdiff = 0
    syllables_sqdiff = 0
    speech_rate_sqdiff = 0

    print("Now calculating the std. dev..")
    file_count = 0
    #directly access the stats from the results array, not re-loading the audio files
    for r in results:
        pitch_sqdiff += (r['pitch'] - pitch_mean) ** 2
        loudness_sqdiff += (r['loudness'] - loudness_mean) ** 2
        jitter_sqdiff += (r['jitter'] - jitter_mean) ** 2
        shimmer_sqdiff += (r['shimmer'] - shimmer_mean) ** 2
        intensity_sqdiff += (r['intensity'] - intensity_mean) ** 2
        syllables_sqdiff += (r['syllables_rate'] - syllables_rate_mean) ** 2
        speech_rate_sqdiff += (r['speech_rate'] - speech_rate_mean) ** 2
        
        if file_count % 100 == 99:
            print(f"Processing file {file_count + 1}")
        file_count += 1

    pitch_std = (pitch_sqdiff / 2610) ** 0.5
    loudness_std = (loudness_sqdiff / 2610) ** 0.5
    jitter_std = (jitter_sqdiff / 2610) ** 0.5
    shimmer_std = (shimmer_sqdiff / 2610) ** 0.5
    intensity_std = (intensity_sqdiff / 2610) ** 0.5
    syllables_std = (syllables_sqdiff / 2610) ** 0.5
    speech_rate_std = (speech_rate_sqdiff / 2610) ** 0.5

    stats = {
        'pitch': {'mean': float(pitch_mean), 'std': float(pitch_std)},
        'loudness': {'mean': float(loudness_mean), 'std': float(loudness_std)},
        'jitter': {'mean': float(jitter_mean), 'std': float(jitter_std)},
        'shimmer': {'mean': float(shimmer_mean), 'std': float(shimmer_std)},
        'intensity': {'mean': float(intensity_mean), 'std': float(intensity_std)},
        'syllables': {'mean': float(syllables_rate_mean), 'std': float(syllables_std)},
        'speech_rate': {'mean': float(speech_rate_mean), 'std': float(speech_rate_std)},
    }

    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)


#Calculates the sentiment and keyword using RoBERTa and keyword using KeyBERT, saving it to the processed cache_path.

def calculate_sentiment_keyword(wav_dir, cache_path):
    valid_files = clean_files(wav_dir)
    i = 0
    
    for file in valid_files:
        if not file.endswith('.wav'):
            continue
        
        file_path = os.path.join(wav_dir, file)
        utterance = get_utterance(file_path)
        sentiment = get_sentiment(utterance)
        keyword = get_keyword(utterance)
        all_features.loc[all_features['filename'] == file, 'sentiment'] = sentiment
        all_features.loc[all_features['filename'] == file, 'keyword'] =  keyword
        
        if i % 99 == 0:
            print(f"Processed keyword and sentiment for file {i+1}..")
        i += 1
    try:
        all_features.to_csv(cache_path, index = False)
    except Exception as e:
        print(f"An exception occured in saving. {e}")
    


    


#assuming the all_features is loaded
def calculate_individual_file(filename):
    
    full_path = os.path.join('/content/drive/MyDrive/MELD.Raw/output_repeated_splits_test_wav/', filename)

    if not os.path.exists(full_path):
        print(f"Error. {full_path} does not exist.")
        return
    

    #utterance = get_utterance(full_path)
    utterance = all_features[all_features['filename'] == filename].iloc[0]['Utterance']

    audio, sr = librosa.load(full_path, sr=None)
    features = smile.process_file(full_path)
    duration = librosa.get_duration(y=audio, sr=sr)

    pitch = features['F0semitoneFrom27.5Hz_sma3nz_amean'].values[0]
    loudness = features['loudness_sma3_amean'].values[0]
    jitter = features['jitterLocal_sma3nz_amean'].values[0]
    shimmer = features['shimmerLocaldB_sma3nz_amean'].values[0]
    intensity = np.mean(audio**2)
    syllables_rate = textstat.syllable_count(utterance) / duration
    speech_rate = textstat.lexicon_count(utterance) / duration

    print(f"""
    File: {filename}
    Utterance: {utterance}
    Duration: {duration:.2f}s

    Pitch:        {pitch:.4f}
    Loudness:     {loudness:.4f}
    Jitter:       {jitter:.4f}
    Shimmer:      {shimmer:.4f}
    Intensity:    {intensity:.6f}
    Syllables/s:  {syllables_rate:.4f}
    Speech rate:  {speech_rate:.4f}
    """)

    

if __name__ == '__main__':
    audio_dir = '/content/drive/MyDrive/MELD.Raw/output_repeated_splits_test' #YOUR RAW DATA PATH HERE
    wav_dir = '/content/drive/MyDrive/MELD.Raw/output_repeated_splits_test_wav/' #YOUR DATA, IN WAV FILE PATH HERE
    json_path = '/content/drive/MyDrive/MELD.Raw/corpus_stats_v2.json' #YOUR .JSON SAVE PATH HERE
    cache_path = '/content/drive/MyDrive/MELD.Raw/all_features.csv' #YOUR FEATURE SAVE PATH HERE

    #USE WHEN CALCULATING STATISTICAL VALUES FOR CORPUS
    #calculate_corpus_stats(wav_dir, cache_path, json_path)
    
    
    #USE WHEN CALCULATING SENTIMENT AND KEYWORD
    calculate_sentiment_keyword(wav_dir, cache_path)