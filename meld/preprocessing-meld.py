
#Get the utterance based on the file name.
#Calculate the mean and std dev of each major feature, appending it to a corpus.
import opensmile
import os
import librosa
import numpy as np
import pandas as pd
import textstat
import re


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


wav_path = '/content/drive/MyDrive/MELD.Raw/output_repeated_splits_test_wav/' #YOUR WAV PATH HERE
cache_path = '/content/drive/MyDrive/MELD.Raw/acoustic_features.csv' #YOUR SAVED PATH HERE

pd.read_csv('/content/drive/MyDrive/MELD/data/MELD/test_sent_emo.csv')


def get_utterance(wav_path):
    if "dia" not in wav_path or "utt" not in wav_path:
        return None
    
    #Extract the numbers from the wav file
    numbers = re.findall(r'\d+',wav_path)
    
    dia_id = numbers[0]
    utt_id = numbers[1]
    
    
    row = df[(df['Dialogue_ID'] == numbers[0]) & (df['Utterance_ID'] == numbers[1])].iloc[0]
    utterance = row.iloc[0]['Utterance']





#Calculates the means for the MELD dataset.
def calculate_corpus_stats(wav_path, cache_path):
    pitch_total = 0
    loudness_total = 0
    jitter_total = 0
    shimmer_total = 0
    syllables_rate_total = 0
    speech_rate_total = 0
    intensity_total = 0
    file_count = 0
    results = []

    for file in os.listdir(wav_path):
        if not file.endswith('.wav'):
            continue
        utterance = df.iloc[csv_files_dict[file]]['Utterance'] #take the row num., locate it in the pds dataframe, extract the utterance
        file_path = os.path.join(wav_path, file)

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


    # calculate the meansquared difference, then divide by 2610




    
    # convert to dataframe and save to Drive once
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

    print(pitch_mean, loudness_mean, jitter_mean, shimmer_mean, intensity_mean)



    