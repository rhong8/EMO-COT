import pandas as pd

# YOUR ALL FEATURES FILE HERE
AF = '/content/drive/MyDrive/MELD.Raw/all_features.csv'

all_features = pd.read_csv(AF)

count = (all_features['sentiment'] == 'objective').sum()
all_features.loc[all_features['sentiment'] == 'objective', 'sentiment'] = 'neutral'

all_features.to_csv(AF, index=False)
print(f"Replaced {count} 'objective' entries with 'neutral' in {AF}")
