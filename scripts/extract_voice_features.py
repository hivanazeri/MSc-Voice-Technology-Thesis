import os
import csv
import pandas as pd
import numpy as np
import librosa
from os.path import join
import argparse
import parselmouth

# Argument parsing
parser = argparse.ArgumentParser(description="Extract features from Mozilla Common Voice audio files.")
parser.add_argument('--lang', type=str, default="en", help="Language code, e.g., 'en'")
args = parser.parse_args()

# Path variables
LANG = args.lang
DATA_DIR = f"/scratch/s6028497/Thesis/data/cv-corpus-21.0-2025-03-14/{LANG}"
AUDIO_DIR = "/scratch/s6028497/Thesis/data/cv-corpus-21.0-2025-03-14/en/clips"
FEATURE_DIR = "/scratch/s6028497/Thesis/data/cv-corpus-21.0-2025-03-14/en/"

# Ensure feature folder exists
os.makedirs(FEATURE_DIR, exist_ok=True)

# Get gender from TSV row
def get_gender(df, path):
    file_name = path.split("/")[-1].replace(".mp3", "")
    gender_row = df[df['path'].str.contains(file_name)]
    if gender_row.empty:
        return -1
    gender_value = gender_row.iloc[0]['gender']
    return 0 if gender_value == 'male' else 1 if gender_value == 'female' else -1


# Filter the Age of the data
# Load metadata TSV
tsv_path = join(DATA_DIR, "validated.tsv")
df = pd.read_csv(tsv_path, sep="\t")

# Convert age to numeric (if it's not already) and filter
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df[df['age'].between(18, 80)]

# Drop rows with missing paths just in case
df = df.dropna(subset=['path'])

# Loop through filtered files
feature_list = []
for idx, row in df.iterrows():
    file_path = join(AUDIO_DIR, row['path'])
    try:
        features = feature_extraction(file_path, df)
        feature_list.append([row['path']] + features.tolist())
    except Exception as e:
        print(f"Error processing {file_path}: {e}")




def feature_extraction(path, df, sampling_rate=48000):
    features = []
    audio, sr = librosa.load(path, sr=sampling_rate)
    gender = get_gender(df, path)

    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    rms = np.mean(librosa.feature.rms(y=audio))

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_means = [np.mean(coeff) for coeff in mfcc]

    # Pitch (F0)
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_mean = np.nanmean(f0) if not np.isnan(f0).all() else 0
    f0_std = np.nanstd(f0) if not np.isnan(f0).all() else 0

    # Jitter (approximation)
    jitter = f0_std / f0_mean if f0_mean != 0 else 0

    # Shimmer (RMS variation as approximation)
    frame_length = 2048
    hop_length = 512
    rms_vals = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    shimmer = np.std(rms_vals) / np.mean(rms_vals) if np.mean(rms_vals) != 0 else 0

    # Spectral tilt (linear regression over log power spectrum)
    stft = np.abs(librosa.stft(audio))**2
    log_spectrum = librosa.power_to_db(stft)
    spectral_tilt = np.polyfit(range(log_spectrum.shape[0]), np.mean(log_spectrum, axis=1), 1)[0]

    # Speech rate (syllables per second approximation)
    duration = librosa.get_duration(y=audio, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    speech_rate = tempo / 60.0  # beats per second as proxy for syllables/sec

    # Formants using parselmouth
    snd = parselmouth.Sound(path)
    formant = snd.to_formant_burg()
    try:
        f1 = np.mean([formant.get_value_at_time(1, t) for t in np.linspace(0, snd.duration, 100)])
        f2 = np.mean([formant.get_value_at_time(2, t) for t in np.linspace(0, snd.duration, 100)])
        f3 = np.mean([formant.get_value_at_time(3, t) for t in np.linspace(0, snd.duration, 100)])
    except Exception as e:
        print(f"Formant error in {path}: {e}")
        f1, f2, f3 = 0, 0, 0

    # Compile all features
    features.extend([
        gender,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zcr,
        rms,
        *mfcc_means,
        f0_mean,
        f0_std,
        jitter,
        shimmer,
        spectral_tilt,
        speech_rate,
        f1,
        f2,
        f3
    ])
    return np.array(features)


def create_header():
    # Define header
    header = ['path', 'gender', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zcr', 'rms'] + \
            [f'mfcc_{i+1}' for i in range(20)] + \
            ['f0_mean', 'f0_std', 'jitter', 'shimmer', 'spectral_tilt', 'speech_rate', 'formant1', 'formant2', 'formant3']





# Create features CSV for a given TSV
def create_feature_csv(tsv_file):
    df = pd.read_csv(join(DATA_DIR, tsv_file), sep='\t', dtype={'age': str})  # read age as string to avoid dtype warnings
    df = df[df['age'].notna()]  # 🔥 Filter only rows where age is not NaN

    output_csv_path = join(FEATURE_DIR, tsv_file.replace(".tsv", ".csv"))

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(create_header())

    for idx, row in df.iterrows():
        path = join(AUDIO_DIR, row["path"])
        if not os.path.exists(path):
            continue
        try:
            features = feature_extraction(path, df)
            row_out = [row["path"]] + list(features) + [row["age"]]
            with open(output_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_out)
        except Exception as e:
            print(f"Error processing {row['path']}: {e}")

# Main execution
if __name__ == "__main__":
    for split in ["train.tsv", "dev.tsv", "test.tsv"]:
        print(f"Processing {split}...")
        create_feature_csv(split)

    df_combined = pd.concat([
        pd.read_csv(join(FEATURE_DIR, "train.csv")),
        pd.read_csv(join(FEATURE_DIR, "dev.csv")),
        pd.read_csv(join(FEATURE_DIR, "test.csv"))
    ])

    # Final check for NaNs in age just in case
    df_combined = df_combined[df_combined['age'].notna()]

    print("✅ Combined shape:", df_combined.shape)
    print(df_combined.head())
