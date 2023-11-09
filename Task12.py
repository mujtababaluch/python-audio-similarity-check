import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def compare_audio_files(file_path1, file_path2, n_subbands=20, n_neighbors=5, subsegment_duration=5):
    # Load audio files
    audio1, sr1 = librosa.load(file_path1, sr=None)
    audio2, sr2 = librosa.load(file_path2, sr=None)

    # Determine which audio is longer
    if len(audio1) > len(audio2):
        longer_audio, shorter_audio = audio1, audio2
        sr_long, sr_short = sr1, sr2
    else:
        longer_audio, shorter_audio = audio2, audio1
        sr_long, sr_short = sr2, sr1

    # Compute mel spectrograms for the longer audio
    stft_long = librosa.stft(longer_audio)
    subbands_long = librosa.amplitude_to_db(librosa.feature.melspectrogram(S=stft_long, n_mels=n_subbands**2, fmin=0, fmax=sr_long/2), ref=np.max)

    # Reshape the subbands to be a 2D matrix
    subbands_long = subbands_long.reshape(-1, subbands_long.shape[-1])

    # Standardize the feature vectors
    scaler = StandardScaler()
    subbands_long = scaler.fit_transform(subbands_long)

    # Fit the KNN classifier
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(subbands_long, np.zeros(subbands_long.shape[0]))

    # Compute the number of subsegments in the longer audio
    n_subsegments = int(np.ceil(len(longer_audio) / (subsegment_duration * sr_long)))

    # Initialize an array to store the similarity scores for each subsegment
    similarity_scores = np.zeros(n_subsegments)

    # Compare subsegments of the longer audio with the shorter audio
    for i in range(n_subsegments):
        # Determine the start and end time of the subsegment
        start_time = i * subsegment_duration
        end_time = min((i + 1) * subsegment_duration, len(longer_audio))

        # Extract the subsegment and compute its mel spectrogram
        subsegment = longer_audio[start_time:end_time]
        stft_sub = librosa.stft(subsegment)
        subbands_sub = librosa.amplitude_to_db(librosa.feature.melspectrogram(S=stft_sub, n_mels=n_subbands**2, fmin=0, fmax=sr_long/2), ref=np.max)

        # Reshape the subbands to be a 2D matrix
        subbands_sub = subbands_sub.reshape(-1, subbands_sub.shape[-1])

        # Standardize the feature vectors
        subbands_sub = scaler.transform(subbands_sub)

        # Predict the labels for the subsegment
        distances, indices = classifier.kneighbors(subbands_sub, n_neighbors=n_neighbors, return_distance=True)

        # Compute the similarity score as the average distance between the nearest neighbors
        similarity_scores[i] = np.mean(distances)

    # Compute the overall similarity score as the average similarity score across all subsegments
    overall_similarity_score = np.mean(similarity_scores)

    return overall_similarity_score