import librosa
import numpy as np

def extract_audio_features(y, sr):
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=25), axis=1)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo if not hasattr(tempo, '__len__') else tempo[0])

    energy = np.mean(librosa.feature.rms(y=y))

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # dictionary for ui
    feature_dict = {
        "mfcc": mfcc,
        "chroma": chroma,
        "spectral_contrast": spectral_contrast,
        "tempo": tempo,
        "energy": energy,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "zcr": zcr
    }

    # vector (for model)
    feature_vector = np.concatenate([
        mfcc,
        chroma,
        spectral_contrast,
        [tempo, energy, spectral_centroid, spectral_bandwidth, zcr]
    ])

    return feature_dict, feature_vector