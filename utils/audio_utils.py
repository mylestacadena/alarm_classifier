import numpy as np
import librosa

def extract_mfcc(audio, sr, n_mfcc=13, max_pad_len=100):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc.T[np.newaxis, ...].astype(np.float32)  # (1, 100, 13)
