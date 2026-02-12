import numpy as np
import torch
from typing import Tuple
import librosa

# MoviePy-based audio extraction
from moviepy.editor import VideoFileClip

def extract_audio_array(path: str, sr: int = 16000) -> np.ndarray:
    """Return mono audio as float32 numpy array at target sr using MoviePy."""
    clip = VideoFileClip(path)
    # MoviePy returns audio at its native sampling rate; resample with librosa
    audioclip = clip.audio
    if audioclip is None:
        return np.zeros(1, dtype=np.float32)
    # to_soundarray can be large; for short clips (classroom) it's okay
    # Fix for numpy compatibility: convert generator to list
    try:
        audio = audioclip.to_soundarray(fps=sr)  # shape [T, channels] at target fps
    except TypeError:
        # Fallback for newer numpy versions that require sequence types
        chunks = list(audioclip.iter_chunks(fps=sr, quantize=True, nbytes=2, chunksize=2000))
        audio = np.vstack(chunks) if chunks else np.zeros((1, 2), dtype=np.int16)
        # Convert from int16 to float32 in range [-1, 1]
        audio = audio.astype(np.float32) / 32768.0
    
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)

def log_mel_spectrogram(audio: np.ndarray, sr: int = 16000, n_mels: int = 64,
                        win_length: float = 0.025, hop_length: float = 0.010) -> torch.Tensor:
    """Compute log-mel spectrogram [1, Mels, T]."""
    n_fft = int(sr * win_length)
    hop = int(sr * hop_length)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
    logS = librosa.power_to_db(S + 1e-10)
    logS = torch.from_numpy(logS).unsqueeze(0).float()   # [1,M,T]
    return logS
