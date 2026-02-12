from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import pandas as pd

def build_timegrid(num_frames: int, fps: int) -> np.ndarray:
    """Return center time (sec) for each sampled frame index."""
    return np.arange(num_frames, dtype=np.float32) / float(fps)

def slice_mel_by_time(mel: torch.Tensor, sr: int, hop_s: float, t_start: float, t_end: float, target_length: Optional[int] = None) -> torch.Tensor:
    """Slice mel [1,M,T] by time range (seconds) and pad/truncate to exact length.
    
    Args:
        mel: Mel spectrogram tensor [1, M, T]
        sr: Sample rate (unused, kept for compatibility)
        hop_s: Hop length in seconds
        t_start: Start time in seconds
        t_end: End time in seconds
        target_length: Target length for the output. If None, returns the slice as-is.
                      If specified, pads with zeros or truncates to this length.
    
    Returns:
        Sliced (and optionally padded/truncated) mel spectrogram [1, M, target_length]
    """
    # hop_s is the time per hop (sec). Index range:
    i0 = int(np.floor(t_start / hop_s))
    i1 = int(np.ceil(t_end   / hop_s))
    i0 = max(i0, 0)
    i1 = min(i1, mel.shape[-1])
    sl = mel[..., i0:i1]
    
    # Pad or truncate to target length if specified
    if target_length is not None:
        current_length = sl.shape[-1]
        if current_length < target_length:
            # Pad with zeros
            pad_amount = target_length - current_length
            sl = torch.nn.functional.pad(sl, (0, pad_amount), mode='constant', value=0)
        elif current_length > target_length:
            # Truncate
            sl = sl[..., :target_length]
    
    return sl

def make_windows(diff_stacks: List[torch.Tensor], mel: torch.Tensor, fps: int, 
                 win_sec: float, stride_sec: float, mel_hop_s: float) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
    """Create sliding windows from video frames and mel spectrograms.
    
    Args:
        diff_stacks: List of frame difference tensors [3, H, W]
        mel: Mel spectrogram tensor [1, M, T]
        fps: Frames per second
        win_sec: Window size in seconds
        stride_sec: Stride between windows in seconds
        mel_hop_s: Hop length in seconds for mel spectrogram
    
    Returns:
        Tuple of (video_windows, mel_windows, window_centers)
        - video_windows: List of video frame tensors
        - mel_windows: List of mel spectrogram tensors
        - window_centers: List of center times for each window
    """
    num_frames = len(diff_stacks)
    timegrid = build_timegrid(num_frames, fps)
    
    # Calculate window parameters
    win_frames = int(win_sec * fps)
    stride_frames = int(stride_sec * fps)
    
    # Calculate target mel length based on window duration
    target_mel_length = int(np.ceil(win_sec / mel_hop_s))
    
    vids, mels, centers = [], [], []
    
    # Create sliding windows
    for start_idx in range(0, num_frames - win_frames + 1, stride_frames):
        end_idx = start_idx + win_frames
        
        # Get center time of this window
        t_start = timegrid[start_idx]
        t_end = timegrid[end_idx - 1] if end_idx - 1 < len(timegrid) else timegrid[-1]
        t_center = (t_start + t_end) / 2.0
        
        # Extract video window (use the middle frame or last frame of the window)
        vid_window = diff_stacks[end_idx - 1]
        
        # Extract mel window with consistent target length
        mel_window = slice_mel_by_time(mel, sr=0, hop_s=mel_hop_s, t_start=t_start, t_end=t_end, target_length=target_mel_length)
        
        vids.append(vid_window)
        mels.append(mel_window)
        centers.append(t_center)
    
    return vids, mels, centers

def label_windows(window_centers: List[float], ann_df: pd.DataFrame, horizon_sec: float = 1.0) -> torch.Tensor:
    """Binary label = 1 if an 'explosion' event starts within horizon after window end.
    Expected columns in ann_df: video_id, start_sec, end_sec, event_type
    """
    starts = ann_df[ann_df['event_type']=='explosion']['start_sec'].values.astype(float)
    labels = []
    for t_end in window_centers:
        # if any start in (t_end, t_end + horizon], label 1
        labels.append(1 if np.any((starts > t_end) & (starts <= t_end + horizon_sec)) else 0)
    return torch.tensor(labels, dtype=torch.long)
