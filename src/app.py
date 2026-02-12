import io, time
import numpy as np
import torch
from PIL import Image
import streamlit as st
import tempfile, os
from models.explex_net import ExplexNet
from helpers.decode_video import sample_frames, frame_diff_stack, get_video_info
from helpers.audio import extract_audio_array, log_mel_spectrogram
from helpers.windows import make_windows

st.set_page_config(page_title="Explosion or Explanation", page_icon="💥" )
st.title("💥 Explosion or Explanation — Science Docu-Drama Detector")

@st.cache_resource
def load_model(weights_path: str = "weights/best.pt"):
    model = ExplexNet()
    try:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    except Exception:
        st.warning("Weights file not found or incompatible — using randomly initialized model.")
    model.eval()
    return model

model = load_model()

video_file = st.file_uploader("Upload a short MP4 clip (30–120s)", type=["mp4"])
fps = st.slider("Sampling FPS (video)", 2, 8, 4, 1)
win_sec = st.slider("Window (sec)", 1.0, 4.0, 2.0, 0.5)
stride_sec = st.slider("Stride (sec)", 0.2, 2.0, 0.5, 0.1)
sr = st.select_slider("Audio sample rate", options=[16000, 22050, 32000], value=16000)
n_mels = st.select_slider("Mel bins", options=[40, 64, 80], value=64)
threshold = st.slider("Alert threshold", 0.1, 0.9, 0.6, 0.05)

if video_file is not None:
    # Show the video immediately
    st.video(video_file)

    # Save uploaded file to a temp buffer for MoviePy
    with st.spinner("Analyzing (this runs once, then displays results)…"):
        # MoviePy needs a real path; write to temp
        tmp_path = os.path.join(tempfile.gettempdir(), "clip.mp4")
        with open(tmp_path, 'wb') as f:
            f.write(video_file.getbuffer())

        # Decode frames (low FPS) and audio → mel
        frames = sample_frames(tmp_path, fps=fps, resize=224)
        # Build diff stacks per frame idx (simple motion cue)
        diff_stacks = []
        for i in range(len(frames)):
            if i < 1:
                diff_stacks.append(frames[i].repeat(3,1,1))
            else:
                diff_stacks.append(torch.cat([frames[i]-frames[i-1]]*3, dim=0))

        audio = extract_audio_array(tmp_path, sr=sr)
        mel = log_mel_spectrogram(audio, sr=sr, n_mels=n_mels, win_length=0.025, hop_length=0.010)  # [1,M,T]

        vids, mels, centers = make_windows(diff_stacks, mel, fps=fps, win_sec=win_sec, stride_sec=stride_sec, mel_hop_s=0.010)

        probs = []
        with torch.inference_mode():
            for v, a in zip(vids, mels):
                v = v.unsqueeze(0)    # [1,C,H,W]
                a = a                 # already [1,1,M,T]
                logit = model(v, a).squeeze(1)
                p = torch.sigmoid(logit).item()
                probs.append(p)

    if not probs:
        st.error("No windows were generated — try increasing clip length or lowering FPS.")
    else:
        st.subheader("Explosion likelihood over time (probability)")
        st.line_chart(probs)

        idx_hits = [i for i,p in enumerate(probs) if p >= threshold]
        if idx_hits:
            st.success(f"Predicted explosion regions near windows: {idx_hits[:15]}{' …' if len(idx_hits)>15 else ''}")
        else:
            st.info("No regions crossed the threshold. Looks like a calm explanation! ✍️📈")

        st.caption("Tip: Adjust FPS/Window/Stride for latency vs. stability trade-offs.")
