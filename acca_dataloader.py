import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import decord
import librosa
import numpy as np
import warnings

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")
decord.bridge.set_bridge('torch')

class HardNegativeDataset(Dataset):
    def __init__(self, json_path, video_dir):
        with open(json_path, 'r') as f_in:
            self.entries = json.load(f_in)
        self.video_dir = video_dir

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        warnings.filterwarnings("ignore")
        entry = self.entries[idx]
        v_path = f"{self.video_dir}/{entry['video_id']}.mp4"

        try:
            vr = decord.VideoReader(v_path)
            indices = torch.linspace(0, len(vr) - 1, 16).long()
            video_tensor = vr.get_batch(indices).permute(0, 3, 1, 2).float() / 255.0
            video_tensor = F.interpolate(video_tensor, size=(378, 378), mode='bilinear')
        except Exception:
            video_tensor = torch.randn((16, 3, 378, 378)) * 0.01

        try:
            y, _ = librosa.load(v_path, sr=16000, mono=True, duration=10.0)
            S = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
            S_tensor = torch.from_numpy(S_norm).unsqueeze(0).unsqueeze(0)
            S_resized = F.interpolate(S_tensor, size=(378, 378)).squeeze(0)
            audio_img = S_resized.repeat(3, 1, 1)
        except Exception:
            audio_img = torch.randn((3, 378, 378)) * 0.01

        return {'video': video_tensor, 'audio': audio_img}
