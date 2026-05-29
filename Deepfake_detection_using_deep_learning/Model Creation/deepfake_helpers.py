import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

try:
    import face_recognition  # type: ignore[import]
    FACE_RECOG_AVAILABLE = True
except ImportError:
    face_recognition = None
    FACE_RECOG_AVAILABLE = False

IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'}

# Use a Resize + CenterCrop to preserve aspect ratio, then normalize
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.CenterCrop(IM_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# Legacy transforms (original project preprocessing) kept for compatibility
legacy_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


def allowed_video_file(filename: str) -> bool:
    return filename.split('.')[-1].lower() in ALLOWED_VIDEO_EXTENSIONS


def frame_extract(video_path: str):
    cap = cv2.VideoCapture(video_path)
    success = True
    while success:
        success, frame = cap.read()
        if success and frame is not None:
            yield frame
    cap.release()


def crop_face(frame: np.ndarray, padding: int = 40) -> np.ndarray:
    if FACE_RECOG_AVAILABLE:
        locations = face_recognition.face_locations(frame)
        if len(locations) > 0:
            top, right, bottom, left = locations[0]
            top = max(top - padding, 0)
            left = max(left - padding, 0)
            bottom = min(bottom + padding, frame.shape[0])
            right = min(right + padding, frame.shape[1])
            return frame[top:bottom, left:right]

    # fallback to Haar cascade face detector if available, otherwise center crop
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            # choose the largest detected face
            x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
            top = max(y - padding, 0)
            left = max(x - padding, 0)
            bottom = min(y + h + padding, frame.shape[0])
            right = min(x + w + padding, frame.shape[1])
            return frame[top:bottom, left:right]
    except Exception:
        pass

    # final fallback: center crop
    height, width = frame.shape[:2]
    crop_size = min(height, width)
    y1 = max((height - crop_size) // 2, 0)
    x1 = max((width - crop_size) // 2, 0)
    return frame[y1:y1 + crop_size, x1:x1 + crop_size]


def sample_video_frames(video_path: str, sequence_length: int, start_idx: int = None):
    frames = list(frame_extract(video_path))
    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    if len(frames) < sequence_length:
        # repeat the last frame when there are too few frames
        frames = frames + [frames[-1]] * (sequence_length - len(frames))

    if start_idx is None:
        indices = np.linspace(0, len(frames) - 1, num=sequence_length, dtype=int)
        return [frames[i] for i in indices]

    # take a contiguous clip starting at start_idx (clamp to valid range)
    start = int(max(0, min(start_idx, len(frames) - sequence_length)))
    return frames[start:start + sequence_length]


def frames_to_tensor(frames: list, transform=train_transforms) -> torch.Tensor:
    processed = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cropped = crop_face(frame)
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            cropped = frame
        processed.append(transform(cropped))

    tensor = torch.stack(processed)
    return tensor.unsqueeze(0)


def get_all_frames(video_path: str):
    return list(frame_extract(video_path))


class DeepfakeModel(nn.Module):
    def __init__(self, num_classes: int = 2, latent_dim: int = 2048, lstm_layers: int = 1, hidden_dim: int = 2048, bidirectional: bool = False, pretrained: bool = False):
        super(DeepfakeModel, self).__init__()
        backbone = models.resnext50_32x4d(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        linear_in = hidden_dim * (2 if bidirectional else 1)
        self.linear = nn.Linear(linear_in, num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.feature_extractor(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)
        x_lstm, _ = self.lstm(x)
        logits = self.dropout(self.linear(x_lstm[:, -1, :]))
        return fmap, logits


def select_model_by_sequence_length(model_folder: str, sequence_length: int) -> str:
    model_folder_path = Path(model_folder)
    if not model_folder_path.exists() or not model_folder_path.is_dir():
        raise FileNotFoundError(f"Model folder does not exist: {model_folder}")

    candidates = []
    for path in model_folder_path.glob('*.pt'):
        parts = path.stem.split('_')
        if len(parts) >= 4:
            try:
                seq = int(parts[3])
                if seq == sequence_length:
                    candidates.append(path)
            except ValueError:
                continue

    if len(candidates) == 0:
        raise FileNotFoundError(f"No model file found for sequence length {sequence_length} in {model_folder}")

    def parse_accuracy(p: Path):
        parts = p.stem.split('_')
        try:
            return float(parts[1])
        except (IndexError, ValueError):
            return 0.0

    candidates.sort(key=parse_accuracy, reverse=True)
    return str(candidates[0])


def select_top_models_by_sequence_length(model_folder: str, sequence_length: int, top_k: int = 3):
    """Return up to top_k model paths matching the given sequence_length, sorted by accuracy (descending)."""
    model_folder_path = Path(model_folder)
    if not model_folder_path.exists() or not model_folder_path.is_dir():
        raise FileNotFoundError(f"Model folder does not exist: {model_folder}")

    candidates = []
    for path in model_folder_path.glob('*.pt'):
        parts = path.stem.split('_')
        if len(parts) >= 4:
            try:
                seq = int(parts[3])
                if seq == sequence_length:
                    candidates.append(path)
            except ValueError:
                continue

    if len(candidates) == 0:
        raise FileNotFoundError(f"No model file found for sequence length {sequence_length} in {model_folder}")

    def parse_accuracy(p: Path):
        parts = p.stem.split('_')
        try:
            return float(parts[1])
        except (IndexError, ValueError):
            return 0.0

    candidates.sort(key=parse_accuracy, reverse=True)
    return [str(p) for p in candidates[:top_k]]


def load_model_weights(model: nn.Module, model_path: str, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    # support full checkpoint or raw state_dict
    if isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model_state_dict' in checkpoint):
        state = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
        model.load_state_dict(state)
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        # raw state_dict saved as dict
        model.load_state_dict(checkpoint)
    else:
        # fallback: try to load as state_dict directly
        try:
            model.load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Unable to load model checkpoint: {e}")
    model.to(device)
    model.eval()
    return model


def predict_video(model: nn.Module, video_path: str, sequence_length: int, device: torch.device, threshold: float = 0.08, n_clips: int = 5, use_legacy_transforms: bool = False, use_tta: bool = False):
    # Read all frames once
    frames = get_all_frames(video_path)
    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    model.to(device)
    model.eval()

    probs_accum = None
    # choose start indices evenly across the video for ensemble
    max_start = max(0, len(frames) - sequence_length)
    if n_clips <= 1 or max_start == 0:
        starts = [0]
    else:
        starts = [int(x) for x in np.linspace(0, max_start, num=n_clips, dtype=int)]

    with torch.no_grad():
        for s in starts:
            clip = sample_video_frames(video_path, sequence_length, start_idx=s)
            transform = legacy_transforms if use_legacy_transforms else train_transforms
            tensor = frames_to_tensor(clip, transform=transform).to(device)
            _, logits = model(tensor)
            p = torch.softmax(logits, dim=1)
            # Test-time augmentation: horizontal flip average
            if use_tta:
                try:
                    flipped = [np.fliplr(f) for f in clip]
                    tensor_f = frames_to_tensor(flipped, transform=transform).to(device)
                    _, logits_f = model(tensor_f)
                    p_f = torch.softmax(logits_f, dim=1)
                    p = (p + p_f) / 2.0
                except Exception:
                    # if TTA fails for any reason, fall back to original prediction
                    pass
            if probs_accum is None:
                probs_accum = p
            else:
                probs_accum = probs_accum + p

    probs = probs_accum / len(starts)
    fake_prob = float(probs[0, 0].item() * 100)
    real_prob = float(probs[0, 1].item() * 100)
    max_prob = max(fake_prob, real_prob)
    confidence = float(max_prob)

    # Convert threshold to percentage
    thresh_pct = threshold * 100.0
    if abs(fake_prob - real_prob) < thresh_pct:
        label = 'UNCERTAIN'
    else:
        label = 'REAL' if real_prob > fake_prob else 'FAKE'

    return label, confidence, fake_prob, real_prob


def predict_video_ensemble(model_paths: list, video_path: str, sequence_length: int, device: torch.device, threshold: float = 0.08, n_clips: int = 5, use_legacy_transforms: bool = False, use_tta: bool = False, pretrained_backbone: bool = False):
    """Load each model path, run inference, and average probabilities across models and temporal clips."""
    if len(model_paths) == 0:
        raise ValueError("No model paths provided for ensemble prediction")

    frames = get_all_frames(video_path)
    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    device = device
    max_start = max(0, len(frames) - sequence_length)
    if n_clips <= 1 or max_start == 0:
        starts = [0]
    else:
        starts = [int(x) for x in np.linspace(0, max_start, num=n_clips, dtype=int)]

    probs_models_accum = None

    for mp in model_paths:
        model = DeepfakeModel(num_classes=2, pretrained=pretrained_backbone)
        load_model_weights(model, str(mp), device)
        model.to(device)
        model.eval()

        probs_accum = None
        with torch.no_grad():
            for s in starts:
                clip = sample_video_frames(video_path, sequence_length, start_idx=s)
                transform = legacy_transforms if use_legacy_transforms else train_transforms
                tensor = frames_to_tensor(clip, transform=transform).to(device)
                _, logits = model(tensor)
                p = torch.softmax(logits, dim=1)
                if use_tta:
                    try:
                        flipped = [np.fliplr(f) for f in clip]
                        tensor_f = frames_to_tensor(flipped, transform=transform).to(device)
                        _, logits_f = model(tensor_f)
                        p_f = torch.softmax(logits_f, dim=1)
                        p = (p + p_f) / 2.0
                    except Exception:
                        pass

                if probs_accum is None:
                    probs_accum = p
                else:
                    probs_accum = probs_accum + p

        probs_avg = probs_accum / len(starts)
        if probs_models_accum is None:
            probs_models_accum = probs_avg
        else:
            probs_models_accum = probs_models_accum + probs_avg

    probs = probs_models_accum / len(model_paths)
    fake_prob = float(probs[0, 0].item() * 100)
    real_prob = float(probs[0, 1].item() * 100)
    max_prob = max(fake_prob, real_prob)
    confidence = float(max_prob)

    thresh_pct = threshold * 100.0
    if abs(fake_prob - real_prob) < thresh_pct:
        label = 'UNCERTAIN'
    else:
        label = 'REAL' if real_prob > fake_prob else 'FAKE'

    return label, confidence, fake_prob, real_prob
