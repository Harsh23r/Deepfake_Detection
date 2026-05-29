import cv2
import numpy as np
import torch
from pathlib import Path
from deepfake_helpers import DeepfakeModel

video_path = Path('test_dummy.mp4')
model_path = Path('model_90_acc_60_frames_final_data.pt')

if video_path.exists():
    video_path.unlink()
if model_path.exists():
    model_path.unlink()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (112, 112))
for i in range(20):
    frame = np.full((112, 112, 3), int(i * 10) % 256, dtype=np.uint8)
    out.write(frame)
out.release()

model = DeepfakeModel(num_classes=2, pretrained=False)
torch.save(model.state_dict(), str(model_path))
print(f'Created {video_path} and {model_path}')
