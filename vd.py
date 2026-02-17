
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Load trained model
model = torch.load("deepfake_model.pth", map_location="cpu")
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()

    return prob

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        score = predict_frame(frame)
        scores.append(score)

    cap.release()

    avg_score = np.mean(scores)
    label = "FAKE" if avg_score > 0.5 else "REAL"

    return label, avg_score


