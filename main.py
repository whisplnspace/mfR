import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from feat.detector import Detector
import librosa
import gradio as gr
import tempfile
import os
import matplotlib.pyplot as plt
from PIL import Image


# Optical Flow Enhancement using TV-L1 and Farneback
def compute_optical_flow(prev_frame, next_frame):
    tv_l1 = cv2.optflow.createOptFlow_DualTVL1()
    flow_tvl1 = tv_l1.calc(prev_frame, next_frame, None)
    flow_farneback = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow_tvl1, flow_farneback


# AU Feature Extraction using OpenFace
feat_detector = Detector(device='cuda' if torch.cuda.is_available() else 'cpu', output_size=112)


def extract_au_features(image):
    try:
        pil_image = Image.fromarray(image).convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            pil_image.save(temp_img.name)
            temp_img_path = temp_img.name

        result = feat_detector.detect_image(temp_img_path)
        os.remove(temp_img_path)
        return result.iloc[0].values if not result.empty else np.zeros(20)
    except Exception as e:
        print(f"Error in AU Extraction: {e}")
        return np.zeros(20)


# Facial EMG Simulation
def simulate_emg_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.array([np.mean(gray) / 255.0])


# Extract Speech-Based Features
def extract_mfcc(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1) if mfccs.shape[1] > 0 else np.zeros(13)
    except Exception as e:
        print(f"Error in MFCC Extraction: {e}")
        return np.zeros(13)


# Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(Image.fromarray(image)).unsqueeze(0)


# Load Model
student_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
student_model.fc = nn.Linear(512, 5)
student_model.eval()

# Emotion Labels
emotion_labels = ["Neutral", "Happy", "Sad", "Angry", "Surprised"]


# Prediction Function
def predict(image, audio):
    print("Processing Image...")
    image = np.array(image)
    au_features = extract_au_features(image)
    emg_features = simulate_emg_features(image)
    img_tensor = preprocess_image(image)

    print("Processing Audio...")
    mfcc_features = extract_mfcc(str(audio))

    # Convert features to tensors
    au_tensor = torch.tensor(au_features).float().unsqueeze(0)
    emg_tensor = torch.tensor(emg_features).float().unsqueeze(0)
    mfcc_tensor = torch.tensor(mfcc_features).float().unsqueeze(0)
    input_tensor = torch.cat((au_tensor, emg_tensor, mfcc_tensor), dim=1)

    # Model Prediction
    with torch.no_grad():
        img_pred = student_model(img_tensor)
        img_pred = F.softmax(torch.tensor(img_pred), dim=1).numpy()
        combined_pred = np.concatenate((img_pred, input_tensor.numpy()), axis=1)

    # Format Output
    formatted_output = {
        "Image Prediction (Probabilities)": {emotion_labels[i]: round(img_pred[0][i], 3) for i in
                                             range(len(emotion_labels))},
        "AU Features": au_features.tolist(),
        "EMG Signal": emg_features.tolist(),
        "MFCC Features": mfcc_features.tolist()
    }

    print("Formatted Prediction Output:", formatted_output)

    # Visualization
    plt.figure(figsize=(6, 4))
    plt.bar(emotion_labels, img_pred[0], color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.xlabel("Emotions")
    plt.ylabel("Probability")
    plt.title("Emotion Prediction Distribution")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    plt.savefig(plot_path)
    plt.close()

    return formatted_output, plot_path


# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Facial Image"),
        gr.Audio(type="filepath", label="Upload Voice Sample")
    ],
    outputs=[
        gr.JSON(label="Predicted Emotional Data"),
        gr.Image(label="Emotion Probability Distribution")
    ],
    title="🎭 Micro-Expression & Speech Emotion Analyzer",
    description="💡 Upload an image and an audio file to analyze subtle micro-expressions and emotional cues using AI.",
    theme="compact",
    allow_flagging="never",
    live=True
)

if __name__ == "__main__":
    iface.launch()