import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from io import BytesIO
from flask import send_file
import base64

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture
class FacialExpressionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FacialExpressionModel, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Fourth Convolutional Block
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Calculate the flattened size
        self.flat_features = 512 * 3 * 3
        
        # First Fully Connected Layer
        self.fc1 = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # Second Fully Connected Layer
        self.fc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # Output Layer
        self.output = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(-1, self.flat_features)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load available models
MODEL_DIR = "./models"
models_available = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def load_model(model_path):
    model = FacialExpressionModel(num_classes=len(EMOTIONS)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# Face detection (using Haar cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion(model, face_img):
    image_tensor = transform(Image.fromarray(face_img)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return EMOTIONS[predicted.item()]

def draw_face_boxes(img, faces, emotions):
    for (x, y, w, h), emotion in zip(faces, emotions):
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Put emotion label above the rectangle
        label = emotion
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(img, label, (x, y - 10), font, font_scale, color, thickness)

    return img

@app.route('/')
def index():
    return render_template('index.html', models=models_available)

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    model_path = os.path.join(MODEL_DIR, model_name)
    model = load_model(model_path)

    if 'image' in request.files:
        file = request.files['image']
        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        emotions = []

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (56, 56))
            emotion = predict_emotion(model, face_img)
            emotions.append(emotion)

        # Draw the bounding boxes and emotion labels on the image
        img_with_boxes = draw_face_boxes(img, faces, emotions)

        # Convert image to PNG format
        _, img_encoded = cv2.imencode('.png', img_with_boxes)
        img_bytes = img_encoded.tobytes()

        # Send the image as a response
        return send_file(BytesIO(img_bytes), mimetype='image/png')

    return "No image provided", 400

@app.route('/webcam_feed')
def webcam_feed():
    return render_template('webcam.html', models=models_available)

@app.route('/live_predict', methods=['POST'])
def live_predict():
    model_name = request.form['model']
    model_path = os.path.join(MODEL_DIR, model_name)
    model = load_model(model_path)

    # Decode base64 image sent from frontend
    data = request.form['image'].split(',')[1]
    img_data = base64.b64decode(data)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotions = []

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (56, 56))
        emotion = predict_emotion(model, face_img)
        emotions.append(emotion)

    # Draw the bounding boxes and emotion labels on the image
    img_with_boxes = draw_face_boxes(img, faces, emotions)

    # Convert image to PNG format
    _, img_encoded = cv2.imencode('.png', img_with_boxes)
    img_bytes = img_encoded.tobytes()

    # Send the image as a response
    return send_file(BytesIO(img_bytes), mimetype='image/png')

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)