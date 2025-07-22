from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16
from PIL import Image
import io

app = Flask(__name__, static_folder='assets')

# Load model
model = torch.load('pneumonAI_model.pth', map_location=torch.device('cpu'))

model.eval()

class_to_label = {0: 'Normal', 1: 'Pneumonia'}

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Load image
    img = Image.open(io.BytesIO(file.read()))
    
    # Preprocess image
    img_tensor = preprocess(img).unsqueeze(0) 

    # Make the prediction
    with torch.no_grad():
        output = model(img_tensor)
    
    # Apply softmax to get confidence scores
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get class label and confidence
    predicted_class = torch.argmax(probabilities, 1).item()
    confidence = probabilities[0][predicted_class].item()

    return jsonify({'prediction': class_to_label[predicted_class], 'confidence': confidence * 100})

if __name__ == '__main__':
    app.run(debug=True)
