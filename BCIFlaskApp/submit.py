from flask import Flask, request, render_template, jsonify
import torch
from model import EEGAutoencoderClassifier  # Ensure your model is properly imported
import numpy as np

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EEGAutoencoderClassifier(num_classes=5).to(device)
model.load_state_dict(torch.load('model.pth'))  # Load your pre-trained model
model.eval()

def process_file(file):
    # Here you should include the code to process the file
    # For example, loading a .npy file and converting it to a tensor
    data = np.load(file)
    tensor = torch.Tensor(data).to(device)
    output = model(tensor.unsqueeze(0))  # Assume model expects batch dimension
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        label_index = process_file(file)
        labels = ['hello', 'help me', 'stop', 'thank you', 'yes']
        result = labels[label_index]
        return jsonify({'label': result})

if __name__ == '__main__':
    app.run(debug=True)
