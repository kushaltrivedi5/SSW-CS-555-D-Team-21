import numpy as np
import torch
import json

# Load the test data and labels
test_data = np.load('/Users/siddhartharaovs/Downloads/CS555-Agile project Files/BCI_project/test_data.npy')
test_labels = np.load('/Users/siddhartharaovs/Downloads/CS555-Agile project Files/BCI_project/test_label.npy')

# Convert the test data to PyTorch tensor
x_test_tensor = torch.Tensor(test_data)
class EEGAutoencoderClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EEGAutoencoderClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 795, 512), # Input dimention is 64 channel * 795 time point, and use 256 units for first NN layer
            nn.ReLU(), # Use ReLu function for NN training
            nn.Linear(512, 256), # 256 NN units to 128 units
            nn.ReLU(),
            nn.Linear(256, 128),#  128 NN units to 64 units
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes), # num_classes is 5 ("hello,” “help me,” “stop,” “thank you,” and “yes”)
            nn.LogSoftmax(dim=1)  # Use LogSoftmax for multi-class classification
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)

        # import pdb;pdb.set_trace()
        x = self.classifier(x)
        return x
    

# Perform inference using the trained model on the test data
model.eval()
with torch.no_grad():
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs, 1)

# Map the predicted numerical labels to class labels
predicted_labels = [label_to_class[label.item()] for label in predicted]

# Calculate accuracy
correct = (predicted == test_labels).sum().item()
total = len(test_labels)
accuracy = correct / total

# Convert the output classes to emojis (for demonstration purposes)
class_to_emoji = {
    "hello": "👋",
    "help me": "🆘",
    "stop": "🛑",
    "thank you": "🙏",
    "yes": "✅"
}
predicted_emojis = [class_to_emoji[label] for label in predicted_labels]

# Format the results as a JSON object
results = {
    "accuracy": accuracy,
    "predicted_labels": predicted_labels,
    "predicted_emojis": predicted_emojis
}

# Print the results as JSON
print(json.dumps(results, indent=4))
