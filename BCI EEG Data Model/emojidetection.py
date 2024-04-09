import numpy as np
import torch
import json

# Load the test data and labels
test_data = np.load('/Users/siddhartharaovs/Downloads/CS555-Agile project Files/BCI_project/test_data.npy')
test_labels = np.load('/Users/siddhartharaovs/Downloads/CS555-Agile project Files/BCI_project/test_label.npy')

# Convert the test data to PyTorch tensor
x_test_tensor = torch.Tensor(test_data)


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
