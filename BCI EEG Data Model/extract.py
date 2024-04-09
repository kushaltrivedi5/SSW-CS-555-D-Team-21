import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Load the EEG data and labels
data_path = '/Users/siddhartharaovs/Downloads/CS555-Agile project Files/BCI_project/'
eeg_data = np.load(data_path + 'train_data.npy')
labels = np.load(data_path + 'train_label.npy')

# Function to label the EEG data
def label_eeg_data(eeg_data, labels):
    labeled_data = []
    for label in labels:
        if label == "hello":
            labeled_data.append((eeg_data, 0))  # Assign label 0 for "hello"
        elif label == "help me":
            labeled_data.append((eeg_data, 1))  # Assign label 1 for "help me"
        elif label == "stop":
            labeled_data.append((eeg_data, 2))  # Assign label 2 for "stop"
        elif label == "thank you":
            labeled_data.append((eeg_data, 3))  # Assign label 3 for "thank you"
        elif label == "yes":
            labeled_data.append((eeg_data, 4))  # Assign label 4 for "yes"
        else:
            print("Invalid label:", label)
    return labeled_data

# Label the EEG data
labeled_data = label_eeg_data(eeg_data, labels)

# Save the labeled data
np.save(data_path + 'labeled_data.npy', labeled_data)

# To convert the labeled data into PyTorch tensors
x_tensor = torch.Tensor([sample[0] for sample in labeled_data])
y_tensor = torch.LongTensor([sample[1] for sample in labeled_data])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the batch size
batch_size = 64

# Split the data into train and test sets
split_ratio = 0.8
split_index = int(len(x_tensor) * split_ratio)

x_train_tensor = x_tensor[:split_index].to(device)
y_train_tensor = y_tensor[:split_index].to(device)
x_test_tensor = x_tensor[split_index:].to(device)
y_test_tensor = y_tensor[split_index:].to(device)

# Create data loaders
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model
class EEGAutoencoderClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EEGAutoencoderClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 795, 256),  # Input dimention is 64 channel * 795 time point, and use 256 units for first NN layer
            nn.ReLU(),  # Use ReLu function for NN training
            nn.Linear(256, 128),  # 256 NN units to 128 units
            nn.ReLU(),
            nn.Linear(128, 64),  # 128 NN units to 64 units
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes),  # num_classes is 5 (hello,” “help me,” “stop,” “thank you,” and “yes”)
            nn.LogSoftmax(dim=1)  # Use LogSoftmax for multi-class classification
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.classifier(x)
        return x

# Instantiate the model
num_classes = 5  # Number of classes
model = EEGAutoencoderClassifier(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.NLLLoss()  # Use NLLLoss function to optimize
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Setting parameters learning rate = 0.001

# Train the model
num_epochs = 20  # Number of training epochs
for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
correct = 0
total = 0
class_correct = list(0. for _ in range(num_classes))
class_total = list(0. for _ in range(num_classes))

with torch.no_grad():
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(num_classes)]
overall_accuracy = 100 * correct / total

# Print overall accuracy and class-wise accuracy
print(f'Overall Accuracy: {overall_accuracy:.2f}%')
for i, acc in enumerate(class_accuracy):
    print(f'Class {i} Accuracy: {acc:.2f}%')

# Plotting the results
# True Positive Rate Matrix
true_positive_rate_matrix = np.array(class_correct) / np.array(class_total)
plt.figure(figsize=(8, 6))
sns.heatmap(true_positive_rate_matrix.reshape(1, -1), annot=True, cmap='viridis', fmt=".2f", linewidths=.5, cbar=True)
plt.title('True Positive Rate Matrix')
plt.xlabel('Class')
plt.ylabel('True Positive Rate')
plt.xticks(ticks=np.arange(5) + 0.5, labels=['hello', 'help me', 'stop', 'thank you', 'yes'])
plt.tight_layout()
plt.show()
