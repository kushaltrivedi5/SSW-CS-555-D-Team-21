import numpy as np

# Load the train_label.npy file
train_labels = np.load('/Users/siddhartharaovs/Downloads/CS555-Agile project Files/BCI_project/train_label.npy')

# Get unique numerical labels
unique_labels = np.unique(train_labels)

# Create a dictionary mapping numerical labels to class labels
label_to_class = {label: f"class_{label}" for label in unique_labels}

# Map numerical labels to class labels
class_labels = [label_to_class[label] for label in train_labels]

# Print the first few class labels
print("First few class labels:")
print(class_labels[:10])  # Print the first 10 class labels
