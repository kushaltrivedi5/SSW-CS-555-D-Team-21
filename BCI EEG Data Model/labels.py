import numpy as np

# Load the train_label.npy file
train_labels = np.load('/Users/siddhartharaovs/Downloads/CS555-Agile project Files/BCI_project/train_label.npy')

# Create a dictionary mapping numerical labels to class labels
label_to_class = {
    0: "hello 👋 ",
    1: "help me ",
    2: "stop",
    3: "thank you",
    4: "yes " 
}

# Map numerical labels to class labels with label numbers
class_labels_with_numbers = [(label, label_to_class[label]) for label in train_labels]

# Print the first few class labels with label numbers
print("First few class labels with label numbers:")
print(class_labels_with_numbers[:1000])  # Print the first 1000 class labels with label numbers
