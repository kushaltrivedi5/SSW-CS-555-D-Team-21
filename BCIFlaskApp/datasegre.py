import matplotlib.pyplot as plt
import numpy as np

# Load the test data and labels
test_data = np.load('/Users/siddhartharaovs/Downloads/CS555-Agile project Files/BCI_project/test_data.npy')  # Replace with the path to your test_data.npy file
test_labels = np.load('/Users/siddhartharaovs/Downloads/CS555-Agile project Files/BCI_project/test_label.npy')  # Replace with the path to your test_label.npy file

# Define the class labels and number of channels
class_labels = {1: 'Hello', 2: 'Help me', 3: 'Stop', 4: 'Thank you', 5: 'Yes'}
n_channels = test_data.shape[1]

# Create a figure with subplots - 5 classes x n_channels
fig, axs = plt.subplots(len(class_labels), n_channels, figsize=(20, 10), sharex=True, sharey=True)

# Loop over classes and channels
for class_index, (class_value, class_name) in enumerate(class_labels.items()):
    # Select the data for the current class
    class_data = test_data[test_labels == class_value]

    for channel_index in range(n_channels):
        # Select the data for the current channel
        channel_data = class_data[:, channel_index, :]

        # Plotting the time series for the current channel
        ax = axs[class_index, channel_index] if n_channels > 1 else axs[class_index]
        for instance in channel_data:
            ax.plot(instance, alpha=0.5)  # Plot with some transparency

        # Add title to the first row and y-labels to the first column
        if class_index == 0:
            ax.set_title(f'Channel {channel_index+1}')
        if channel_index == 0:
            ax.set_ylabel(class_name)

# Set common labels
fig.text(0.5, 0.04, 'Time Points', ha='center', va='center')
fig.text(0.06, 0.5, 'EEG Signal Amplitude', ha='center', va='center', rotation='vertical')

plt.tight_layout()
plt.show()