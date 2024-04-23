# %% [markdown]
# # EEG DataLoader

# %% [markdown]
# The dimensions of the training set are as follows: 4,500 samples, 64 channels, and a time length of 795. This corresponds to 5 categories in y_train.
# 
# The dimensions of the testing set are as follows: 750 samples, 64 channels, and a time length of 795. This corresponds to 5 categories in y_test.
# 
# You can download it from this Google Drive link: [https://drive.google.com/drive/folders/1ykR-mn4d4KfFeeNrfR6UdtebsNRY8PU2?usp=sharing].
# Please download the data and place it in your data_path at "./data."

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchviz import make_dot

# %%
data_path = '/Users/siddhartharaovs/Downloads/CS555-Agile project Files/BCI_project/'

# %%
train_data = np.load(data_path + 'train_data.npy')
test_data = np.load(data_path + 'test_data.npy')
train_label = np.load(data_path + 'train_label.npy')
test_label = np.load(data_path + 'test_label.npy')

#To convert the data into PyTorch tensors
x_train_tensor = torch.Tensor(train_data)
y_train_tensor = torch.LongTensor(train_label)
x_test_tensor = torch.Tensor(test_data)
y_test_tensor = torch.LongTensor(test_label)

# %%
y_test_tensor

# %%
import pandas as pd 
ttt=x_test_tensor

# %%

print((ttt))

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Setting GPU on your computer

# %%
train_dataset = TensorDataset(x_train_tensor.to(device), y_train_tensor.to(device)) # input data to Tensor dataloader
train_loader = DataLoader(train_dataset, batch_size=64, drop_last=True, shuffle=True) #  Batch size refers to the number of data sample
test_dataset = TensorDataset(x_test_tensor.to(device), y_test_tensor.to(device))
test_loader = DataLoader(test_dataset, batch_size=64,  drop_last=True,shuffle=False)

# %%
print(type(train_loader))

# %%
count = 0
for batch_idx, (data, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}:")
    print("Data:")
    print(data[:1])  # Print first 5 data entries in the batch
    d1=data[:1]
    print("Labels:")
    print(labels[:1])  # Print first 5 label entries in the batch
    l1=labels[:1]
    count += 1
    if count == 1:
        break  # Stop after printing 5 batches

# %%
print(type(d1))
print(type(l1))

# %%
print(d1)
print(l1)

# %%
import pandas as pd

# Assuming you have two tensors: tensor1 and tensor2

# Convert tensors to numpy arrays
array1 = d1.numpy()
array2 = l1.numpy()
print(array1)
# Create a DataFrame from the numpy arrays
df = pd.DataFrame({'Column1': array1.flatten(), 'Column2': array2.flatten()})

# Display the first 5 entries of the DataFrame
print(df.head())


# %%
import pandas as pd

# Initialize empty lists to store the data
data_list = []

# Iterate over the DataLoader
for batch in train_loader:
    # Extract data and labels from the batch
    x_batch, y_batch = batch
    
    # Convert tensors to numpy arrays
    array1 = x_batch
    array2 = y_batch
    
    # Flatten arrays if needed
    array1_flat = array1.flatten()
    array2_flat = array2.flatten()
    
    # Combine arrays into a list of tuples
    combined_data = list(zip(array1_flat, array2_flat))
    
    # Extend the data list with the combined data from the batch
    data_list.extend(combined_data)

# Create a DataFrame from the combined data
df = pd.DataFrame(data_list, columns=['Column1', 'Column2'])

# Display the first 5 entries of the DataFrame
print(df.head())


# %%
import pandas as pd

# Assuming df is your DataFrame containing the combined data

# Define the file path where you want to save the Excel file
excel_file_path = 'combined_data.xlsx'

# Store the DataFrame in an Excel file
df.to_excel(excel_file_path, index=False)

print(f"DataFrame successfully saved to {excel_file_path}")


# %%
df = pd.read_excel('combined_data.xlsx')

# Extract integer values from tensor strings in Column2
df['Column2'] = df['Column2'].str.extract(r'tensor\((\d+)\)').astype(int)

# Define the mapping from label numbers to class descriptions
label_to_class = {
    0: "hello",
    1: "help me",
    2: "stop",
    3: "thank you",
    4: "yes"
}

# Apply the mapping to create a new column 'Class'
df['Class'] = df['Column2'].map(label_to_class)

# Save the DataFrame with the new 'Class' column to a new Excel file
df.to_excel('combined_data_classified.xlsx', index=False)

# Print the first few rows of the DataFrame to demonstrate the classification
print(df.head())


# %%
# Load the classified Excel file
df = pd.read_excel('combined_data_classified.xlsx')

# Labels and their corresponding file names
labels_to_files = {
    "hello": "hello_data.xlsx",
    "help me": "help_me_data.xlsx",
    "stop": "stop_data.xlsx",
    "thank you": "thank_you_data.xlsx",
    "yes": "yes_data.xlsx"
}

# Loop through each label and save the filtered data to separate Excel files
for label, filename in labels_to_files.items():
    # Filter the DataFrame for the current label
    filtered_df = df[df['Class'] == label]
    # Save the filtered DataFrame to an Excel file
    filtered_df.to_excel(filename, index=False)

# Print a message when all files have been successfully saved
print("Files have been successfully saved based on labels.")


# %%
import pandas as pd
import numpy as np

# Dictionary to link labels with their corresponding Excel file paths
labels_to_files = {
    "hello": "hello_data.xlsx",
    "help me": "help_me_data.xlsx",
    "stop": "stop_data.xlsx",
    "thank you": "thank_you_data.xlsx",
    "yes": "yes_data.xlsx"
}

# Loop through each label and its corresponding file path
for label, file_path in labels_to_files.items():
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Convert the DataFrame to a NumPy array
    data_array = df.values

    # Save the array as a .npy file
    np.save(f'{label}_data.npy', data_array)

    # Print confirmation message
    print(f"The .npy file for {label} has been successfully created.")


# %%
import numpy as np

# List of labels that correspond to the .npy files
labels = ["hello", "help me", "stop", "thank you", "yes"]

# Loop through each label and load the corresponding .npy file
for label in labels:
    # Construct the file path
    file_path = f'{label}_data.npy'
    
    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)
    
    # Print the first five entries of the array
    print(f"First five entries for {label}:")
    print(data[:5])
    print()  # Adds a blank line for better readability between outputs


# %%
import pandas as pd

def compare_excel_files(file1, file2):
    # Load the data from both files
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    # Example: Check if both DataFrames are identical
    comparison_result = df1.equals(df2)
    return comparison_result

# Assume files are renamed or not, load with the actual format
file1 = 'combined_data_classified.xlsx'
file2 = 'hello_data.xlsx'

# Compare the files
are_files_same = compare_excel_files(file1, file2)
print("Are the two Excel files the same?", are_files_same)


# %%
import pandas as pd

def load_and_display_excel(file_path, num_rows=5):
    """ Load an Excel file into a DataFrame and display the first few rows. """
    df = pd.read_excel(file_path)
    print(f"First {num_rows} rows of {file_path.split('/')[-1]}:")
    print(df.head(num_rows))
    return df

def main():
    # Define the file paths
    file_path_1 = 'combined_data_classified.xlsx'
    file_path_2 = 'hello_data.xlsx'
    
    # Load and display the first 5 rows of each file
    df1 = load_and_display_excel(file_path_1)
    df2 = load_and_display_excel(file_path_2)

if __name__ == "__main__":
    main()


# %%
import pandas as pd

def load_excel_data(file_path):
    """ Load an Excel file into a DataFrame. """
    return pd.read_excel(file_path)

def compare_and_display_class_labels(main_df, comparison_df):
    """ Compare values in Column1 and Column2, displaying corresponding class labels from main_df. """
    # Convert DataFrame rows to tuples of (Column1, Column2) for easier comparison
    main_values_tuples = main_df.set_index(['Column1', 'Column2'])['Class'].to_dict()
    
    # Initialize results list to store output strings
    results = []

    # Iterate through comparison DataFrame
    for index, row in comparison_df.iterrows():
        value_tuple = (row['Column1'], row['Column2'])
        if value_tuple in main_values_tuples:
            results.append(f"class label = '{main_values_tuples[value_tuple]}'")
        else:
            results.append(f"class label = 'No match found'")

    return results

def main():
    # File paths
    main_file_path = 'combined_data_classified.xlsx'
    comparison_file_path = 'stop_data.xlsx'
    
    # Load data from both Excel files
    main_df = load_excel_data(main_file_path)
    comparison_df = load_excel_data(comparison_file_path)
    
    # Perform comparison and get results
    classification_results = compare_and_display_class_labels(main_df, comparison_df)
    
    # Print the results
    for result in classification_results:
        print(result)

if __name__ == "__main__":
    main()


# %%
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def load_excel_data(file_path):
    """ Load an Excel file into a DataFrame. """
    return pd.read_excel(file_path)

def compare_and_display_class_labels(main_df, user_df):
    """ Compare values in user_df against main_df and display corresponding class labels. """
    results = []
    for index, row in user_df.iterrows():
        value1, value2 = row['Column1'], row['Column2']
        match = main_df[(main_df['Column1'] == value1) & (main_df['Column2'] == value2)]
        if not match.empty:
            class_label = match.iloc[0]['Class']
            results.append(f"class label = '{class_label}'")
        else:
            results.append("class label = 'No match found'")
    return results

def get_file_path_gui():
    """ Opens a GUI to select the file for classification. """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select an Excel file", filetypes=[("Excel files", "*.xlsx")])
    return file_path

def main():
    # File path for the main data
    main_file_path = 'combined_data_classified.xlsx'
    
    # Load the main DataFrame
    main_df = load_excel_data(main_file_path)
    
    # Get the path of the user input file via GUI
    user_file_path = get_file_path_gui()
    if user_file_path:
        user_df = load_excel_data(user_file_path)
        # Perform comparison and get results
        classification_results = compare_and_display_class_labels(main_df, user_df)
        # Print the results
        for result in classification_results:
            print(result)
    else:
        print("No file selected or operation cancelled.")

if __name__ == "__main__":
    main()


# %%
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def load_excel_data(file_path):
    """ Load an Excel file into a DataFrame. """
    return pd.read_excel(file_path)

def find_first_matching_class_label(main_df, user_df):
    """ Search for the first match in user_df against main_df and return the corresponding class label. """
    for index, row in user_df.iterrows():
        value1, value2 = row['Column1'], row['Column2']
        match = main_df[(main_df['Column1'] == value1) & (main_df['Column2'] == value2)]
        if not match.empty:
            return f"class label = '{match.iloc[0]['Class']}'"
    return "class label = 'No match found'"

def get_file_path_gui():
    """ Opens a GUI to select the file for classification. """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select an Excel file", filetypes=[("Excel files", "*.xlsx")])
    return file_path

def main():
    # File path for the main data
    main_file_path = 'combined_data_classified.xlsx'
    
    # Load the main DataFrame
    main_df = load_excel_data(main_file_path)
    
    # Get the path of the user input file via GUI
    user_file_path = get_file_path_gui()
    if user_file_path:
        user_df = load_excel_data(user_file_path)
        # Find the first matching class label
        class_label_result = find_first_matching_class_label(main_df, user_df)
        # Print the result
        print(class_label_result)
    else:
        print("No file selected or operation cancelled.")

if __name__ == "__main__":
    main()

# %% [markdown]
# # Build simple Deep learning model

# %%
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

# %%
num_classes = 5 # setting final output class
model = EEGAutoencoderClassifier(num_classes).to(device)
criterion = nn.NLLLoss() # Use NLLLoss function to optimize
optimizer = optim.Adam(model.parameters(), lr=0.0001) # Setting parameters learning rate = 0.001

# %%
num_epochs = 30 # setting training epochs (Number of training iterations)
for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# %%
model.eval() # Evaluate your model
correct = 0
total = 0

with torch.no_grad():
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# %%
plt.figure(figsize=(8, 6))
plt.bar(["Test Accuracy"], [accuracy * 100], color='blue')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.ylim(0, 100)
plt.show()


# %%
model = EEGAutoencoderClassifier(num_classes).to(device)
dummy_input = torch.randn(1, 64 * 795).to(device)
output = model(dummy_input)
graph = make_dot(output, params=dict(model.named_parameters()))

#Save the graph
#graph.render(filename='model_graph', format='png')
# Display the graph
graph.view()

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have defined the data loaders and model as per the provided code

# Define function to compute true positive rate matrix
def compute_true_positive_rate(model, test_loader, num_classes):
    model.eval()
    true_positive_counts = torch.zeros(num_classes, num_classes)
    class_counts = torch.zeros(num_classes)
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            for true_label, predicted_label in zip(labels, predicted):
                true_positive_counts[true_label][predicted_label] += 1
                class_counts[true_label] += 1

    true_positive_rate_matrix = true_positive_counts / class_counts[:, None]
    return true_positive_rate_matrix

# Instantiate model and load data as per the provided code

# Train the model
# Code for training the model as per the provided code

# Compute true positive rate matrix
true_positive_rate_matrix = compute_true_positive_rate(model, test_loader, num_classes)

# Class labels
classes = ['hello', 'help me', 'stop', 'thank you', 'yes']

# Plotting true positive rate matrix
plt.figure(figsize=(8, 6))
sns.heatmap(true_positive_rate_matrix.numpy(), annot=True, cmap='viridis', fmt=".2f", linewidths=.5, cbar=True)
plt.title('True Positive Rate Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(ticks=np.arange(5) + 0.5, labels=classes)
plt.yticks(ticks=np.arange(5) + 0.5, labels=classes)

# Create a dummy plot for colorbar
dummy = plt.imshow([[0,0],[0,0]], cmap='viridis')
plt.colorbar(dummy).set_label('True Positive Rate', rotation=270, labelpad=15, fontsize=12)

plt.tight_layout()
plt.show()


# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have defined the data loaders and model as per the provided code

# Define function to compute true positive rate matrix
def compute_true_positive_rate(model, test_loader, num_classes):
    model.eval()
    true_positive_counts = torch.zeros(num_classes, num_classes)
    class_counts = torch.zeros(num_classes)
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            for true_label, predicted_label in zip(labels, predicted):
                true_positive_counts[true_label][predicted_label] += 1
                class_counts[true_label] += 1

    true_positive_rate_matrix = true_positive_counts / class_counts[:, None]
    return true_positive_rate_matrix

# Define function to compute classification accuracy
def compute_accuracy(model, test_loader):
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
    return overall_accuracy, class_accuracy

# Instantiate model and load data as per the provided code

# Train the model
# Code for training the model as per the provided code

# Compute true positive rate matrix
true_positive_rate_matrix = compute_true_positive_rate(model, test_loader, num_classes)

# Compute classification accuracy
overall_accuracy, class_accuracy = compute_accuracy(model, test_loader)

# Class labels
classes = ['hello', 'help me', 'stop', 'thank you', 'yes']

# Plotting true positive rate matrix
plt.figure(figsize=(14, 6))

# Plotting true positive rate matrix
plt.subplot(1, 2, 1)
sns.heatmap(true_positive_rate_matrix.numpy(), annot=True, cmap='viridis', fmt=".2f", linewidths=.5, cbar=True)
plt.title('True Positive Rate Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(ticks=np.arange(5) + 0.5, labels=classes)
plt.yticks(ticks=np.arange(5) + 0.5, labels=classes)

# Create a dummy plot for colorbar
dummy = plt.imshow([[0,0],[0,0]], cmap='viridis')
plt.colorbar(dummy).set_label('True Positive Rate', rotation=270, labelpad=15, fontsize=12)

# Plotting classification accuracy
plt.subplot(1, 2, 2)
plt.bar(classes, class_accuracy, color='skyblue')
plt.title('Classification Accuracy of Classes')
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)

for i, acc in enumerate(class_accuracy):
    plt.text(i, acc, f'{acc:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have defined the data loaders and model as per the provided code

# Define function to create the model
def create_model(num_classes):
    model = EEGAutoencoderClassifier(num_classes).to(device) 
    return model

# Define function to train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader: 
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Define function to evaluate the model
def evaluate_model(model, test_loader):
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

    class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    overall_accuracy = 100 * correct / total
    return overall_accuracy, class_accuracy

# Instantiate model and load data as per the provided code

# Train the model
model = create_model(num_classes)
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs)

# Compute true positive rate matrix
true_positive_rate_matrix = compute_true_positive_rate(model, test_loader, num_classes)

# Compute classification accuracy
overall_accuracy, class_accuracy = evaluate_model(model, test_loader)

# Class labels
classes = ['hello', 'help me', 'stop', 'thank you', 'yes']

# Plotting true positive rate matrix
plt.figure(figsize=(14, 6))

# Plotting true positive rate matrix
plt.subplot(1, 2, 1)
sns.heatmap(true_positive_rate_matrix.numpy(), annot=True, cmap='viridis', fmt=".2f", linewidths=.5, cbar=True)
plt.title('True Positive Rate Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(ticks=np.arange(5) + 0.5, labels=classes)
plt.yticks(ticks=np.arange(5) + 0.5, labels=classes)

# Create a dummy plot for colorbar
dummy = plt.imshow([[0,0],[0,0]], cmap='viridis')
plt.colorbar(dummy).set_label('True Positive Rate', rotation=270, labelpad=15, fontsize=12)

# Plotting classification accuracy
plt.subplot(1, 2, 2)
plt.bar(classes, class_accuracy, color='skyblue')
plt.title('Classification Accuracy of Classes')
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)

for i, acc in enumerate(class_accuracy):
    plt.text(i, acc, f'{acc:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Define function to create the model
def create_model(num_classes):
    model = EEGAutoencoderClassifier(num_classes).to(device) 
    return model

# Define function to train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader: 
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Define function to evaluate the model
def evaluate_model(model, test_loader):
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

    class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    overall_accuracy = 100 * correct / total
    return overall_accuracy, class_accuracy

# Define function to compute true positive rate matrix
def compute_true_positive_rate(model, test_loader, num_classes):
    model.eval()
    true_positive_counts = torch.zeros(num_classes, num_classes)
    class_counts = torch.zeros(num_classes)
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            for true_label, predicted_label in zip(labels, predicted):
                true_positive_counts[true_label][predicted_label] += 1
                class_counts[true_label] += 1

    true_positive_rate_matrix = true_positive_counts / class_counts[:, None]
    return true_positive_rate_matrix

# Instantiate model and load data as per the provided code

# Train the model
model_before = create_model(num_classes)
true_positive_rate_matrix_before = compute_true_positive_rate(model_before, test_loader, num_classes)
overall_accuracy_before, class_accuracy_before = evaluate_model(model_before, test_loader)

model_after = create_model(num_classes)
train_model(model_after, train_loader, test_loader, criterion, optimizer, num_epochs)
true_positive_rate_matrix_after = compute_true_positive_rate(model_after, test_loader, num_classes)
overall_accuracy_after, class_accuracy_after = evaluate_model(model_after, test_loader)

# Class labels
classes = ['hello', 'help me', 'stop', 'thank you', 'yes']

# Plotting comparison
plt.figure(figsize=(18, 6))

# Plotting true positive rate matrix before
plt.subplot(1, 2, 1)
sns.heatmap(true_positive_rate_matrix_before.numpy(), annot=True, cmap='viridis', fmt=".2f", linewidths=.5, cbar=True)
plt.title('True Positive Rate Matrix Before')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(ticks=np.arange(5) + 0.5, labels=classes)
plt.yticks(ticks=np.arange(5) + 0.5, labels=classes)
dummy = plt.imshow([[0,0],[0,0]], cmap='viridis')
plt.colorbar(dummy).set_label('True Positive Rate', rotation=270, labelpad=15, fontsize=12)

# Plotting classification accuracy before
plt.subplot(1, 2, 2)
plt.bar(classes, class_accuracy_before, color='skyblue')
plt.title('Classification Accuracy Before')
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for i, acc in enumerate(class_accuracy_before):
    plt.text(i, acc, f'{acc:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 6))

# Plotting true positive rate matrix after
plt.subplot(1, 2, 1)
sns.heatmap(true_positive_rate_matrix_after.numpy(), annot=True, cmap='viridis', fmt=".2f", linewidths=.5, cbar=True)
plt.title('True Positive Rate Matrix After')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(ticks=np.arange(5) + 0.5, labels=classes)
plt.yticks(ticks=np.arange(5) + 0.5, labels=classes)
dummy = plt.imshow([[0,0],[0,0]], cmap='viridis')
plt.colorbar(dummy).set_label('True Positive Rate', rotation=270, labelpad=15, fontsize=12)

# Plotting classification accuracy after
plt.subplot(1, 2, 2)
plt.bar(classes, class_accuracy_after, color='skyblue')
plt.title('Classification Accuracy After')
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for i, acc in enumerate(class_accuracy_after):
    plt.text(i, acc, f'{acc:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# %%
class EEGAutoencoderClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EEGAutoencoderClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 795, 256), # Input dimension is 64 channels * 795 time points, and use 256 units for first NN layer
            nn.ReLU(), # Use ReLu function for NN training 
            nn.Linear(256, 128), # 256 NN units to 128 units
            nn.ReLU(),
            nn.Linear(128, 64),#  128 NN units to 64 units
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes), # num_classes is 5 (hello, help me, stop, thank you, and yes)
            nn.LogSoftmax(dim=1)  # Use LogSoftmax for multi-class classification
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.classifier(x)
        return x

num_classes = 5 # setting final output class
model = EEGAutoencoderClassifier(num_classes).to(device) 
criterion = nn.NLLLoss() # Use NLLLoss function to optimize
optimizer = optim.Adam(model.parameters(), lr=0.001) # Setting parameters learning rate = 0.001

num_epochs = 20 # setting training epochs (Number of training iterations)
for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader: 
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Define function to compute true positive rate matrix
def compute_true_positive_rate(model, test_loader, num_classes):
    model.eval()
    true_positive_counts = torch.zeros(num_classes, num_classes)
    class_counts = torch.zeros(num_classes)
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            for true_label, predicted_label in zip(labels, predicted):
                true_positive_counts[true_label][predicted_label] += 1
                class_counts[true_label] += 1

    true_positive_rate_matrix = true_positive_counts / class_counts[:, None]
    return true_positive_rate_matrix

# Compute true positive rate matrix
true_positive_rate_matrix = compute_true_positive_rate(model, test_loader, num_classes)

# Class labels
classes = ['hello', 'help me', 'stop', 'thank you', 'yes']

# Plotting true positive rate matrix
plt.figure(figsize=(8, 6))
sns.heatmap(true_positive_rate_matrix.numpy(), annot=True, cmap='viridis', fmt=".2f", linewidths=.5, cbar=True)
plt.title('True Positive Rate Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(ticks=np.arange(5) + 0.5, labels=classes)
plt.yticks(ticks=np.arange(5) + 0.5, labels=classes)

# Create a dummy plot for colorbar
dummy = plt.imshow([[0,0],[0,0]], cmap='viridis')
plt.colorbar(dummy).set_label('True Positive Rate', rotation=270, labelpad=15, fontsize=12)

plt.tight_layout()
plt.show()

# Define function to compute classification accuracy
def compute_accuracy(model, test_loader):
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
    return overall_accuracy, class_accuracy

# Compute classification accuracy
overall_accuracy, class_accuracy = compute_accuracy(model, test_loader)

# Plotting true positive rate matrix
plt.figure(figsize=(14, 6))

# Plotting true positive rate matrix
plt.subplot(1, 2, 1)
sns.heatmap(true_positive_rate_matrix.numpy(), annot=True, cmap='viridis', fmt=".2f", linewidths=.5, cbar=True)
plt.title('True Positive Rate Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(ticks=np.arange(5) + 0.5, labels=classes)
plt.yticks(ticks=np.arange(5) + 0.5, labels=classes)

# Create a dummy plot for colorbar
dummy = plt.imshow([[0,0],[0,0]], cmap='viridis')
plt.colorbar(dummy).set_label('True Positive Rate', rotation=270, labelpad=15, fontsize=12)

# Plotting classification accuracy
plt.subplot(1, 2, 2)
plt.bar(classes, class_accuracy, color='skyblue')
plt.title('Classification Accuracy of Classes')
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)

for i, acc in enumerate(class_accuracy):
    plt.text(i, acc, f'{acc:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


