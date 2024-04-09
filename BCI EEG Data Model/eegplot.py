import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample accuracy data
accuracy = 82.6

# Sample true positive rate matrix (assuming it's a numpy array)
true_positive_rate_matrix = np.array([[0.9, 0.8, 0.7, 0.75, 0.85],
                                      [0.85, 0.75, 0.8, 0.9, 0.7],
                                      [0.8, 0.85, 0.9, 0.75, 0.7],
                                      [0.75, 0.7, 0.8, 0.85, 0.9],
                                      [0.7, 0.75, 0.85, 0.8, 0.9]])

# Class labels
classes = ['hello', 'help me', 'stop', 'thank you', 'yes']

# Plotting true positive rate matrix
plt.figure(figsize=(8, 6))
sns.heatmap(true_positive_rate_matrix, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5)
plt.title('True Positive Rate Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(ticks=np.arange(5) + 0.5, labels=classes)
plt.yticks(ticks=np.arange(5) + 0.5, labels=classes)
plt.show()

