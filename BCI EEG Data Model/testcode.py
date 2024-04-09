# Importing necessary libraries
from sklearn import svm
from sklearn.metrics import accuracy_score

# Initialize SVM classifier
clf = svm.SVC(kernel='linear')

# Training the classifier
clf.fit(X_train, y_train)

# Predicting labels for test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
