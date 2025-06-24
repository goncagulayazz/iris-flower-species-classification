# Import necessary libraries
import numpy as np
import pandas as pd

# Import machine learning tools from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the Iris dataset
iris = load_iris()
# Features: sepal length, sepal width, petal length, petal width
x = iris.data
# Target labels: 0 = setosa, 1 = versicolor, 2 = virginica
y = iris.target

# Print the names of the features and target classes
print(iris.feature_names)
print(iris.target_names)

# Split the data into training and testing sets
# 80% for training, 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize a StandardScaler to normalize the feature values
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
x_train = scaler.fit_transform(x_train)

# Use the same scaler to transform the test data
x_test = scaler.transform(x_test)

# Initialize the K-Nearest Neighbors (KNN) classifier
# n_neighbors=3 means it will consider the 3 closest neighbors to make a decision
knn = KNeighborsClassifier(n_neighbors=3)

# Train the KNN classifier with the scaled training data
knn.fit(x_train, y_train)

# Predict the labels for the test dataset
y_pred = knn.predict(x_test)

# Evaluate the model performance

# Print the accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print the confusion matrix
# Rows = actual classes, Columns = predicted classes
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Print a detailed classification report
# Includes precision, recall, f1-score for each class
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))


# Predicting a new, unseen flower sample

# Define a new flower sample with its 4 feature values: [sepal length, sepal width, petal length, petal width]
# You can modify these values to test different inputs
new_sample = [[7.6, 2.8, 5, 0.1]]

# Important: Scale the new sample using the *already fitted* scaler
# This ensures the new data is on the same scale as training data
new_sample_scaled = scaler.transform(new_sample)

# Predict the class label for the new sample
prediction = knn.predict(new_sample_scaled)

# Print the predicted class name (species)
print("Predicted flower species:", iris.target_names[prediction[0]])
