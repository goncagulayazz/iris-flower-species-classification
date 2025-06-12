import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

iris= load_iris()
x = iris.data
y = iris.target

print(iris.feature_names)
print(iris.target_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Predicting a new flower sample
# Example measurements: sepal length, sepal width, petal length, petal width
# You can change these values.
new_sample = [[7.6, 2.8, 5, 0.1]]

# Scale the new data (using the previously trained scaler)
new_sample_scaled = scaler.transform(new_sample)

# Make a prediction
prediction = knn.predict(new_sample_scaled)

# Print the predicted flower species
print("Predicted flower species:", iris.target_names[prediction[0]])