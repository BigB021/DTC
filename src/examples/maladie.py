import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.Tree import DecisionTreeClassifier

# Assuming the dataset is saved in a CSV file named 'dataset.csv'
data = pd.read_csv("./datasets/maladie.csv")

# Encode categorical features using Label Encoding
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for column in ['Maladie', 'Coloration', 'Paroi']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separate features and target variable
X = data.drop(['No', 'Maladie'], axis=1).values
Y = data['Maladie'].values

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and fit the Decision Tree classifier
classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=3)
classifier.fit(X_train, Y_train)
classifier.print_tree()

# Make predictions and evaluate the model
Y_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

print("Accuracy:", accuracy)

