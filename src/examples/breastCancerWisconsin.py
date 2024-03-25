import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.Tree import DecisionTreeClassifier

# Get the data

col_names = ["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
data = pd.read_csv("./datasets/BreastCancerWisconsin.csv", skiprows=1, header=None, names=col_names)
data.head(10)

# Train-Test split
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Split the dataset into features and target variable
X = data.iloc[:, 2:].values  # Exclude 'id' and 'diagnosis' columns for features
Y = data['diagnosis'].values  # Use 'diagnosis' column as the target variable
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

# Fit the model
classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=2)
classifier.fit(X_train,Y_train)
classifier.print_tree()

# Test the model: Predict and calculate accuracy
Y_pred = classifier.predict(X_test) 
print("Accuracy:",accuracy_score(Y_test, Y_pred))

# Testing decision making on synthetic data

new_data_point_str = '17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189'

# Convert the string to a numpy array of floats
new_data_point = np.array([float(x) for x in new_data_point_str.split(',')])

#new_data_point = new_data_point.reshape(1, -1)

prediction = classifier.predict_single(new_data_point)
print("Predicted Class:", prediction)
