import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.Tree import DecisionTreeClassifier

# Get the data

col_names = ['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
data = pd.read_csv("./datasets/iris.csv", skiprows=1, header=None, names=col_names)
data.head(10)

# Train-Test split

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

# Fit the model
classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=2)
classifier.fit(X_train,Y_train)
classifier.print_tree()

# Test the model: Predict and calculate accuracy
Y_pred = classifier.predict(X_test) 
print("Accuracy:",accuracy_score(Y_test, Y_pred))



