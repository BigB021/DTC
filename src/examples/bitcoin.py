import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.Tree import DecisionTreeClassifier

# Load and preprocess data
col_names = ['date', 'exchange_rate']
data = pd.read_csv("./datasets/bitcoin.csv", skiprows=1, names=col_names)
data['prev_day_rate'] = data['exchange_rate'].shift(1)
data.dropna(inplace=True)  # Drop the first row as it now has a NaN value

# Create a target variable for classification (1: Increase, 0: Decrease)
data['target'] = (data['exchange_rate'] > data['prev_day_rate']).astype(int)

# Optionally, filter out days with a 0 exchange rate if they are not relevant
data = data[data['exchange_rate'] > 0]

# Define features and target; drop the first row as it will not have a 'target'
X = data.iloc[1:, :].drop(['date', 'exchange_rate', 'target'], axis=1).values
Y = data.iloc[1:]['target'].values.reshape(-1, 1)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

# Initialize, fit, and evaluate the model
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train, Y_train)
classifier.print_tree()

# Predict and calculate accuracy
Y_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)
