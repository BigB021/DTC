# Decision Tree Classifier Project

This project implements a custom Decision Tree Classifier from scratch, designed to illustrate the basic principles of decision tree learning algorithms. The project structure organizes code into modules for ease of use and extension. It includes a Node class to represent decision nodes, a Tree class as the core classifier, and an example module demonstrating how to apply the classifier to a dataset. This README provides an overview of the project structure, instructions on how to use the Decision Tree Classifier, and an example application on a dataset.

Most of the code was obtained from [Normalized Nerd](https://www.youtube.com/@NormalizedNerd)

## Project Structure

The project is organized into several directories and files:

src/: Contains the source code of the project.
node.py: Defines the Node class used in the decision tree.
tree.py: Implements the Tree class, which is the decision tree classifier.
examples/: Directory for example scripts that demonstrate the usage of the classifier.
bitcoin.py: An example script that applies the classifier to a Bitcoin dataset.
datasets/: Contains CSV files with datasets to be used with the classifier.
Installation

To run this project, you need Python 3.x and the following Python packages: numpy, pandas, sklearn. You can install them using pip:

 **pip install numpy pandas scikit-learn**

## Usage

**Prepare Your Dataset:** Ensure your dataset is in a CSV format and placed within the datasets directory. The dataset should be structured appropriately for your classification problem.
**Implement the Decision Tree Classifier:**
Import the Tree class from the tree.py module in your script. Here's a basic outline of how to use the classifier:

from src.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
data = pd.read_csv("./datasets/your_dataset.csv")

# Preprocess and prepare your data here
# X = data[feature_columns].values
# Y = data[target_column].values

# Split your data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Initialize the Decision Tree Classifier
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)

# Fit the model on the training data
classifier.fit(X_train, Y_train)

# Predict on the testing data
Y_pred = classifier.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy}")


## Running Examples:
To run the provided example, navigate to the project directory and use:

**python -m src.examples.bitcoin**

This will execute the bitcoin.py example script, applying the Decision Tree Classifier to the Bitcoin dataset included in the project, and print the accuracy of the model.
Contributing

We welcome contributions to improve this project! Please feel free to fork the repository, make changes, and submit pull requests.

License

This project is open-source and available under the MIT License. See the LICENSE file for more details.

[Original Project](https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20classification.ipynb)