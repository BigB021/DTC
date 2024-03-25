# Tree.py
# Decision Tree Class

from tkinter import Y
import numpy as np
from src.Node import Node



class DecisionTreeClassifier():
    def __init__(self, min_samples_split = 2, max_depth = 2) :
        ''' Constructor '''
        
        # Initialize the root of the tree
        self.root = None
        
        # Stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' Recursive function to build the tree '''
    
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # Split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # Find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # Check if a beneficial split was found
            if 'info_gain' in best_split and best_split['info_gain'] > 0:
                # Recur right
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth + 1)
                # Recur left
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth + 1)
                # Return decision node
                return Node(best_split['feature_index'], best_split['threshold'],
                            left_subtree, right_subtree, best_split['info_gain'])
            # If no beneficial split was found, treat the current node as a leaf node

        # Compute leaf node (either because no split improves the gain, or stopping condition met)
        leaf_value = self.calculate_leaf_value(Y)
        # Return leaf node
        return Node(value=leaf_value)

    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split 
    
    def split(self,dataset,feature_index, threshold):
        ''' Function to split the data '''
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left,dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode = "entropy"):
        ''' Function to calculate information gain '''
        
       # Probability of child node relative to parent
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        
        if mode == "gini":
            gain = self.entropy(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l *self.entropy(l_child) + weight_r *self.entropy(r_child) )
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            if p_cls > 0 :
                entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self,y):
        ''' Function to calculate gini index '''
        # Gini = 1 − ∑(pi)2
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
            
        return 1 - gini
            
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        Y = list(Y)
        return max(Y, key=Y.count)

    
    def print_tree(self, tree=None, indent="", last=True):
        ''' Function to print the tree in a more visually appealing way '''
        if tree is None:
            tree = self.root

        prefix = "└── " if last else "├── "
        if tree.value is not None:
            print(indent + prefix + str(tree.value))
        else:
            print(indent + prefix + "X" + str(tree.feature_index) + " < " + str(tree.threshold) + "? (Gain: " + str(tree.info_gain) + ")")
            if tree.left or tree.right:
                if tree.left:  # Print left child
                    self.print_tree(tree.left, indent + ("    " if last else "│   "), last=not tree.right)
                if tree.right:  # Print right child
                    self.print_tree(tree.right, indent + ("    " if last else "│   "), last=True)

            
    def fit(self,X,Y):
        ''' Function to train the tree '''
        Y = Y.reshape(-1, 1)  
        dataset = np.concatenate((X,Y), axis = 1)
        self.root = self.build_tree(dataset)
    
    def predict(self,X):
        ''' Function to predict new dataset '''
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x, tree):
        ''' Function to predict a single data point '''
        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x,tree.left)
        else:
            return self.make_prediction(x,tree.right)
        
    def predict_single(self, data_point, node=None):
        ''' Predict the class for a single data point. ''' 
        if node is None:
            node = self.root
        if node.value is not None:
            return node.value
        if data_point[node.feature_index] < node.threshold:
            return self.predict_single(data_point, node.left)
        else:
            return self.predict_single(data_point, node.right)

                    
                 
