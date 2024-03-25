# Node Class

class Node:
    def __init__(self,feature_index = None, threshold = None, left = None, right = None, info_gain = None, value = None) :
        ''' Constructor '''
        
        # For decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # For leaf node
        self.value = value
        


# Explanations:
# The decision node contains a decision that is defined by 'feature_index' and the 'threshold' value for that particular feature
# 'left' 'right' accesses the left or right child node respectively

