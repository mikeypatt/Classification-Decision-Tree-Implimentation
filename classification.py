##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the
# DecisionTreeClassifier
#http://mlwiki.org/index.php/Cost-Complexity_Pruning
##############################################################################

import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import matplotlib.patches as pp
from matplotlib.collections import PatchCollection
from tempfile import TemporaryFile
from eval import Evaluator
import copy
from scipy.stats import chi2


class DecisionTreeClassifier(object):
    """
    A decision tree classifier
   
    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained
   
    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X
   
    """

    def __init__(self):
        self.is_trained = False
        self.tree = False

    def load_data(self, filename):
        """
        Function to load data from file
        Args:
            filename (str) - name of .txt file you are loading data from
        Output:
            (x, y) (tuple) - x: 2D array of training data where each row
            corresponds to a different sample and each column corresponds to a
            different attribute.
                            y: 1D array where each index corresponds to the
            ground truth label of the sample x[index][]
        """
        #load data to single 2D array
        data_set = np.loadtxt(filename, dtype=str, delimiter=',')
        num_samp = len(data_set) #number of sample_is
        num_att = len(data_set[0]) #number of attributes
       
        #create attribute and label arrays filled with zeros
        x = np.zeros((num_samp, num_att - 1),dtype=int)
        y = np.zeros((num_samp,1), dtype=str)
       
        #fill arrays with correct values
        for sample_i in range(num_samp):
            for attribute_i in range(num_att):
                if attribute_i < (num_att - 1):
                    x[sample_i,attribute_i] = data_set[sample_i,attribute_i]
                else:
                    y[sample_i] = data_set[sample_i,attribute_i]
        return x, y
   
   
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')


    def evaluate_input(self,x,y):

        alphabet, count = np.unique(y, return_counts=True)
        alphabet_count = np.zeros((len(alphabet)))
        alphabet_proportions_1 = count / len(y)
        print("alphabet:")
        print(alphabet)
        print("alphabet proportions:")
        print(alphabet_proportions_1)

        length, width = np.shape(x)
        print('test')
        print(np.amax(x[:, 0]))
        minimum_attribute_value = np.amin(x, axis=0)
        maximum_attribute_value = np.amax(x, axis=0)
        attribute_ranges_1 = maximum_attribute_value - minimum_attribute_value
        print("minimum:")
        print(minimum_attribute_value)
        print("maximum:")
        print(maximum_attribute_value)
        print("attribute ranges:")
        print(attribute_ranges_1)

    def train(self, x, y,pruning=False,name = False):

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
        "Training failed. x and y must have the same number of instances."

        if not pruning and not name:
            #Method 1
            tree_1 = self.induce_decision_tree(x,y,False)

            #Method 1 with Prepruning
            tree_1_pruned = self.induce_decision_tree(x,y,False,True)

            #Method 2
            tree_2 = self.induce_decision_tree(x,y)
            self.tree = tree_2

            #Method 2 with Prepruning
            tree_2_pruned = self.induce_decision_tree(x,y,True,True)
     
            np.save('initial_tree.npy',tree_1)
            np.save('initial_tree_pruned.npy',tree_1_pruned)
            np.save('simple_tree.npy',tree_2)
            np.save('simple_tree_pruned.npy',tree_2_pruned)
        else:
            tree_1 = self.induce_decision_tree(x,y)
            np.save(name, tree_1)
        
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self
    
    def induce_decision_tree(self, x, y,suggested_method=True,pre_pruning=False):

        # Check whether they all equal the same thing
        labels = np.unique(y)
        length = len(labels)
        if length == 1:
            return labels[0]

        # Nothing in the data set
        if len(x) == 0:
            return None
        
        #Choosing between the two methods of splitting the tree
        if not suggested_method:
            node = self.find_best_node_iterative(x, y)
        else:
            node = self.find_best_node_simple(x, y)
            
        child_1, child_2 = self.split_dataset(node)

        if len(child_1["attributes"]) == 0 or len(child_2["attributes"]) == 0:
            return self.terminal_leaf(node["data"]["labels"])[0]
       
        left_probability = len(child_1["labels"])/ len(node["data"])
        right_probability = len(child_2["labels"])/len(node["data"])
     
        parent_labels,parent_count = np.unique(node["data"]["labels"],return_counts=True)
        node["majority_class"] = self.terminal_leaf(node["data"]["labels"])[0]
        del (node["data"])
       
        left_child_labels = self.count_occurrences(parent_labels,child_1["labels"])
        right_child_labels = self.count_occurrences(parent_labels,child_2["labels"])
        
        node["num_children"] = len(child_1["labels"]) + len(child_2["labels"])
        node["parentlabels"] = parent_labels
        node['K'] = self.compute_k(left_probability,left_child_labels,
                                   right_probability,right_child_labels,parent_count)
       
       #CHI2 prepruning - calculating the significance of the split
        if pre_pruning :
            df = len(parent_labels) -1  
            if node['K'] <= chi2.isf(0.05,df):
                return node["majority_class"]
            
        #Recursively call the function on the split dataset
        node["left"] = self.induce_decision_tree(child_1["attributes"], child_1["labels"],suggested_method,pre_pruning)
        node["right"] = self.induce_decision_tree(child_2["attributes"], child_2["labels"],suggested_method,pre_pruning)

        return node
    
    
    def count_occurrences(self,parent_occurrences,child_data):

        child_occurrences = np.zeros((1,len(parent_occurrences)))

        for i in range(len(parent_occurrences)):
            count = 0
            for j in range(len(child_data)):
                if child_data[j] == parent_occurrences[i]:
                    count+=1
            child_occurrences[0,i] = count

        return child_occurrences

    def compute_k(self,left_probability,left_child_labels,right_probability,
                  right_child_labels,parent_labels):

        K = 0
        for i in range(len(left_child_labels)):
            K += ((left_child_labels[0][i]-(parent_labels[i]*left_probability))**2)/(parent_labels[i]*left_probability)
        for i in range(len(right_child_labels)):
            K += ((right_child_labels[0][i] - (parent_labels[i] * right_probability))**2) / (parent_labels[i] * right_probability)

        return K

    def find_total_entropy(self, y):
        """
        Function to find the total entropy in label array, y
        Args:
            y (1D array) -  where each index corresponds to the
            ground truth label of the sample x[index][]
        Output:
            entropy (float) - calculated entropy
        """
        num_samples = len(y)
        if num_samples == 0:
            return 0
        #find probabilities of each label
        labels_unique, label_count = np.unique(y, return_counts=True)
        label_probabilities = label_count / num_samples
        #find entropy using probabilities
        information = label_probabilities * np.log2(label_probabilities)
        entropy = -1 * np.sum(information)
        return entropy
   
    def terminal_leaf(self, data_set):
        """
        Returns the most frequent value in 1D numpy array
        Args:
            data_set (1D array)
        Output:
           most common value in array
        """
        labels, count = np.unique(data_set, return_counts=True)
        index = np.argmax(count)
        return data_set[index]

    def find_best_node_iterative(self, x, y):
        """
        Function to find the attribute and value on which a binary partition of
        the data can be made to maximise information gain (entropy reduction)
        Args:
            x (2D array) - 2D array of training data where each row
            corresponds to a different sample and each column corresponds to a
            different attribute.
            y (1D array) -  where each index corresponds to the
            ground truth label of the sample x[index][]
        Output:
            node (dict) - dictionary contains information on partition,
            including value and attribute to partition over.
        """        
        stored_value = 0
        stored_attribute = 0
        best_gain = 0

        num_samples, num_attributes = np.shape(x)

        root_entropy = self.find_total_entropy(y)

        for attribute in range(num_attributes):
            """
            iterate through each attribute to find which attribute to
            split data on --> find which attribute partition causes the
            greatest information gain.
            """
            #find the min and max value of that attribute
            minimum_attribute_value = int(np.amin(x[:,attribute]))
            maximum_attribute_value = int(np.amax(x[:,attribute]))

            for split_value in range(minimum_attribute_value,
                                     maximum_attribute_value+1):
                """
                iterate through each possible divide of that attribute
                (where to halve data) to find what value of that attribute to
                split data on --> find which partition causes the greatest
                information gain.
                """
                #first half
                subset_1_x = []
                subset_1_y = []

                #second half
                subset_2_x = []
                subset_2_y = []

                for row in range(num_samples):

                    #perform separation of data into two halves
                    if x[row][attribute] < split_value:
                        subset_1_x.append(x[row,:])
                        subset_1_y.append(y[row])
                    else:
                        subset_2_x.append(x[row,:])
                        subset_2_y.append(y[row])

                #find entropy of each half
                subset_1_entropy = self.find_total_entropy(subset_1_y)
                subset_2_entropy = self.find_total_entropy(subset_2_y)

                #normalise entropy for each of sub datasets
                subset_1_entropy_normalised = \
                    subset_1_entropy * len(subset_1_x) /num_samples
                subset_2_entropy_normalised = \
                    subset_2_entropy * len(subset_2_x) /num_samples

                # get total entropy
                total_split_entropy = (subset_1_entropy_normalised +
                                      subset_2_entropy_normalised)

                # get information gain
                information_gain = root_entropy - total_split_entropy

                # check whether it is bigger than the previous
                if (information_gain > best_gain):
                    stored_attribute = attribute
                    stored_value = split_value
                    best_gain = information_gain

        data = {"attributes": x, "labels": y}

        #Returns the node
        return {"value": stored_value, "attribute": stored_attribute, "gain": best_gain, "data": data, "left": None,
                "right": None,'K':None,"majority_class": None, "is_checked": False,"parentlabels":None,"num_children":None}
    
    
    def find_best_node_simple(self, x, y):
        """
        Function to find the attribute and value on which a binary partition of
        the data can be made to maximise information gain (entropy reduction)
        Args:
            x (2D array) - 2D array of training data where each row
            corresponds to a different sample and each column corresponds to a
            different attribute.
            y (1D array) -  where each index corresponds to the
            ground truth label of the sample x[index][]
        Output:
            node (dict) - dictionary contains information on partition,
            including value and attribute to partition over.
        """        
        stored_value = 0
        stored_attribute = 0
        best_gain = 0

        num_samples, num_attributes = np.shape(x)

        root_entropy = self.find_total_entropy(y)

        for attribute in range(num_attributes):
            
            """
            iterate through each attribute whilst sorting the date
            if the class label changes split create a boundar at that
            dataset
            """
            if y.ndim == 1:
                y = np.reshape(y, (len(y), 1))
            
            entire_data = np.append(x,y,axis=1)
            entire_data = np.array(sorted(entire_data,key=lambda on_attribute: int(on_attribute[attribute])))
            unique_values = np.unique(entire_data[:,attribute])
            
            if len(unique_values) ==1:
                continue
            
            for value in unique_values[:-1]:
                #Finds the positions where the sorted dataset equals the current value
                possible_positions = np.where(entire_data[:,attribute] == value)
                
                #Gets the first position of the value
                node_position = possible_positions[0][0]
                
                #Calculates the entropy before the node position
                subset_1_entropy = self.find_total_entropy(entire_data[:node_position,-1])
                #Calculates the entropy after the node position
                subset_2_entropy = self.find_total_entropy(entire_data[node_position:,-1])
                
                #normalise entropy for each of sub datasets
                subset_1_entropy_normalised = \
                        subset_1_entropy * node_position/num_samples
                subset_2_entropy_normalised = \
                        subset_2_entropy * (len(entire_data)-node_position)/num_samples
                    
                # get total entropy
                total_split_entropy = (subset_1_entropy_normalised + 
                                        subset_2_entropy_normalised)
                    
                # get information gain
                information_gain = root_entropy - total_split_entropy
                    
                
                # check whether it is bigger than the previous
                if (information_gain > best_gain):
                    stored_attribute = attribute
                    stored_value = int(value)
                    best_gain = information_gain

        data = {"attributes": x, "labels": y}

        #Returns the node
        return {"value": stored_value, "attribute": stored_attribute, "gain": best_gain, "data": data, "left": None,
                "right": None,'K':None,"majority_class": None, "is_checked": False,"parentlabels":None,"num_children":None}

    def split_dataset(self, node):
        """
        Function to split the data in a node according to partition defined by
        find_best_node_ideal
        Args:
            node (dict) - node which details how to split data
        Output:
            (left, right) (tuple) - dataset split into two halves as defined by
            node["attribute"] and node["value"]
        """
        dataset = node["data"]
        x = dataset["attributes"]
        y = dataset["labels"]
        attribute = node["attribute"]
        split_value = node["value"]
        left_x = []
        right_x = []
        left_y = []
        right_y = []

        for row in range(len(x)):

            if x[row][attribute] < split_value:
                left_x.append(x[row,:])
                left_y.append(y[row])
            else:
                right_x.append(x[row,:])
                right_y.append(y[row])

        left = {"attributes": np.array(left_x), "labels": np.array(left_y)}
        right = {"attributes": np.array(right_x), "labels": np.array(right_y)}

        return left, right

    def predict(self, x,other_tree = False):

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)
       
        # load the classifier
        if not other_tree:
            if self.tree == False:
                tree = np.load('simple_tree.npy',allow_pickle = True).item()  
            else:
                tree = self.tree
        else:
            tree = other_tree
       
        for j in range(0, len(x)):
            predictions[j] = self.recursive_predict(tree, x[j,:])

        # remember to change this if you rename the variable
        return predictions

    def recursive_predict(self, tree, attributes):
        """
        Function to predict the label of a sample based on its attributes
        Args:
            tree (dict) - trained decision tree
            attributes (2D array) - 2D array of test data where each row
            corresponds to a different sample and each column corresponds to a
            different attribute.
        Output:
            string object of label prediction
        """
        #if leaf found return label str
        if not isinstance(tree, dict):
            return tree

        # Check the required attribute is greater or less than the node split
        # then recursively call function on tree from child node.
        if attributes[tree["attribute"]] < tree["value"]:
            return self.recursive_predict(tree["left"], attributes)
       
        return self.recursive_predict(tree["right"], attributes)



    def node_height(self,node):

        if not isinstance(node, dict):
            return 0

        return 1 + max(self.node_height(node["left"]),self.node_height(node["right"]))


    def print_tree(self,tree,name):

        #Attrubute column labels
        attributes = {0:"x-box",1:"y-box",2:"width",3:"high",
                      4:"onpix",5:"x-bar",6:"y-bar",7:"x2bar",
                      8:"y2bar",9:"xybar",10:"x2ybr",11:"xy2br",
                      12:"x-ege",13:"xegvy",14:"y-ege",15:"yegvx"}

        fig,ax = plt.subplots(nrows = 1,ncols=1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        y = 20000
        x1 = 0
        x2 = 20000
        mid_x = (x1 + x2)/2
        height = 50
        width = 1500
        depth = 0

        patches = []
        patches.append(pp.Rectangle((mid_x-width/2,y-height),width,height,
                                    color = 'blue'))
        annotation = "Root:\n" + "\nAttSplit:\n"+ attributes[tree["attribute"]]
        annotation += "<" + str(tree["value"])
        annotation +=  "\nIG:"+str(np.round(tree["gain"],3))
        center_x = mid_x
        center_y = y - height/2.0
        ax.annotate(annotation, (center_x,center_y), color='white',
                    weight='bold',fontsize=4.5, ha='center', va='center')

        self.recursive_print(tree["left"],mid_x,x1,mid_x,y-2*height,
                             attributes,depth+1,patches,ax)
        self.recursive_print(tree["right"],mid_x,mid_x,x2,y-2*height,
                             attributes,depth+1,patches,ax)

        ax.add_collection(PatchCollection(patches,match_original=True))
        ax.set_xlim((0,3000))
        ax.set_ylim((0,3000))
        ax.autoscale()
        fig.tight_layout
        plt.savefig(name,bbox_inches='tight')
        #plt.show()


    def recursive_print(self,node,parent_center_x,x1,x2,y,attributes,
                        depth,patches,ax):

        mid_x = (x1 + x2)/2
        height = 50
        width = 1500

        if not isinstance(node, dict):
            #print a leaf node (different colour

            patches.append(pp.Rectangle((mid_x-width/2,y-height),width,
                                        height,color = 'black'))
            annotation = "Leaf Node \nLabel = " + str(node)
            center_x = mid_x
            center_y = y - height/2.0
           
            ax.annotate(annotation, (center_x, center_y), color='white',
                        weight='bold',fontsize=4.5, ha='center', va='center')
           
            plt.plot([parent_center_x,mid_x],[y+height,y],'black',
                    linestyle=':',marker='')
            return

        else:

            patches.append(pp.Rectangle((mid_x-width/2,y-height),
                                        width,height,color = 'blue'))
            annotation = "IntNode:\n" + "\nAttSplit:\n"+ attributes[node["attribute"]]
            annotation += "<" + str(node["value"])
            annotation +=  "\nIG:"+str(np.round(node["gain"],3))
            center_x = mid_x
            center_y = y - height/2.0
            ax.annotate(annotation, (center_x,center_y), color='white',
                        weight='bold',fontsize=4.5, ha='center', va='center')
            plt.plot([parent_center_x,mid_x],[y+height,y],'black',linestyle=':',marker='')

        #Maximum depth to print out taken from Piazza
        if depth ==3:
            return

        #Create the annotation to place into the center of the rectangle
        annotation = "depth:"+str(0) + " " + attributes[node["attribute"]]
        annotation += "<" + str(node["value"])
        left_height = self.node_height(node["left"]) + 1
        right_height = self.node_height(node["right"]) + 1
        weight = left_height/(left_height + right_height)

        weighted_x = x1 + weight*(x2-x1)

        self.recursive_print(node["left"],mid_x,x1,weighted_x,
                             y-2*height,attributes,depth+1,patches,ax)
        self.recursive_print(node["right"],mid_x,weighted_x,x2,
                             y-2*height,attributes,depth+1,patches,ax)
       



