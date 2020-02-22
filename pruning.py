"""
Prunining Class File - outlines 3 different methods

"""
import numpy as np
import copy
from classification import DecisionTreeClassifier
from eval import Evaluator
from scipy.stats import chi2

class Pruning(object):
    

    #####COST COMPLEXITY PRUNING METHOD########
    def cost_complexity_pruning(self,node):

        trees = np.array([], dtype=np.object)
        tree_copy = node.copy()
        i = 0

        while isinstance(tree_copy['left'],dict) or isinstance(tree_copy['right'],dict):

            trees = np.append(trees, self.prune_tree(tree_copy))
            tree_copy = copy.deepcopy(tree_copy)
        return trees

    def prune_tree(self,node):

        if not isinstance(node['left'],dict) and not isinstance(node['right'],dict):
            return node['majority_class']
        elif isinstance(node['left'],dict):
            node['left'] = self.prune_tree(node['left'])
            return node
        elif isinstance(node['right'], dict):
            node['right'] = self.prune_tree(node['right'])
            return node

        return node

    def calculate_best_pruned_tree(self,original_tree,trees,x_val,y_val):
       
        eval = Evaluator()
        classifier = DecisionTreeClassifier()
        classifier.is_trained = True
        original_predictions = classifier.predict(x_val,original_tree)
        original_error = self.get_apperent_error_rate(original_predictions,y_val)

        stored_j=0
        initial_alpha = 0
        previous_diff = 0
        previous_alpha = 0

        #go through each tree and compute the ratio of caculated error (right/total)
        for j in range(1,len(trees)):
            
            predictions = classifier.predict(x_val,trees[j])
            error = self.get_apperent_error_rate(predictions,y_val)
            original_number_leaves = self.count_leaves(original_tree)
            number_of_leaves = self.count_leaves(trees[j])
            alpha = (original_error - error)/(original_number_leaves - number_of_leaves)
            diff = initial_alpha - alpha
            if diff < previous_diff:
                stored_j = j
                previous_alpha = alpha
               
        return trees[stored_j], previous_alpha
    
                      
    def get_apperent_error_rate(self,predictions,y_test):
       
        count = 0
        for j in range(len(predictions)):
           
            if predictions[j] != y_test[j]:
                count+=1
       
        return count/len(predictions)
    
  ##### END OF COST COMPLEXITY PRUNING METHOD########  
  
  ##### POST CHI^2 PRUNING METHOD########
   
    def post_chi_pruning(self,tree):
        
        """
        Function impliments the same method as the pre-pruning
        CHI^2 implemted above on a completed tree"
        
        As the K-Values were already calucalted and contained in 
        the node it was simple algorithm to perfrom.
        
        """
        tree = copy.deepcopy(tree)
        tree["left"] = self.climb_tree(tree["left"])
        tree["right"] = self.climb_tree(tree["right"])
        
        return tree

        
    def climb_tree(self,node):
        
        if not isinstance(node,dict):
            return node
        
        #If it has reached a node with leaf nodes as children
        #if not isinstance(node['left'],dict) and not isinstance(node['right'],dict):
        df = len(node["parentlabels"])-1
        if node['K'] <= chi2.isf(0.05,df):
            return node["majority_class"]
                        

        node["left"] = self.climb_tree(node["left"])
        node["right"] = self.climb_tree(node["right"])
        
        return node 
            
        
    ##### END OF POST CHI^2 PRUNING METHOD########             

     ##### REDUCED ERROR PRUNINIG METHOD########
   
    def prune_tree_reduced_error(self, tree, x_val, y_val):
        """
        Function to accept prunes which increase the tree's accuracy, otherwise
        ignore
        Args:
            tree (dict) - tree to be pruned
            x_val (2D array) - 2D array of attributes of validation set where
                each row is a differnt sample and each column is a differnt
                attribute
            y_val (1D array) - 1D array of correct labels for x_val validation
                data
        Output:
            tree (dict or str) - tree pruned such that any additional pruning
                would lower predictive accuracy on validation set.
        """
        classifier = DecisionTreeClassifier()
        classifier.is_trained = True
        predictions = classifier.predict(x_val)
        eval = Evaluator()
        confusion = eval.confusion_matrix(predictions, y_val)
        root_accuracy = eval.accuracy(confusion)
        print("Results on Validation set")
        print("Original Accuracy: ", root_accuracy)

        is_pruned = True
        while (is_pruned and isinstance(tree, dict)):
            #make copy of tree then attempt to prune copy
            tree_copy = copy.deepcopy(tree)
            (is_pruned, tree_copy, tree) = self.prune(tree_copy, tree)
            if is_pruned:
                #compare accuracy of pruned tree to original
                new_predictions = classifier.predict(x_val,tree_copy)
                new_confusion = eval.confusion_matrix(new_predictions, y_val)
                new_accuracy = eval.accuracy(new_confusion)
                if new_accuracy >= root_accuracy:
                    #if greater or equal accuracy make tree = copy
                    root_accuracy = new_accuracy
                    tree = copy.deepcopy(tree_copy)
       
        print("New Accuracy: ", root_accuracy)
        return tree
   
    def prune(self, tree_copy, tree):
        """
        Recursive function to replace first node with two leaves as the
        majority class of that node. It will only do this if the node has not
        already been checked by the outer algorithm.
        Args:
            tree (dict or str) - current tree
            tree_copy (dict or str) - copy of tree
        Output:
            is_pruned, tree_copy, tree
           
            is_pruned (bool) - was a node found that could be pruned
            tree_copy (dict or str) - pruned tree
            tree (dict or str) - unpruned tree with node which was pruned in
                tree_copy marked as checked: tree["is_checked"] = True
        """
        is_left_leaf = not isinstance(tree["left"], dict)
        is_right_leaf = not isinstance(tree["right"], dict)

        #if both children leaves
        if is_left_leaf and is_right_leaf and not tree["is_checked"]:
            tree_copy = copy.deepcopy(tree_copy["majority_class"])
            tree["is_checked"] = True
            return True, tree_copy, tree

        #if left not leaf
        if not is_left_leaf:
            branch_a = self.prune(tree_copy["left"], tree["left"])
            #save updated trees
            tree_copy["left"] = copy.deepcopy(branch_a[1])
            tree["left"] = copy.deepcopy(branch_a[2])
            if branch_a[0]:
                #return tree if pruning done otherwise try right branch
                return True, tree_copy, tree

        #if right not leaf
        if not is_right_leaf:
            branch_a = self.prune(tree_copy["right"], tree["right"])
            #save updated tree
            tree_copy["right"] = copy.deepcopy(branch_a[1])
            tree["right"] = copy.deepcopy(branch_a[2])
            if branch_a[0]:
                #return tree if pruning done
                return True, tree_copy, tree

        return False, tree_copy, tree
   
    def count_leaves(self, tree, count = 0):
        """
        Recursive function to count number of leaves in a tree
        Args:
            tree (dict) - tree to test
            count (int) - current count for number of leaves (used in recusive
                  call)
        Output:
            count (int) - number of leaves in tree
        """

        is_left_leaf = not isinstance(tree["left"], dict)
        is_right_leaf = not isinstance(tree["right"], dict)

        #if both children leaves
        if is_left_leaf and is_right_leaf:
            count +=2
            return count

        #if left not leaf
        if  not is_left_leaf:
            if is_right_leaf:
                count += 1  
            count = self.count_leaves(tree["left"], count)

        #if right not leaf
        if not is_right_leaf:
            if is_left_leaf:
                count += 1
            count = self.count_leaves(tree["right"], count)
           
        return count
    
   ##### END OF REDUCED ERROR PRUNINIG METHOD########
