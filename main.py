import numpy as np


from classification import DecisionTreeClassifier
from eval import Evaluator
from pruning import Pruning
import copy

"""
Function to split data
    Args:
        x (C x N np.array) - array containing the N attributes for C instances
        y (C x 1 np.array) - array containing ground truths for C instances
        k (int) - nr of parts the data is to be split in 
    Output:
        (x, y) (tuple of np.arrays) - x (k x C x N): array containing the N attributes for C instances split                                        into k parts
        y(k x 1 x N): Array containing ground truths for C instances split into k parts
"""
def data_split(x, y, k):

    data = np.array(np.concatenate((x, y), axis=1))
    np.random.shuffle(data)
    data = np.array(np.split(data,k))

    #split and return
    xpart = np.array(data[:,:,:-1],dtype=int)
    ypart = np.array(data[:,:,-1],dtype=str)

    return xpart, ypart

"""
Function to perform cross validation
    Args:
        x (C x N np.array) - array containing the N attributes for C instances
        y (C x 1 np.array) - array containing ground truths for C instances
        k (int) - nr of folds
    Output:
        (accuracy, classifiers) - accuracy (1 x k np.array): array containing the accuracy for each model
                                - classifiers (1 x k np.array): Array containing the DecisionTreeClassifier()
                                trained models
"""
def cross_validation(x, y, k):

    xpart, ypart = data_split(x,y,k)
    accuracy = np.zeros(k)
    classifiers = np.empty(k,dtype=object)

    for i in range(k):

        # split data correctly
        xval = xpart[i]
        yval = ypart[i]
        xtrain = np.delete(xpart,i,0).reshape((k-1)*xval.shape[0],xval.shape[1])
        ytrain = np.delete(ypart,i,0).reshape((k-1)*xval.shape[0],1)


        # train on training slice
        classifiers[i] = DecisionTreeClassifier()
        classifiers[i] = classifiers[i].train(xtrain, ytrain)

        #predict for test class
        predictions = classifiers[i].predict(xval)

        # validate using statistics
        eval = Evaluator()
        confusion = eval.confusion_matrix(predictions, yval)
        accuracy[i] = eval.accuracy(confusion)

    return accuracy, classifiers

"""
Function to let k models vote on a solution
    Args:
        classifiers (1 x k np.array): Array containing the DecisionTreeClassifier()
                                trained models
        x (C x N np.array) - array containing the N attributes for C instances of test set 
    Output:
        results (1 x C np.array) - final predictions for C instances as voted by the k models
"""
def weighted_predict(classifiers,x_test):

    predictions = np.zeros([len(classifiers),len(x_test)],dtype=object)
    result = np.zeros(len(x_test),dtype=object)

    for i in range(len(classifiers)):
        predictions[i,:] = classifiers[i].predict(x_test)


    for i in range(predictions.shape[1]):
        vals, counts = np.unique(predictions[:,i], return_counts = True)
        result[i] = vals[np.argmax(counts)]        

    return result

"""
Function to print validation metrics
    Args:
        predictions (1 x C np.array) - final predictions for C instances as voted by the k models
        y (C x 1 np.array) - array containing ground truths for C instances
    Output:
        None
"""
def print_stats(predictions,y_test):
    
    eval = Evaluator()
    confusion = eval.confusion_matrix(predictions, y_test)

    accuracy = eval.accuracy(confusion)
    precision = eval.precision(confusion)
    recall = eval.recall(confusion)
    f1 = eval.f1_score(confusion)

    print("confusion", confusion)
    print("accuracy", accuracy)
    print("precision", precision)
    print("recall", recall)
    print("f1", f1)
 
    
    return



if __name__ == "__main__":
    
    #QUESTION 1
    print("Question 1")
    print("Loading the data")
    
    filename = "data/train_full.txt"
    classifier = DecisionTreeClassifier()
    x,y = classifier.load_data(filename)
    

    #QUESTION 2
    print("Question 2")
    print("Training the tree with two different methods")

    print("Training the decision tree...")
    classifier = classifier.train(x,y)
    
    print("Loading the test set...")

    filename = "data/test.txt"
    x_test, y_test = classifier.load_data(filename)
    
    print("\nPredicting on test.txt data with 4 different trees")
    
    #Load the evaulator class
    eval = Evaluator()
    prune = Pruning()
    
    print("\nTree 2 unpruned")
    tree_3 = np.load('simple_tree.npy',allow_pickle = True).item()
    predictions = classifier.predict(x_test)
    confusion = eval.confusion_matrix(predictions, y_test)
    accuracy_3 = eval.accuracy(confusion)
    print("number of leaves:",prune.count_leaves(tree_3))
    print("Tree 2 unpruned Accuracy: " + str(np.round(accuracy_3*100,2)))
    
    print("\nTree 2 pruned")
    tree_4 = np.load('simple_tree_pruned.npy',allow_pickle = True).item()
    predictions = classifier.predict(x_test,tree_4)
    confusion = eval.confusion_matrix(predictions, y_test)
    accuracy_4 = eval.accuracy(confusion)
    print("number of leaves:",prune.count_leaves(tree_4))
    print("Tree 2 pruned Accuracy: " + str(np.round(accuracy_4*100,2)))
    
    print("Question 2.3")
    print("Printing the tree")
    classifier.print_tree(tree_3,"Method_2_UnPruned.pdf")
    
    print("\n\n")
    
    #### QUESTION 3 ##########
    print("Question 3")
    filename = "data/test.txt"
    classifier = DecisionTreeClassifier()
    x_test,y_test = classifier.load_data(filename)

    
    #Question 3.1
    print("\nQ3.1")

    filenames = ["data/train_full.txt", "data/train_sub.txt","data/train_noisy.txt"]
    for f in filenames:

        print("\ntraining " + f)
        classifier = DecisionTreeClassifier()
        x,y = classifier.load_data(f)
        classifier = classifier.train(x,y)
        predictions = classifier.predict(x_test)
        print_stats(predictions,y_test)

    #Question 3.3
    print("\nQ3.3")
    filename = "data/train_full.txt"
    x,y = classifier.load_data(filename)

    crossval = cross_validation(x,y,10)
    accuracy = crossval[0]
    average_acc = sum(accuracy)/len(accuracy)
    std = np.std(accuracy)

    print("average acc", average_acc)
    print("std", std)
    
    #Question 3.4
    print("\nQ3.4")
    predictions = crossval[1][np.argmax(crossval[0])].predict(x_test)
    print_stats(predictions,y_test)

    #Question 3.5
    print("\nQ3.5")
    predictions = weighted_predict(crossval[1],x_test)
    print_stats(predictions,y_test)
    
 #QUESTION 4 - PRUNING
    
    print("QUESTION 4")
    eval = Evaluator()
    print("Method 1: Reduced Error Pruning\n")
    x,y = classifier.load_data('data/train_noisy.txt')
    classifier = classifier.train(x, y, True, "Tree_Noisy")
    noisy_tree = np.load("Tree_Noisy.npy",allow_pickle = True).item()


    trained_full_left_height = classifier.node_height(tree_3['left']) + 1
    trained_full_right_height = classifier.node_height(tree_3['right']) + 1
    trained_noisy_left_height = classifier.node_height(noisy_tree['left']) + 1
    trained_noisy_right_height = classifier.node_height(noisy_tree['right']) + 1

    print("Train Full Tree height is: " +str(trained_full_left_height) + "and" + str(trained_full_right_height))
    print("Train Noisy Tree height is: " + str(trained_noisy_left_height) +"and" + str(trained_noisy_right_height))

    test_filename = "data/test.txt"
    val_filename = "data/validation.txt"
    x_val, y_val = classifier.load_data(val_filename)
    x_test,y_test = classifier.load_data(test_filename)

    predictions_noisy = classifier.predict(x_test,noisy_tree)
    confusion_noisy = eval.confusion_matrix(predictions_noisy, y_test)
    accuracy_noisy = eval.accuracy(confusion_noisy)
    print("\noriginal accuracy noisy:" + str(accuracy_noisy))
    
    
    ## Trained on Full Data
    print("Pruning the Tree trained on Train Full")
    reduced_error_tree_1 = copy.deepcopy(tree_3)
    new_tree = prune.prune_tree_reduced_error(reduced_error_tree_1, x_val, y_val)
   
    print("number of leaves before:"+ str(prune.count_leaves(reduced_error_tree_1)))
    print("number of leaves after:"+str(prune.count_leaves(new_tree)))

    full_pruned_left_height = classifier.node_height(new_tree['left']) + 1
    full_pruned_right_height = classifier.node_height(new_tree['right']) + 1
    print("Pruned full tree depth is: " + str(full_pruned_left_height ) + "and" + str(full_pruned_right_height))

    predictions_new = classifier.predict(x_test,new_tree)
    confusion_new = eval.confusion_matrix(predictions_new, y_test)
    accuracy_new = eval.accuracy(confusion_new)
    print("\nOld accuracy on test set:" + str(accuracy_3))
    print("New accuracy on test set:" + str(accuracy_new))


    
    ## Trained on Noisy data
    
    print("\nPruning the Tree trained on the noisy data")
    reduced_error_tree_2 = copy.deepcopy(noisy_tree)
    new_tree = prune.prune_tree_reduced_error(reduced_error_tree_2, x_val, y_val)
    print("\nnumber of leaves before:"+ str(prune.count_leaves(reduced_error_tree_2)))
    print("number of leaves after:"+str(prune.count_leaves(new_tree)))

    noisy_pruned_left_height = classifier.node_height(new_tree['left']) + 1
    noisy_pruned_right_height = classifier.node_height(new_tree['right']) + 1
    print("Pruned Noisy tree depth is: " + str(noisy_pruned_left_height) + "and" + str(noisy_pruned_right_height))


    predictions_new = classifier.predict(x_test,new_tree)
    confusion_new = eval.confusion_matrix(predictions_new, y_test)
    accuracy_new = eval.accuracy(confusion_new)
    print("\nOld accuracy on test set:" + str(accuracy_noisy))
    print("New accuracy on test set:" + str(accuracy_new))
    
    
    ##POST CHI^2 PRUNING METHOD#####
    
    print("\n")
    print("\nMethod 2: Post CHI^2 Pruning\n")
    ## METHOD 2's UNPREPRUNED TREE 
    print("Pruning the Tree trained on Train Full")
    chi_1_tree = prune.post_chi_pruning(tree_3)
    predictions_new = classifier.predict(x_test,chi_1_tree)
    confusion_new = eval.confusion_matrix(predictions_new, y_test)
    accuracy_new = eval.accuracy(confusion_new)
    print("\nOld accuracy on test set:" + str(accuracy_3))
    print("New accuracy on test set:" + str(accuracy_new))

    predictions_new = classifier.predict(x_val,chi_1_tree)
    confusion_new = eval.confusion_matrix(predictions_new, y_val)
    accuracy_new = eval.accuracy(confusion_new)
    print("New accuracy on validation set:" + str(accuracy_new))

    print("number of leaves before:"+ str(prune.count_leaves(tree_3)))
    print("number of leaves after:"+str(prune.count_leaves(chi_1_tree)))

    full_pruned_left_height = classifier.node_height(chi_1_tree['left']) + 1
    full_pruned_right_height = classifier.node_height(chi_1_tree['right']) + 1
    print("Pruned full tree depth is: " + str(full_pruned_left_height) + "and" + str(full_pruned_right_height))
    
    ## Tree trained on Noisy data
    print("\nPruning the Tree trained on the noisy data\n")
    chi_2_tree = prune.post_chi_pruning(noisy_tree)
    predictions_new = classifier.predict(x_test,chi_2_tree)
    confusion_new = eval.confusion_matrix(predictions_new, y_test)
    accuracy_new = eval.accuracy(confusion_new)
    print("\nOld accuracy on test set:" + str(accuracy_noisy))
    print("New accuracy on test set:" + str(accuracy_new))

    print("number of leaves before:"+ str(prune.count_leaves(noisy_tree)))
    print("number of leaves after:"+str(prune.count_leaves(chi_2_tree)))

    noisy_pruned_left_height = classifier.node_height(chi_2_tree['left']) + 1
    noisy_pruned_right_height = classifier.node_height(chi_2_tree['right']) + 1
    print("Pruned Noisy tree depth is: " + str(noisy_pruned_left_height) + "and" + str(noisy_pruned_right_height))

    predictions_new = classifier.predict(x_val,chi_2_tree)
    confusion_new = eval.confusion_matrix(predictions_new, y_val)
    accuracy_new = eval.accuracy(confusion_new)
    print("New accuracy on validation set:" + str(accuracy_new))

    ##COST COMPLEXITY#####
    print("\nCOMPLEXITY PRUNING\n")
    cost_complexity_trees_1 = prune.cost_complexity_pruning(tree_3)
    best_tree_full,alpha = prune.calculate_best_pruned_tree(tree_3,cost_complexity_trees_1, x_val, y_val)
    predictions_new = classifier.predict(x_test, best_tree_full)
    confusion_new = eval.confusion_matrix(predictions_new, y_test)
    accuracy_new = eval.accuracy(confusion_new)
    print("\nOld accuracy on test set:" + str(accuracy_3))
    print("New accuracy on test set:" + str(accuracy_new))

    print("number of leaves before:"+ str(prune.count_leaves(tree_3)))
    print("number of leaves after:"+str(prune.count_leaves(best_tree_full)))

    full_pruned_left_height = classifier.node_height(best_tree_full['left']) + 1
    full_pruned_right_height = classifier.node_height(best_tree_full['right']) + 1
    print("Pruned full tree depth is: " + str(full_pruned_left_height) + "and" + str(full_pruned_right_height))
    print("alpha"+str(alpha))

    cost_complexity_trees_2 = prune.cost_complexity_pruning(noisy_tree)
    best_tree_noisy,alpha = prune.calculate_best_pruned_tree(noisy_tree,cost_complexity_trees_2, x_val, y_val)
    predictions_new = classifier.predict(x_test,best_tree_noisy)
    confusion_new = eval.confusion_matrix(predictions_new, y_test)
    accuracy_new = eval.accuracy(confusion_new)
    print("\nOld accuracy on test set:" + str(accuracy_noisy))
    print("New accuracy on test set:" + str(accuracy_new))

    print("number of leaves before:"+ str(prune.count_leaves(noisy_tree)))
    print("number of leaves after:"+str(prune.count_leaves(best_tree_noisy)))

    noisy_pruned_left_height = classifier.node_height(best_tree_noisy['left']) + 1
    noisy_pruned_right_height = classifier.node_height(best_tree_noisy['right']) + 1
    print("Pruned full tree depth is: " + str(noisy_pruned_left_height) + "and" + str(noisy_pruned_right_height))
    print("alpha"+str(alpha))
    
    
    
    
    
    

    
    




