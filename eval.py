##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks:
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################
import numpy as np

class Evaluator(object):
    """ Class to perform evaluation
    """
    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.

        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.

        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        if not class_labels:
            class_labels = np.unique(annotation)

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)


        for i in range(len(annotation)):

            row = np.where(class_labels == annotation[i])
            column = np.where(class_labels == prediction[i])

            if (row[0].size != 0 and column[0].size != 0):
                confusion[row[0], column[0]] += 1

        return confusion


    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """

        num_predictions = np.sum(confusion)
        
        if(num_predictions == 0):
            return 0.0
        
        true_total = 0.0

        for i in range(len(confusion)):
            true_total += confusion[i, i]

        return true_total / num_predictions


    def precision(self, confusion):
        """ Computes the precision.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        (float, np.array)
            tuple with float for average precision and a 1D numpy array with precision
            for each class
        """

        p = np.zeros(len(confusion))

        for i in range(confusion.shape[0]):  

            true_positive_total = confusion[i, i]
            false_positive_total = 0.0
            
            for j in range(len(confusion)): 
                if j != i:
                    false_positive_total += confusion[j, i]

                    
            p[i] = true_positive_total / (true_positive_total + false_positive_total)

        return p, np.mean(p)

    
    def recall(self, confusion):
        """ Computes the recall.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        (float, np.array)
            tuple with float for average recall and a 1D numpy array with recall
            for each class
        """
        r = np.zeros(len(confusion))

        for i in range(len(confusion)):

            total_true_positive = confusion[i, i]
            total_false_negative = 0.0

            for j in range(len(confusion)):
                    if j != i:
                        total_false_negative += confusion[i, j]

            r[i] = total_true_positive / (total_true_positive + total_false_negative)

        return r, np.mean(r)

    
    def f1_score(self, confusion):
        """ Computes the f1-score.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        (float, np.array)
            tuple with float for average f1-score and a 1D numpy array with f1-score
            for each class
        """
        f = np.zeros((len(confusion),))

        r = self.recall(confusion)[0]
        p = self.precision(confusion)[0]

        for i in range(len(confusion)):
            f[i] =  2 * ((p[i] * r[i]) / (p[i] + r[i]))

        return f, np.mean(f)
