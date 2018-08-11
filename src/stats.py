#!/usr/bin/env python3
""" 
General statistical tools
""" 
def accuracy_precision_recall_f1_f2(trueNegative, falsePositive, falseNegative, truePositive):
    """
    Given the confusion matrix results computes:
    
    - The accuracy score is the fraction of correct predictions.

    - The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

    - The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

    - The F-beta (beta in [1,2]) score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.


    """
    total = trueNegative + falsePositive + falseNegative + truePositive
    accuracy = 1.0*(truePositive + trueNegative)/total
    precision = 1.0*truePositive/(truePositive+falsePositive)
    recall = 1.0*truePositive/(truePositive+falseNegative)
    f1 = 2.0 * truePositive/(2*truePositive + falsePositive+falseNegative)
    f2 = (1+4.0) * precision*recall/(4*precision + recall)
    return accuracy, precision, recall, f1, f2