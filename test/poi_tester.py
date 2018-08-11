#!/usr/bin/env python3

"""
    This script checks the results given by  Indetification_of_PersonsOfInterest_in_EnronDataset.ipynb
 
    It requires that the classification algorithm, dataset, and features list
    be saved in .pkl files respectively. 
    
    It returns the classifier steps (parameters) and the statistical report (classifier performance)

    
"""
import pickle
import sys
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.append("../src/")
from stats import *
from dataset_format import *
from sklearn.metrics import confusion_matrix


def evaluate_classifier(classifier,data_dict,target="poi",featureList="all",nbFold=1000):
    target_arr, feature_arr = classLabel_feature_split(data_dict, target =target,featureList= featureList)
    cross_valid = StratifiedShuffleSplit(nbFold,test_size=0.1, random_state = 42)
    target_test_total = []
    prediction_target_test_total = []
    trueNegative_tot, falsePositive_tot, falseNegative_tot, truePositive_tot = 0,0,0,0
    print("#=========== pipeline steps==================================")

    print(classifier.steps)
    print("#=================================")
    for train_index,test_index in cross_valid.split(feature_arr,target_arr):
        feature_train,feature_test  = feature_arr[train_index],feature_arr[test_index]
        target_train,target_test   = target_arr[train_index],target_arr[test_index]
        classifier.fit(feature_train, target_train)
        pred_target_test = classifier.predict(feature_test)
        #--------------------------------------------------------------
        # we can save all predictions and tests then apply sklean metrics outside the loop
        # using the two following lines, however, this takes a lot of space lists my became very long
        #target_test_total = target_test_total+ list(target_test) 
        #prediction_target_test_total = prediction_target_test_total+ list(pred_target_test) 
        # ----------------------------------------------------------------------------
        trueNegative, falsePositive, falseNegative, truePositive = confusion_matrix(target_test, pred_target_test).ravel()
        trueNegative_tot += trueNegative
        falsePositive_tot += falsePositive
        falseNegative_tot += falseNegative
        truePositive_tot += truePositive
    accuracy,precision,recall,fbeta_1,fbeta_2 = accuracy_precision_recall_f1_f2(trueNegative_tot,falsePositive_tot,falseNegative_tot,truePositive_tot)
    #=======================
    total = trueNegative_tot + falsePositive_tot + falseNegative_tot + truePositive_tot
 
    stats_str1 = 'Accuracy: {:>0.2f} Precision: {:>0.2f} Recall: {:>0.2f} F1: {:>0.2f} F2: {:>0.2f}'
    print(stats_str1.format(accuracy, precision, recall, fbeta_1, fbeta_2))
    stats_str2 = 'trueNegative: {:>0.1f}% falsePositive: {:>0.1f}% falseNegative: {:>0.1f}% truePositive: {:>0.1f}%'

    print(stats_str2.format(trueNegative_tot*100./total, falsePositive_tot*100./total, falseNegative_tot*100./total, truePositive_tot*100./total))
    print("#==================================================================")

    return [accuracy, precision, recall, fbeta_1, fbeta_2,trueNegative_tot*100./total, falsePositive_tot*100./total, falseNegative_tot*100./total, truePositive_tot*100./total]



def load_classifier_and_data(CLF_PICKLE_FILENAME,DATASET_PICKLE_FILENAME,FEATURE_LIST_FILENAME):
    with open(CLF_PICKLE_FILENAME, "rb") as clf_infile:
        clf = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "rb") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "rb") as featurelist_infile:
        feature_list = pickle.load(featurelist_infile)
    return clf, dataset, feature_list
def main():

    CLF_PICKLE_FILENAME = "../data/my_classifier.pkl"
    DATASET_PICKLE_FILENAME = "../data/my_dataset.pkl"
    FEATURE_LIST_FILENAME = "../data/my_feature_list.pkl"
    clf, dataset, feature_list = load_classifier_and_data(CLF_PICKLE_FILENAME,DATASET_PICKLE_FILENAME,FEATURE_LIST_FILENAME)
    _ = evaluate_classifier(clf,dataset,"poi",feature_list,nbFold=1000)
    
if __name__ == '__main__':
    main()