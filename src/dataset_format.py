#!/usr/bin/python
""" 
 General tools for data formatting
"""

import pandas as pd
import numpy as np
def dict_to_dataframe(data_dict,remove_NaN=True,orient='index'):
    """ 
        input:  
            - data_dict : a dictionnary like data set
            - remove_NaN : replace nan values by 0
            - orient : "index" or "columns", dict keys on rows or columns
        
        output: a DataFrame 
    """

    data_frame = pd.DataFrame.from_dict(data_dict,orient=orient)
    data_frame = data_frame.replace('NaN',np.nan)
    if remove_NaN : data_frame = data_frame.replace(np.nan,0)
    return data_frame
def classLabel_feature_split(data_dict, target ="poi",featureList= 'all'):
    """ 
        input:  
            - data_dict : a dictionnary like data set
            - target (str) : name of feature to guess (class)
            - featureList (str list): features to use by the classifier model 
        
        output: target array (1 dimension) and feature array (2 dimensions) 
    """

    data_frame = dict_to_dataframe(data_dict)
    target_arr = 1.* data_frame[target].values
    if featureList=='all':
        feature_arr = data_frame.values
    else:
        feature_arr = data_frame[featureList].values
    return target_arr, feature_arr