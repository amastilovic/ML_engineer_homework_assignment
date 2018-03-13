import numpy as np
import pandas as pd
import scipy.stats as stats

import math


def entropy(categoricalVariableAsList):
    '''
    Calculate entropy:
    H(Y) = -sigmaOveri(P(Y=yi)*log2(P(Y=yi)))
    '''

    probabilityDistributionOfY = pd.Series(categoricalVariableAsList).value_counts(normalize=True)
    entropy = -1.0*sum([p*math.log(p, 2) for p in probabilityDistributionOfY])

    return entropy


def conditionalEntropy(YasList, XasList):
    '''
    Calculate conditional entropy:
    H(Y|X) = sigmaOverj(P(X=xj)*H(Y|X=xj))
    '''

    pairs = zip(XasList, YasList)
    probabilityDistributionOfX = pd.Series(XasList).value_counts(normalize=True)
    conditionalEntropy = 0.0

    for xj in np.unique(XasList):
        conditionalEntropy += probabilityDistributionOfX[xj]*entropy([item for item in pairs if item[0]==xj])

    return conditionalEntropy


def informationGain(YasList, XasList):
    '''
    Calculate information gain:
    IG(Y|X) = H(Y) - H(Y|X)
    '''
    return entropy(YasList) - conditionalEntropy(YasList, XasList)


def computeCorrelationUsingInfoGain(dataframe, targetName, featureName, missingValueLabel='missing'):

    # filter away rows with missing target of feature values
    df = dataframe[(dataframe[targetName]!=missingValueLabel)&(dataframe[featureName]!=missingValueLabel)]

    target = df[targetName]
    feature = df[featureName]

    #print entropy(target)
    #print conditionalEntropy(target, feature)
    #print informationGain(target, feature)

    return informationGain(target, feature)

    #let's test it
    #computeCorrelationUsingInfoGain(df, 'target', 'q58')


def deep_average(list_of_similar_things):
    '''
    Given a list of instances of a nested structure of metrics, return a similar structure with the averages across instances filled in
    '''

    first_thing = list_of_similar_things[0]

    if type(first_thing) is int or type(first_thing) is float or type(first_thing) is np.float64:
        average = np.average(list_of_similar_things)
        return average

    if type(first_thing) is list:
        average = []
        for index in range(0,len(first_thing)):
            average.append( deep_average([x[index] for x in list_of_similar_things]) )
        return average

    if type(first_thing) is np.ndarray:
        return None

    if type(first_thing) is dict:
        average = {}
        for key in first_thing.keys():
            average[key] = deep_average([x[key] for x in list_of_similar_things])
        return average

    if type(first_thing) is type(None):
        return None

    raise TypeError("found type "+str(type(first_thing))+": "+str(list_of_similar_things) )


def deep_average_with_stat_errs(list_of_similar_things):
    '''
    Like for deep_average function, but track statistical errors of each quantity as well
    (applies central limit theorem, so only accurate if significant number of bootstraps applied)
    '''
    first_thing = list_of_similar_things[0]
    if type(first_thing) is int or type(first_thing) is float or type(first_thing) is np.float64:
       average = np.average(list_of_similar_things)
       average_err = np.std(list_of_similar_things) / len(list_of_similar_things)
       return average, average_err

    if type(first_thing) is list:
       averages = []
       average_errs = []
       for index in range(0,len(first_thing)):
           this_average, this_average_err = deep_average_with_stat_errs([x[index] for x in list_of_similar_things])
           averages.append(this_average)
           average_errs.append(this_average_err)
       return averages, average_errs

    if type(first_thing) is np.ndarray:
        return None, None

    if type(first_thing) is dict:
       averages = {}
       average_errs = {}
       for key in first_thing.keys():
           this_average, this_average_err = deep_average_with_stat_errs([x[key] for x in list_of_similar_things])
           averages[key] = this_average
           average_errs[key] = this_average_err
       return averages, average_errs

    if type(first_thing) is type(None):
       return None, None

    raise TypeError("found type "+str(type(first_thing))+": "+str(list_of_similar_things) )
