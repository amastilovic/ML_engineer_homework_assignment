import math
import random
import marshal, pickle, types
import copy as cp
import collections
import itertools
import logging

from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn import cross_validation
from sklearn import feature_extraction
from sklearn import linear_model

from sklearn_0p18_bridge_code import *

from cognoa_ds_lib.constants import *
from cognoa_ds_lib.ds_helper_inject import *
from cognoa_ds_lib.ds_helper_stats import *
from cognoa_ds_lib.io.ds_helper_io import *

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.stats as stats


# set up logging
DEFAULT_LOG_LEVEL = logging.INFO # or logging.DEBUG or logging.WARNING or logging.ERROR

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.setLevel(DEFAULT_LOG_LEVEL)


def normalize_range(column):
    '''
    Handy function that operates on a DF column and maps the data range to [0-1]
    '''
    min_val = min(column.values)
    max_val = max(column.values)
    output = (column.values - min_val) / (max_val - min_val)
    return output


def columns_about(df, keyword):
    '''
    Handy function to return list of DataFrame columns that have a keyword (like ADIR or ADOS) somewhere in their title
    '''
    return [x for x in list(df.columns) if keyword.lower() in x.lower()]


def replace_values_in_dataframe_columns(df, columns, values, replacement, replace_if_equal=True):
    '''
    Handy function to replace certain values in certain columns of a dataframe. Useful for feature value mapping before training
    '''
    for column in columns:
        if (replace_if_equal):
            mask = df[column].isin(values)
        else:
            mask = np.logical_not( df[column].isin(values) )

        df[column][mask] = replacement


def subsample_per_class(df, class_column_name, dict_ratio_per_class):
    '''
    Handy function to subsample dataframe by choosing x% of the samples of each 'class' as defined by a column in the df
    '''
    output_df = pd.DataFrame()
    for class_name in dict_ratio_per_class.keys():
        df_this_class = df[df[class_column_name]==class_name]
        ratio = dict_ratio_per_class[class_name]
        total = len(df_this_class)
        sample_size = int(float(total) * ratio)
        subset = df_this_class.loc[np.random.choice(df_this_class.index, sample_size, replace=False)]
        subset = subset.reset_index()
        output_df = pd.concat([output_df, subset])
    return output_df.reset_index(drop=True)


def balance_dataset_on_dimensions(dataset, dimensions, enforce_group_weight_instructions=None, verbose=False):
    ''' If only dataset and dimensions are passed, this function will enforce equal weighting
    for the sum of all rows in all combinations of dimensions.
	....... Note: many new functions are added lower in this file for purposes of constructing the enforce_group_weight_instructions
	....... or the inputs that are needed to determine it,
	....... such as get_desired_condition_fracs, get_general_diagnoses_from_specific_diagnoses, get_combined_column_from_one_hot_columns

    If enforce_group_weight_instructions is defined, then this function will first use dimensions to make a set of equal weights,
    then make a new set of weights using enforce_group_weight_instructions, and will multiply the two sets of weights together

    The specification of the format of enforce_group_weight_instructions is:
        enforce_group_weight_instructions = {
            'balance_across_this_key': balance_column_name,
            'bin_by': bin_by_column_name,
            'desired_tot_weights_df: desired_tot_weights_df
        }
        ... Where balance_column_name is a column of dataset that tells you what column needs to be re-balanced (example: 'condition')
        ... Where bin_by_column_name tells you the bins across which this re-balancing should be done (example: 'age_category'). Need not match
            any of the dimensions of the initial balancing.
        ... Where desired_fracs_df is a dataframe that provides the desired output fractional weights, with rows representing the bins
            and columns representing the instructions for each possible value of the balance_column_name in that bin.

    Here is an example of enforce_group_weight_instructions:
        enforce_group_weight_instructions = {
            'balance_across_this_key': 'condition',
            'bin_by': 'age_category',
            'desired_tot_weights_df': desired_tot_weights_df
        }
        desired_tot_weights_df = pd.DataFrame({
            'age_category': ['3-', '4+'],
            'autism': [0.5, 0.5],
            'ADHD': [0.1, 0.3],
            'neurotypical': [0.4, 0.2]
        })

    '''

    if 'sample_weights' in dataset.columns:
        dataset = dataset.drop('sample_weights', 1)

    if 'pre_scaled_sample_weights' in dataset.columns:
        dataset = dataset.drop('pre_scaled_sample_weights', 1)

    counts = {}
    weights = {}
    total_count = 0
    for  index, row in dataset.iterrows():
        key = tuple([row[dimension] for dimension in dimensions])
        counts[key] = counts[key]+1 if key in counts else 1
        total_count += 1

    for key,count in counts.iteritems():
        weight = float(total_count) / float(count)
        weights[key] = weight

    for group in counts:
        logger.info("%s: %s out of %s -> weight %s", (str(group), str(counts[group]), str(total_count), str(weights[group])))

    sample_weight_dict = {}
    for  index, row in dataset.iterrows():
        sample_weight =  weights[tuple([row[dimension] for dimension in dimensions])]
        sample_weight_dict[index] = sample_weight

    dataset['sample_weights'] = pd.Series(sample_weight_dict)

    ## The rest of this function optionally enforces a specific requested normalization of sample weights,
    ## According to enforce_group_weight_instructions

    def weight_scaling_function(row, enforce_group_weight_instructions, weight_scaling_factors_by_bin):
        '''
        Helper function for balance_dataset_on_dimensions, which determines scaling factors needed
        to achieve enforce_group_weight_instructions
        '''
        balance_across_this_key = enforce_group_weight_instructions['balance_across_this_key']
        bin_by = enforce_group_weight_instructions['bin_by']
        current_weight = row['sample_weights']
        if row[bin_by] not in weight_scaling_factors_by_bin.keys():
            raise ValueError('Error, weights for grouping '+row[bin_by]+' not understood')
        scaling_factors = weight_scaling_factors_by_bin[row[bin_by]]
        this_group = row[balance_across_this_key]
        scaling_factor = scaling_factors[this_group]
        scaled_weight = current_weight*scaling_factor

        return scaled_weight

    if enforce_group_weight_instructions is not None:
        balance_across_this_key = enforce_group_weight_instructions['balance_across_this_key']
        allowed_groups_to_balance = np.unique(dataset[balance_across_this_key].values)
        # balance_across_this_key might be a column that specifies non-overlapping conditions,
        # or any other grouping column else you want to fix weights on
        bin_by = enforce_group_weight_instructions['bin_by']
        # An example of bin_by would be age_category. Will enforce weighting according to
        # instructions in each bin.
        unique_bins = np.unique(dataset[bin_by].values)
        dataset['pre_scaled_sample_weights'] = cp.deepcopy(dataset['sample_weights'].values)
        weight_scaling_factors_by_bin = {}
        # Now loop over bins and determine needed scaling factors
        for this_bin in unique_bins:
            this_bin_dataset = dataset[dataset[bin_by]==this_bin]
            desired_fracs_df = enforce_group_weight_instructions['desired_tot_weights_df']
            # desired_bin_fracs are the weights we want to enforce in this bin
            desired_bin_fracs = (desired_fracs_df[desired_fracs_df[bin_by]==this_bin][allowed_groups_to_balance].reset_index()).iloc[0]
            starting_weights = (this_bin_dataset.groupby(balance_across_this_key).sum())['pre_scaled_sample_weights']
            sum_starting_weights = np.sum(starting_weights.values)
            starting_fractions = starting_weights / sum_starting_weights
            scaling_factors = desired_bin_fracs / starting_fractions
            weight_scaling_factors_by_bin[this_bin] = scaling_factors

        # Now that we know the scaling factors, apply them
        dataset['sample_weights'] = dataset.apply(
                    weight_scaling_function, args=(enforce_group_weight_instructions, weight_scaling_factors_by_bin), axis=1)
    # Extract the weights we want to return into a separate series and drop the redundant columns in the dataframe
    weights_to_return = cp.deepcopy(dataset['sample_weights'])
    if 'sample_weights' in dataset.columns:
        dataset = dataset.drop('sample_weights', 1)
    if 'pre_scaled_sample_weights' in dataset.columns:
        dataset = dataset.drop('pre_scaled_sample_weights', 1)

    return weights_to_return


def get_presence_of_behavior_rules_dict():
    presence_rules_dict = {
        'ados1_a1': ['0', '1', '2', '3'],
        'ados1_a3': ['1', '2'],
        'ados1_a7': ['0', '1'],
        'ados1_a8': ['0', '1'],
        'ados1_b9': ['0', '1'],
        'ados1_b10': ['0', '1'],
        'ados1_b12': ['0', '1'],
        'ados1_d1': ['1', '2'],
        'ados1_d2': ['1', '2'],
        'ados1_d4': ['1', '2', '3'],
        'ados2_a3': ['1', '2'],
        'ados2_a5': ['2', '3'],
        'ados2_b1': ['2'],
        'ados2_b2': ['0', '1'],
        'ados2_b6': ['0', '1'],
        'ados2_d1': ['1', '2'],
        'ados2_d2': ['1', '2'],
        'ados2_d4': ['1', '2', '3'],
        'ados2_e3': ['1', '2'],
        'ados1_a2': ['0', '1'],
        'ados1_b1': ['2'],
        'ados1_b2': ['0', '1'] ,
        'ados1_b5': ['0', '1'] ,
        'ados1_c1': ['0', '1', '2'],
        'ados1_c2': ['0', '1'],
        'ados2_a8': ['0', '1'] ,
        'ados2_b3': ['0', '1'] ,
        'ados2_b8': ['0', '1'] ,
        'ados2_b10': ['0', '1']
    }
    return presence_rules_dict


def prepare_data_for_modeling(df, feature_columns, feature_encoding_map, target_column, force_encoded_features=[]):
    '''
    Prepare a  dataset for modeling by preprocessing every feature into the appropriate encoding
    and splitting into two matrices X=features and Y=target.

    force_encoded_features is an optional argument when you want to force agreement with particular encoded feature set
    '''

    ####
    ## motivation for "pseudo mixed ordinal/categorical" features:
    ## values 0 - 4 are considered to be ordered from low to high severity
    ## however other values and missing values could have all kinds of different meanings
    ## in rare cases it is obvious that a value like 8 can be considered "more sever" than
    ## a lower value, but this it is more common that no clear interpretation like this can be made
    ##
    ## Goal: make it easy for decision trees to track the severity of the values (0-4) by making feature numeric
    ## however, other "categorical" values exist. These should be clustered together on one side of the real
    ## numerical values to make it as easy as possible for a decision tree to branch between them and the numerical values.
    ## The most common "categorical" values are those like 7 and 8. Map negative values (values exist as -1, -5, and -8) to a positive value
    ## by adding 20 to the value
    ## non numeric values such as '' are mapped to +50
    ##
    ## These choices are pretty arbitrary, but they keep the categorical variables on one side of the distribution without merging any of them
    ####

    def safe_convert_to_number(x, dtype=float, problem_val=999):
        try:
            if pd.isnull(x): return problem_val
            converted_val = dtype(x)
            return converted_val
        except:
            return problem_val


    def num_cat_XForm(inValue, minVal=-0.0001):
        try:

            outValue = float(inValue) if float(inValue) > -0.0001 else float(inValue)+20
        except:
            # Values like '' cannot be converted to floats and will end up here
            outValue = 50.
        return outValue


    mixed_numeric_categorical_features = [x for x in feature_columns if (x in feature_encoding_map and feature_encoding_map[x]=='mixed_numeric_categorical')]
    mixed_numeric_categorical_X = df[mixed_numeric_categorical_features]
    for column in mixed_numeric_categorical_X.columns:
        mixed_numeric_categorical_X[column].apply(num_cat_XForm)

    #scalar features don't require any enconding
    scalar_encoded_features = [x for x in feature_columns if (x in feature_encoding_map and feature_encoding_map[x]=='scalar')]
    scalar_encoded_X = df[scalar_encoded_features]
    for column in scalar_encoded_X.columns:
        scalar_encoded_X[column] = scalar_encoded_X[column].apply(safe_convert_to_number)

    #one_hot_encoding features are handled using DictVectorizer
    one_hot_encoded_features = [x for x in feature_columns if (x in feature_encoding_map and feature_encoding_map[x]=='one_hot')]
    one_hot_encoded_feature_dataset = df[one_hot_encoded_features]
    vectorizer = feature_extraction.DictVectorizer(sparse = False)
    # this becomes a list of dicts of rows
    one_hot_encoded_feature_dataset_dict = one_hot_encoded_feature_dataset.T.to_dict().values()
    # this becomes a 2D array
    one_hot_encoded_feature_dataset_dict_vectorized = vectorizer.fit_transform( one_hot_encoded_feature_dataset_dict )
    one_hot_encoded_feature_dataset = pd.DataFrame(one_hot_encoded_feature_dataset_dict_vectorized, columns=vectorizer.feature_names_)
    one_hot_encoded_features = vectorizer.feature_names_

    #exclude all features that include the word "missing"
    one_hot_encoded_features = [x for x in one_hot_encoded_features if "missing" not in x]

    #prepare X
    one_hot_encoded_X=one_hot_encoded_feature_dataset[one_hot_encoded_features]

    #discrete encoding features is handled manually
    discrete_encoded_features = [x for x in feature_columns if (x in feature_encoding_map and feature_encoding_map[x]=='discrete')]
    discrete_encoded_X = df[discrete_encoded_features]
    for feature in discrete_encoded_X.columns:
        possible_values = discrete_encoded_X[feature].unique()

        discrete_encoded_X[feature+"=0"] = discrete_encoded_X[feature].apply(lambda x: 1 if x == '0' else 0)

        discrete_encoded_X[feature+"=1+"] = discrete_encoded_X[feature].apply(lambda x: 1 if x in ['1', '2', '3', '4'] else 0)

        if ('2' in possible_values) or ('3' in possible_values) or ('4' in possible_values):
            discrete_encoded_X[feature+"=2+"] = discrete_encoded_X[feature].apply(lambda x: 1 if x in ['2', '3', '4'] else 0)

        if ('3' in possible_values) or ('4' in possible_values):
            discrete_encoded_X[feature+"=3+"] = discrete_encoded_X[feature].apply(lambda x: 1 if x in ['3', '4'] else 0)

        if '4' in possible_values:
            discrete_encoded_X[feature+"=4"] = discrete_encoded_X[feature].apply(lambda x: 1 if x == '4' else 0)

        discrete_encoded_X = discrete_encoded_X.drop(feature, axis=1)
    discrete_encoded_features = [x for x in discrete_encoded_X.columns]

    #presence of behavior features is an experimental ADOS encoding scheme for now
    presence_of_behavior_encoded_features = [x for x in feature_columns if (x in feature_encoding_map and feature_encoding_map[x]=='presence_of_behavior')]
    presence_of_behavior_encoded_X = df[presence_of_behavior_encoded_features]
    presence_of_behavior_rules_dict = get_presence_of_behavior_rules_dict()
    for feature in presence_of_behavior_encoded_X.columns:
        if feature in presence_of_behavior_rules_dict.keys():
            presence_of_behavior_encoded_X[feature+"_behavior_present"] = presence_of_behavior_encoded_X[feature].apply(lambda x: 1 if x in presence_of_behavior_rules_dict[feature] else 0)
        else:
            logger.warning("Warning: presence encoding rules not defined for this feature, so set to zero. Feature=%s" % feature)
            presence_of_behavior_encoded_X[feature+"_behavior_present"] = 0
    	presence_of_behavior_encoded_X = presence_of_behavior_encoded_X.drop(feature, axis=1)
    presence_of_behavior_encoded_features = [x for x in presence_of_behavior_encoded_X.columns]

    #any features not present in the feature encoding map will be added as-is without encoding
    other_features = [x for x in feature_columns if x not in feature_encoding_map]
    other_features_X = df[other_features]

    #stitch all sets of features together into one X
    X =  pd.concat([other_features_X.reset_index(drop=True), mixed_numeric_categorical_X.reset_index(drop=True), scalar_encoded_X.reset_index(drop=True), one_hot_encoded_X.reset_index(drop=True), discrete_encoded_X.reset_index(drop=True), presence_of_behavior_encoded_X.reset_index(drop=True)], axis=1)
    features = other_features + mixed_numeric_categorical_features + scalar_encoded_features + one_hot_encoded_features + discrete_encoded_features + presence_of_behavior_encoded_features

    #y is just the target column
    y=np.asarray(df[target_column], dtype="|S20")

    if force_encoded_features != []:
        missing_features = list(set(force_encoded_features) - set(features))
        extra_features = list(set(features) - set(force_encoded_features))
        for feature in missing_features:
            X[feature] = np.zeros(len(X.index))
        for feature in extra_features:
            X = X.drop(feature, axis=1)
        features = force_encoded_features

    return X,y,features


def training_data_statistical_stability_tests(dataset,
                                              sample_frac_sizes,
                                              feature_columns,
                                              feature_encoding_map,
                                              target_column,
                                              sample_weights,
                                              dunno_range,
                                              model_function,
                                              outcome_classes,
                                              outcome_class_priors,
                                              cross_validate_group_id,
                                              n_folds,
                                              n_duplicate_runs,
                                              do_plotting=False,
                                              plot_title='',
                                              **model_parameters):

    '''
    Run tests to see how statistically limited our sample is (training and X-validation errors)
    '''

    train_auc_vals = []
    Xvalidate_auc_vals = []
    for sample_frac in sample_frac_sizes:
        duplicate_train_auc_vals = []
        duplicate_Xvalidate_auc_vals = []
        use_n_duplicates = n_duplicate_runs
        if sample_frac < 0.04: use_n_duplicates = 4*n_duplicate_runs
        if sample_frac < 0.08: use_n_duplicates = 2*n_duplicate_runs
        for i in range(use_n_duplicates):   # run a number of times and average performances to iron out uncertainties
            try:
                frac_dataset = dataset.sample(frac=sample_frac)
                frac_sample_weights = sample_weights.iloc[frac_dataset.index]
                train_model, train_features, train_y_predicted_without_dunno, train_y_predicted_with_dunno, train_y_predicted_probs =\
                    all_data_model(frac_dataset, feature_columns=feature_columns, feature_encoding_map=feature_encoding_map, target_column=target_column, sample_weight=frac_sample_weights, dunno_range=dunno_range, model_function=model_function, **model_parameters)

                train_metrics = get_classifier_performance_metrics(outcome_classes, outcome_class_priors, frac_dataset[target_column], train_y_predicted_without_dunno, train_y_predicted_with_dunno, train_y_predicted_probs)
                train_auc = train_metrics[KEY_WITHOUT_DUNNO]['auc']
                duplicate_train_auc_vals.append(train_auc)

                Xvalidate_output = cross_validate_model(frac_dataset, sample_weights=frac_sample_weights, feature_columns=feature_columns,
                    feature_encoding_map=feature_encoding_map, target_column=target_column, dunno_range=dunno_range, n_folds=n_folds,
                    outcome_classes=outcome_classes, outcome_class_priors=outcome_class_priors, model_function=model_function, groupid=cross_validate_group_id, **model_parameters)
                Xvalidate_auc = Xvalidate_output['overall_metrics'][KEY_WITHOUT_DUNNO]['auc']
                logger.info("For sample_frac: %.3f, train AUC: %s, Xvalidate_auc: %s", (sample_frac, str(train_auc), str(Xvalidate_auc)))
                duplicate_Xvalidate_auc_vals.append(Xvalidate_auc)
            except:
                logger.error('Bad statistical fluctuation leading to all samples of same target output. Skip this round.')

        logger.info("For %.3f, average train AUC: %s, average XV AUC: %s"), sample_frac, str(np.mean(duplicate_train_auc_vals)), str(np.mean(duplicate_Xvalidate_auc_vals))
        train_auc_vals.append(np.mean(duplicate_train_auc_vals))
        Xvalidate_auc_vals.append(np.mean(duplicate_Xvalidate_auc_vals))

    if do_plotting:
        plt.figure(figsize=(10,8))
        plt.plot(sample_frac_sizes, train_auc_vals, color='red', label='Training')
        plt.plot(sample_frac_sizes, Xvalidate_auc_vals, color='black', label='Cross validation')
        plt.grid(True)
        plt.xlabel('Fraction of the dataset used', fontsize=20)
        plt.ylabel('Average AUC', fontsize=20)
        plt.title(plot_title, fontsize=22)
        plt.legend(loc='lower right', fontsize=24)
        plt.gca().tick_params(axis='x', labelsize=16)
        plt.gca().tick_params(axis='y', labelsize=16)
        plt.show()


def combine_classifier_performance_metrics(values1, values2, numbers_in_sample_1, numbers_in_sample_2, values1_err=None, values2_err=None):
    '''
    Note: this function does not know what the type of metric is. It is the user's responsibility to use this
    function only on valid metrics. This calculation has been verified for recall and precision (so sensitivity
    and specificity are ok). It will not be as accurate for AUC.

    This function could have been written to operate on the output of a single measurement, but it is structured to
    run on arrays of performance values (values1, values2), and arrays of the numbers in each sample (numbs_in_sample1, ...)
    Each element of the arrays would represent the performance of a different measurement


    Note that this combination is for performances at the fixed operating that were used to determine values1 and values2.
    If these thresholds were optimized independently before the combination it may produce a sub-optimal combined result
    compared to the case where the thresholds of the two algorithms were both floated in the optimization with the combination
    in mind.

    Assumes that all inputs are numpy arrays of values. If you only want to do this combination for a single
    operating point then arrays can be of size 1.

    values1 and values2 represent arrays of the metric value (recall, or precision)

    numbers_in_sample_1 and numbers_in_sample_2 represent arrays of the number of children in bucket 1 or bucket 2.

    In the case of autism recall this would be the number of children with autism in sample 1 or 2.

    In the case of autism precision this would be the number of children who the model thinks have autism
       in sample 1 or 2.

    If calculating a "real life" precision the "n's" should be chosen to get the correct proportions
       for the real life hypothesis (any n1 = n2 would be fine for a 50% each real life hypothesis)..

    Derivation (done separately on each index of the arrays):
    ~~~~~~~~~
    # Combining young and old samples: the calculations:
    N_a,c = #  children with autism, correctly diagnosed
    N_a,ic = # children with autism, incorrectly diagnosed
    N_n,c = # children without autism, correctly diagnosed
    N_n,ic = # children without autism, incorrectly diagnosed
    N_a = # children with autism, total = N_a,c + N_a,ic
    N_n = # children without autism, total = N_n,c + N_n,ic
    N_p = # children with a positive diagnosis = N_a,c + N_n,ic
    N_not = # children with a negative diagnosis = N_n,c + N_a,ic
    N_3,x = Same numbers for 3- bin
    N_4,x = Same numbers for 4+ bin, etc ...

    # Definitions in the 3- bin:
    Recall_3,a = N_3,a,c / (N_3,a,c + N_3,a,ic)
    Recall_3,n = N_3,n,c / (N_3,n,c + N_3,n,ic)
    Precision_3,a = N_3,a,c / (N_3,a,c + N_3,n,ic)
    Precision_3,n = N_3,n,c / (N_3,n,c + N_3,a,ic)

    # Now calculate definitions in the combined, 3- and 4+ bin (or comparable for young vs old in video)
    **... the definition of combined recall is:**
    Recall_a = (N_3,a,c + N_4,a,c) / (N_3,a,c + N_3,a,ic + N_4,a,c + N_4,a,ic)

    ... now define the denominator as N_tot,a and separate the terms
    Recall_a = (N_3,a,c / N_a) + (N_4,a,c / N_a)

    ... Now use definition N_a = N_3,a * (N_a / N_3,a), and same with 4
    Recall_a = (N_3,a / N_a) * (N_3,a,c / N_3,a) + (N_4,a / N_a) * (N_4,a,c / N_4,a)
    **Recall_a = (N_3,a / N_a) * Recall_3,a + (N_4,a / N_a) * Recall_4,a**
    ... Meaning Recall_a is the weighted average of the recalls in the 3- and the 4+ bins

    **... Similarly, the definition of the combined precision is:**
    Precision_a = (N_3,a,c + N_4,a,c) / (N_3,a,c + N_3,n,ic + N_4,a,c + N_4,n,ic)

    ... Now substitute in terms of number of total with positive diagnosis and separate terms
    Precision_a = (N_3,a,c / N_p) + (N_4,a,c / N_p)

    ... Now do subsitutions of N_p = N_3,p * (N_p / N_3,p)
    Precision_a = (N_3,p / N_p) * (N_3,a,c / N_3,p) + (N_4,p / N_p) * (N_4,a,c / N_4,p)
    ** Precision_a = (N_3,p / N_p) * Precision_3,a + (N_4,p / N_p) * Precision_4,a **
    '''

    assert len(numbers_in_sample_1+numbers_in_sample_2) == len(numbers_in_sample_1)   # Simple test that n1 and n2 are numpy arrays
    logger.info('values1: ', values1)
    logger.info('values2: ', values2)

    assert len(values1 + values2) == len(values1)
    numbers_in_sample__total = (numbers_in_sample_1 + numbers_in_sample_2).astype(float)
    weights1 = ((numbers_in_sample_1) / numbers_in_sample__total).astype(float)
    weights2 = ((numbers_in_sample_2) / numbers_in_sample__total).astype(float)

    ## weight1 + weight2 = 1.0 by construction, so don't need to divide by weights in average
    weighted_average_metrics = (weights1 * values1) + (weights2 * values2)
    if values1_err is not None and values2_err is not None:
        # ignore errors on weights: probably small compared to errors on values
        weighted_average_metrics_err = np.sqrt(((weights1*values1_err)**2.) + ((weights2*values2_err)**2.))
    else:
        weighted_average_metrics_err = None

    return weighted_average_metrics, weighted_average_metrics_err


def get_classifier_performance_metrics(class_names,
                                       class_priors,
                                       labels,
                                       predictions_without_dunno,
                                       predictions_with_dunno,
                                       prediction_probabilities,
                                       weights=None):
    '''
    Compute metrics on the predictive power of a multi-class classifer and return them in a dictionary
    '''
    # handy function we're going to use a few times in here
    def compute_precision_recall_accuracy(class_names, confusion_matrix):
        precision_per_class = {}
        recall_per_class = {}
        correct = 0
        total = 0
        for class_name in class_names:
    	    class_index = class_names.index(class_name)
    	    correct += confusion_matrix[class_index][class_index]
    	    total += sum(confusion_matrix[class_index])
    	    try:
    	        precision_per_class[class_name] = confusion_matrix[class_index][class_index] / float(sum([line[class_index] for line in confusion_matrix]))
    	    except ZeroDivisionError:
    	        precision_per_class[class_name] = 0.0
    	    try:
    	        recall_per_class[class_name] = confusion_matrix[class_index][class_index] / float(sum(confusion_matrix[class_index]))
    	    except ZeroDivisionError:
    	        recall_per_class[class_name] = 0.0
    	try:
    	    accuracy = float(correct) / float(total)
    	except ZeroDivisionError:
    	    accuracy = 0.0
        return precision_per_class, recall_per_class, accuracy


    def apply_priors_to_confusion_matrix(matrix, priors):
        new_matrix = matrix.copy()
        matrix_total = float(sum(sum(matrix)))
        for i in range(0, len(priors)):
            class_prior = priors[i]
            class_proportion = float(sum(matrix[i]))/matrix_total if matrix_total>0.0 else 0.0
            class_multiplier = (class_prior / class_proportion) if class_proportion!=0 else 1
            new_matrix[i] = np.array([100*value*class_multiplier for value in new_matrix[i]])
        return new_matrix

    # metrics related to dataset profile

    number_samples = len(labels)

    # compute number of samples for every class
    samples_per_class = {}

    for class_name in class_names:
        samples_per_class[class_name] = len([x for x in labels if x==class_name])

    # metrics related to classification excluding dunno class logic

    positive_probabilities = [x[0] for x in prediction_probabilities]
    auc_without_dunno = metrics.roc_auc_score([x==class_names[0] for x in labels], positive_probabilities, sample_weight=weights)
    dataset_confusion_without_dunno = confusion_matrix_0p18(labels, predictions_without_dunno, labels=class_names, sample_weight=weights)
    dataset_precision_per_class_without_dunno, dataset_recall_per_class_without_dunno, dataset_accuracy_without_dunno = compute_precision_recall_accuracy(class_names, dataset_confusion_without_dunno)

    reallife_confusion_without_dunno = apply_priors_to_confusion_matrix(dataset_confusion_without_dunno, class_priors)
    reallife_precision_per_class_without_dunno, reallife_recall_per_class_without_dunno, reallife_accuracy_without_dunno = compute_precision_recall_accuracy(class_names, reallife_confusion_without_dunno)

    # metrics related to classification including dunno class logic

    try:
        dataset_confusion_with_dunno = confusion_matrix_0p18(labels, predictions_with_dunno, class_names+['dunno'], sample_weight=weights)
        dataset_precision_per_class_with_dunno, dataset_recall_per_class_with_dunno, dataset_accuracy_with_dunno = compute_precision_recall_accuracy(class_names, dataset_confusion_with_dunno)

        reallife_confusion_with_dunno = apply_priors_to_confusion_matrix(dataset_confusion_with_dunno, class_priors)
        reallife_precision_per_class_with_dunno, reallife_recall_per_class_with_dunno, reallife_accuracy_with_dunno = compute_precision_recall_accuracy(class_names, reallife_confusion_with_dunno)
    except Exception as msg:   # usually because of so broad a dunno range that confusion matrix is not defined
        logger.debug("Getting classifier performance metrics with dunno classified failed with message" % msg)
        dataset_confusion_with_dunno = None
        dataset_precision_per_class_with_dunno = {class_name: np.nan for class_name in class_names}
        dataset_recall_per_class_with_dunno = {class_name: np.nan for class_name in class_names}
        dataset_accuracy_with_dunno = np.nan
        reallife_confusion_with_dunno = None
        reallife_precision_per_class_with_dunno = {class_name: np.nan for class_name in class_names}
        reallife_recall_per_class_with_dunno = {class_name: np.nan for class_name in class_names}
        reallife_accuracy_with_dunno = {class_name: np.nan for class_name in class_names}
        dataset_classification_rate = 0.
        reallife_classification_rate = 0.

    #create a list of labels, predictions and scores excluding the unclassified cases
    z = zip(labels.tolist(), predictions_with_dunno.tolist(), positive_probabilities)
    labels_where_classified = [x[0] for x in z if x[1]!="dunno"]
    predictions_where_classified = [x[1] for x in z if x[1]!="dunno"]
    probabilities_where_classified = [x[2] for x in z if x[1]!="dunno"]
    if weights is not None:
        z = zip(weights.tolist(), predictions_with_dunno.tolist())
        weights_where_classified = [x[0] for x in z if x[1]!="dunno"]
    else:
        weights_where_classified = None

    #then compute some metrics over those
    #dataset_confusion_where_classified = metrics.confusion_matrix(labels_where_classified, predictions_where_classified, class_names)
    try:
        dataset_confusion_where_classified = confusion_matrix_0p18(labels_where_classified, predictions_where_classified, class_names, sample_weight=weights_where_classified)
        dataset_precision_per_class_where_classified, dataset_recall_per_class_where_classified, dataset_accuracy_where_classified = compute_precision_recall_accuracy(class_names, dataset_confusion_where_classified)
        reallife_confusion_where_classified = apply_priors_to_confusion_matrix(dataset_confusion_where_classified, class_priors)
        reallife_precision_per_class_where_classified, reallife_recall_per_class_where_classified, reallife_accuracy_where_classified = compute_precision_recall_accuracy(class_names, reallife_confusion_where_classified)
        dataset_classification_rate = float(sum(sum(dataset_confusion_where_classified))) / float(sum(sum(dataset_confusion_with_dunno)))
        reallife_classification_rate = float(sum(sum(reallife_confusion_where_classified))) / float(sum(sum(reallife_confusion_with_dunno)))
    except Exception as msg:
        logger.debug("Getting classifier performance metrics where classified failed with message" % msg)
        dataset_confusion_where_classified = None
        dataset_precision_per_class_where_classified = {class_name: np.nan for class_name in class_names}
        dataset_recall_per_class_where_classified = {class_name: np.nan for class_name in class_names}
        dataset_accuracy_where_classified = np.nan
        reallife_confusion_where_classified = None
        reallife_precision_per_class_where_classified = {class_name: np.nan for class_name in class_names}
        reallife_recall_per_class_where_classified = {class_name: np.nan for class_name in class_names}
        reallife_accuracy_where_classified = {class_name: np.nan for class_name in class_names}
        dataset_classification_rate = 0.
        reallife_classification_rate = 0.

    try:
        # May fail when others succeed due to roc_auc_score demand of presence of both classes
        auc_where_classified = metrics.roc_auc_score([label==class_names[0] for label in labels_where_classified], probabilities_where_classified, sample_weight=weights_where_classified)
    except:
        auc_where_classified = np.nan

    output = {
        'number_samples':number_samples,
        'samples_per_class':samples_per_class,
        KEY_WITHOUT_DUNNO: {
            'auc': auc_without_dunno,
            'dataset_accuracy':dataset_accuracy_without_dunno,
            'reallife_accuracy':reallife_accuracy_without_dunno,
            'dataset_precision_per_class':dataset_precision_per_class_without_dunno,
            'reallife_precision_per_class':reallife_precision_per_class_without_dunno,
            'dataset_recall_per_class':dataset_recall_per_class_without_dunno,
            'reallife_recall_per_class':reallife_recall_per_class_without_dunno,
            'dataset_confusion': dataset_confusion_without_dunno,
        },
        KEY_WITH_DUNNO: {
            'auc': auc_where_classified,
            'dataset_classification_rate': dataset_classification_rate,
            'reallife_classification_rate': reallife_classification_rate,
            'dataset_accuracy_where_classified': dataset_accuracy_where_classified,
            'reallife_accuracy_where_classified': reallife_accuracy_where_classified,
            'dataset_precision_per_class':dataset_precision_per_class_with_dunno,
            'reallife_precision_per_class':reallife_precision_per_class_with_dunno,
            'dataset_recall_per_class':dataset_recall_per_class_with_dunno,
            'reallife_recall_per_class':reallife_recall_per_class_with_dunno,
            'dataset_precision_per_class_where_classified':dataset_precision_per_class_where_classified,
            'reallife_precision_per_class_where_classified':reallife_precision_per_class_where_classified,
            'dataset_recall_per_class_where_classified':dataset_recall_per_class_where_classified,
            'reallife_recall_per_class_where_classified':reallife_recall_per_class_where_classified,
            'dataset_confusion': dataset_confusion_with_dunno,
        }
    }

    return output


def get_important_features(model, feature_names, relative_weight_cutoff=0.01):
    '''
    Returns a sorted list of (important feature, importance value) pairs for a passed model
    '''
    sorted_feature_importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    max_feature_importance = sorted_feature_importances[0][1]
    min_feature_importance = relative_weight_cutoff*max_feature_importance
    trimmed_sorted_feature_importances = [x for x in sorted_feature_importances if x[1]>min_feature_importance]
    return trimmed_sorted_feature_importances


def dedup_list(mylist):
    return sorted(set(mylist), key = lambda x : mylist.index(x))


def get_best_features(important_features, number_of_features_to_keep, identifiers_for_suffixes_to_ignore, features_to_skip):
    '''
    Gets best features among features that are not skipped. Encoding may lead to variable names
    like 'feature=3' or 'feature_behavior_present'. identifiers_for_suffixes_to_ignore is a list
    of strings. If any of these strings is contained in a feature name, function will ignore that string
    and any suffix following it.

    Given the output of get_important_features(), returns the n best features to keep,
    ignoring anything after a certain suffix char in the feature name, and excluding certain features
    '''

    candidate_features = [x[0] for x in important_features]

    #trim everything after 'suffix_to_ignore' in every feature
    for string_for_suffix_to_ignore in identifiers_for_suffixes_to_ignore:
        candidate_features = [x.split(string_for_suffix_to_ignore)[0] for x in candidate_features]

    #skip features to skip
    candidate_features = [x for x in candidate_features if x not in features_to_skip]

    #dedup
    candidate_features = dedup_list(candidate_features)

    #return top N candidate features
    return candidate_features[0:number_of_features_to_keep]


def get_modeled_results(model, X, dunno_range):
    '''
    Helper function to extract modeled results for
    all_data_model_withAlternates function below
    '''
    y_predicted_without_dunno = model.predict(X)
    y_predicted_probs = model.predict_proba(X)
    y_predicted_with_dunno = np.array(y_predicted_without_dunno, copy=True)
    if dunno_range:
        for i in range(0,len(y_predicted_with_dunno)):
            if (dunno_range[0] < y_predicted_probs[i][0] < dunno_range[1]):
                y_predicted_with_dunno[i] = "dunno"
    return y_predicted_without_dunno, y_predicted_with_dunno, y_predicted_probs


def all_data_model_withAlternates(dataset, feature_columns, feature_encoding_map, target_column, sample_weight, dunno_range, alternate_models, model_function, **model_parameters ):
    ''' Like all_data_model below, but modified to support the comparison with many alternate models '''

    X,y,features = prepare_data_for_modeling(dataset, feature_columns, feature_encoding_map, target_column)

    model = model_function(**model_parameters)
    model.fit(X=X, y=y, sample_weight=sample_weight.values)#.tolist())

    y_predicted_without_dunno = model.predict(X)
    y_predicted_probs = model.predict_proba(X)

    # These new modeled results here:
    y_predicted_without_dunno, y_predicted_with_dunno, y_predicted_probs = get_modeled_results(model, X, dunno_range)
    model_results = {'y_predicted_without_dunno': y_predicted_without_dunno,
            'y_predicted_with_dunno': y_predicted_with_dunno,
            'y_predicted_probs': y_predicted_probs}

    # Now compare with older modeled results:
    alt_model_results = collections.OrderedDict()
    for alt_model_info in alternate_models:
        alt_model = alt_model_info['model']
        alt_y_pred_without_dunno, alt_y_pred_with_dunno, alt_y_pred_probs = get_modeled_results(alt_model, X, dunno_range)
        alt_model_results[alt_model_info['desc']] = {'y_predicted_without_dunno': alt_y_pred_without_dunno,
            'y_predicted_with_dunno': alt_y_pred_with_dunno,
            'y_predicted_probs': alt_y_pred_probs}

    return model, features, model_results, alt_model_results


def all_data_model(dataset, feature_columns, feature_encoding_map, target_column, sample_weight, dunno_range, model_function, **model_parameters):
    '''
    Trains a model using the entire dataset passed
    '''

    X,y,features = prepare_data_for_modeling(dataset, feature_columns, feature_encoding_map, target_column)

    model = model_function(**model_parameters)
    if (sample_weight is not None):
    	model.fit(X=X, y=y, sample_weight=sample_weight.values)
    else:
    	model.fit(X=X, y=y)

    y_predicted_without_dunno = model.predict(X)
    y_predicted_probs = model.predict_proba(X)

    #replace predicted class with "dunno" class where appropriate
    y_predicted_with_dunno = np.array(y_predicted_without_dunno, copy=True)
    if dunno_range:
        for i in range(0,len(y_predicted_with_dunno)):
            if (dunno_range[0] < y_predicted_probs[i][0] < dunno_range[1]):
                y_predicted_with_dunno[i] = "dunno"

    return model, features, y_predicted_without_dunno, y_predicted_with_dunno, y_predicted_probs


def get_stratified_labeled_KFolds(myDF, n_folds, target_column='diagnosis', groupKey=None):
    '''
    Mixes the functionality of stratified k-fold and labeled k-fold
    Will split into n folds with approximately equal amounts of each dignosis.
    Groups, as defined by rows with common values of groupKey, will be preserved when
    doing this. If groupKey is left as None this functionality is ignored
    and the function reduces to simple StratifiedKFolds.

    Returns: an n-element list of training groups and an n-element list of validate
    groups. These groups are each lists of positions of the rows that should be selected
    in that particular fold
    '''

    #myDF[target_column+'encoding'] = myDF[target_column].apply(lambda x : 1. if x == 'autism' else 0.)
    if groupKey is None:
        # no grouping to do
        avgByGroupDF = myDF
        # This just reduces to the non labeled KFold case
        valsForFolding = myDF[target_column].values
    else:
        # If more than one row is in group, hopefully they all have the same
        # value of the target column. If not then take the most common value
        # (mode) as the one to use for the group.
        # Note: the [0][0] is actually needed to retrieve the mode from
        # the weird object that scipy stats returns
        avgByGroupDF = myDF.groupby(groupKey).agg(lambda x: stats.mode(x)[0][0])
        valsForFolding = avgByGroupDF[target_column].values

    cross_validation_folds = cross_validation.StratifiedKFold(n_folds=n_folds, y=valsForFolding, shuffle=True)
    logger.debug("cross validation folds: %s" % cross_validation_folds)

    trainingGroups, validateGroups = [], []

    # Cross validation folds tell us the numpy row index values (*not the pandas index*)
    # for the training and validation data for each fold
    # of the rows in the underlying 2d numpy array of the avgByGroupDF dataframe
    # .... next need to convert this back to the assocaited numpy index values for the
    # non-grouped dataframe (myDF)
    # If not running with groups then avgByGroupDF is the same as myDF so
    # this will just return the original values
    for train,validate in cross_validation_folds:

        if groupKey is None:   # no grouping, so no need to map back to original DF
            trainingGroups.append(train)
            validateGroups.append(validate)
        else:
            # Need to map backto original DF
            trainingIDs = avgByGroupDF.index.values[train]
            validateIDs = avgByGroupDF.index.values[validate]
            origTrainingPositions = np.where(myDF[groupKey].isin(trainingIDs) == True)[0]
            origValidatePositions = np.where(myDF[groupKey].isin(validateIDs) == True)[0]
            trainingGroups.append(origTrainingPositions)
            validateGroups.append(origValidatePositions)

    return trainingGroups, validateGroups


def cross_validate_model(dataset, sample_weights, feature_columns, feature_encoding_map,
        target_column, dunno_range, n_folds, outcome_classes, outcome_class_priors, model_function,
        groupid=None, validation_weights=None, **model_parameters):
    '''
    sample_weights is for the actual training.

    validation_weights is optional in case you want certain events to carry more importance in the
    cross validation (need not match sample_weights)
    '''

    metrics = {'fold_metrics':[], 'fold_important_features':[], 'overall_metrics':[]}
    df_to_use = cp.deepcopy(dataset)
    if sample_weights is not None:
        df_to_use['sample_weights'] = sample_weights

    X,y,features = prepare_data_for_modeling(dataset, feature_columns, feature_encoding_map, target_column)
    metrics['features'] = features

    # split the dataset into n_folds cross validation folds
    trainPositionsGroups, validatePositionsGroups = get_stratified_labeled_KFolds(df_to_use, n_folds=n_folds, target_column=target_column, groupKey=groupid)

    # these are going to be overall prediction lists across folds
    overall_y_real = []
    overall_y_predicted_without_dunno = []
    overall_y_predicted_with_dunno = []
    overall_y_predicted_probs = []
    overall_validation_weights = None if validation_weights is None else []

    #indices_with_correct_Xvalid_results = np.array([])
    correctly_predicted_sample_indices = np.array([])

    #handle folds one at  at ime
    fold_num = 0
    for trainPositions, validatePositions in zip(trainPositionsGroups, validatePositionsGroups):
        fold_num += 1

        X_train = X.iloc[trainPositions]
        X_validate = X.iloc[validatePositions]
        y_train = pd.Series(y).iloc[trainPositions]
        y_validate = pd.Series(y).iloc[validatePositions]
        sample_weight_train = sample_weights.iloc[trainPositions]
        weights_validate = None if validation_weights is None else validation_weights.iloc[validatePositions]

        model = model_function(**model_parameters)
        if sample_weights is not None:
            model.fit(X=X_train, y=y_train, sample_weight=sample_weight_train.values)
        else:
            model.fit(X=X_train, y=y_train)

        y_predicted_without_dunno = model.predict(X=X_validate)
        y_predicted_probs = model.predict_proba(X=X_validate)

        #replace predicted class with "dunno" class where appropriate
        y_predicted_with_dunno = np.array(y_predicted_without_dunno, copy=True)
        if dunno_range:
            for i in range(0,len(y_predicted_with_dunno)):
                if (dunno_range[0] < y_predicted_probs[i][0] < dunno_range[1]):
                    y_predicted_with_dunno[i] = "dunno"

        #maintain overall prediction lists across folds
        overall_y_real.extend(y_validate.values)
        overall_y_predicted_without_dunno.extend(y_predicted_without_dunno)
        overall_y_predicted_with_dunno.extend(y_predicted_with_dunno)
        overall_y_predicted_probs.extend(y_predicted_probs)
        if weights_validate is not None:
            overall_validation_weights.extend(weights_validate)

        values_match = y_validate == y_predicted_without_dunno
        indices_that_match = values_match[values_match==True].index

        correctly_predicted_sample_indices = np.concatenate([correctly_predicted_sample_indices, indices_that_match])

        #log metrics for this fold
        metrics['fold_metrics'].append( get_classifier_performance_metrics(outcome_classes, outcome_class_priors, y_validate, y_predicted_without_dunno, y_predicted_with_dunno, y_predicted_probs, weights_validate) )
        metrics['fold_important_features'].append ( get_important_features(model, features, 0.01) )


    #log overall metrics across folds
    validation_weights_arr = None if overall_validation_weights is None else np.array(overall_validation_weights)
    metrics['overall_metrics'] = get_classifier_performance_metrics(outcome_classes, outcome_class_priors, np.array(overall_y_real), np.array(overall_y_predicted_without_dunno), np.array(overall_y_predicted_with_dunno), np.array(overall_y_predicted_probs), validation_weights_arr)
    metrics['correctly_predicted_sample_indices'] = correctly_predicted_sample_indices

    return metrics


def cross_validate_model_with_addon(dataset, sample_weights, feature_columns, feature_encoding_map, target_column,  dunno_range, n_folds, outcome_classes, outcome_class_priors, model_function, validation_weights=None, **model_parameters):
    '''
    Cross_validates a model using stratified K-fold
    Uses a special 'addon' group [here hardcoded to 4 yearolds] for training only and not for validation
    '''

    metrics = {'fold_metrics':[], 'fold_important_features':[], 'overall_metrics':[]}

    #prepare the dataset for modeling
    X,y,features = prepare_data_for_modeling(dataset, feature_columns, feature_encoding_map, target_column)

    #split into 'main' vs 'addon_training' parts according to age
    X_main = X[dataset.age_years<=3]
    X_addon_training = X[dataset.age_years==4]
    y_main = y[np.array(dataset.age_years<=3)]
    weights_validate_main = None if validation_weights is None else validation_weights[np.array(dataset.age_years<=3)]
    y_addon_training = y[np.array(dataset.age_years==4)]
    sample_weights_main = sample_weights[dataset.age_years<=3]
    sample_weights_addon_training = sample_weights[dataset.age_years==4]

    #split the main part into n_folds cross validation folds
    cross_validation_folds = cross_validation.StratifiedKFold(n_folds=n_folds, y=y_main, shuffle=True)

    #these are going to be overall prediction lists across folds
    overall_y_real = []
    overall_y_predicted_without_dunno = []
    overall_y_predicted_with_dunno = []
    overall_y_predicted_probs = []
    overall_validation_weights = None if validation_weights is None else []

    #handle folds one at  at ime
    for train,validate in cross_validation_folds:
        #every training fold consists of the training part of the main dataset plus the entire addon set
        X_main_training = X_main.iloc[train]
        X_train = np.append(X_main_training,X_addon_training, axis=0)
        y_main_training = pd.Series(y_main).iloc[train]
        y_train = np.append(y_main_training,y_addon_training)
        sample_weights_main_training = sample_weights_main.iloc[train]
        sample_weights_train = sample_weights_main_training.append(sample_weights_addon_training)

        #every validation fold consists of the validation part of the main dataset
        X_validate = X_main.iloc[validate]
        y_validate = pd.Series(y_main).iloc[validate]
        weights_validate = None if validation_weights is None else weights_validate_main.iloc[validate]
        assert weights_validate is None or len(X_validate.index) == len(weights_validate)
        assert weights_validate is None or len(weights_validate) == len(validate)

        #fit a model for this fold
        model = model_function(**model_parameters)
        model.fit(X=X_train, y=y_train, sample_weight=sample_weights_train.values)

        y_predicted_without_dunno = model.predict(X=X_validate)
        y_predicted_probs = model.predict_proba(X=X_validate)

        #replace predicted class with "dunno" class where appropriate
        y_predicted_with_dunno = np.array(y_predicted_without_dunno, copy=True)
        if dunno_range:
            for i in range(0,len(y_predicted_with_dunno)):
                if (dunno_range[0] < y_predicted_probs[i][0] < dunno_range[1]):
                    y_predicted_with_dunno[i] = "dunno"

        #maintain overall prediction lists across folds
        overall_y_real.extend(y_validate)
        overall_y_predicted_without_dunno.extend(y_predicted_without_dunno)
        overall_y_predicted_with_dunno.extend(y_predicted_with_dunno)
        overall_y_predicted_probs.extend(y_predicted_probs)
        if weights_validate is not None:
            overall_validation_weights.extend(weights_validate)

        #log metrics for this fold
        metrics['fold_metrics'].append( get_classifier_performance_metrics(outcome_classes, outcome_class_priors, y_validate, y_predicted_without_dunno, y_predicted_with_dunno, y_predicted_probs, weights_validate) )
        metrics['fold_important_features'].append ( get_important_features(model, features, 0.01) )

    #log overall metrics across folds
    validation_weights_arr = None if overall_validation_weights is None else np.array(overall_validation_weights)
    metrics['overall_metrics'] = get_classifier_performance_metrics(outcome_classes, outcome_class_priors, np.array(overall_y_real), np.array(overall_y_predicted_without_dunno), np.array(overall_y_predicted_with_dunno), np.array(overall_y_predicted_probs), validation_weights_arr)

    return metrics


def bootstrap(dataset, number_of_tries, sampling_function_per_try, ml_function_per_try, return_errs=False, verbose=False ):
    '''
    Performs general purpose bootstrapping on a dataset applying given function each time and aaverageing reported metrics back to caller
    '''

    metrics_per_try = []
    for try_index in range(0, number_of_tries):
        if verbose:
            logger.info("On try_index %d out of %d" % (try_index, number_of_tries))
        #sample dataset before every try
        dataset_for_this_try = sampling_function_per_try(dataset)

        #run the ml function and collect reported metrics
        metrics_per_try.append( ml_function_per_try(dataset_for_this_try) )

    #average out the metrics in the (nested) metrics structures that we got back from every try
    if return_errs:
        average_metrics, average_metrics_err = deep_average_with_stat_errs(metrics_per_try)
        return average_metrics, average_metrics_err
    else:
        average_metrics = deep_average(metrics_per_try)
        return average_metrics

    return average_metrics


def get_combinations(list_of_lists):
    '''
    Given a bunch of variables, each in a list of possible ranges, returns tuples for every possible combination of variable values
    for example, given [ ['a','b'], [1,2], ['x','y'] ]
    it returns [ ('a',1,'x'), ('a',1,'y'), ('a',2,'x'), ('a',2,'y'), ('b',1,'x') ... ]
    '''
    return list(itertools.product(*list_of_lists))


def grid_search(modeling_function, param_combinations, reporting_function, verbose=False):
    '''
    Performs a grid search over a modeling_function using different param_combinations each time, and collecting the outputs of a reporting_fuction in successive runs
    '''
    final_report = []
    n_total_combs = len(param_combinations)
    n_so_far = 0
    for param_combination in param_combinations:
        n_so_far += 1
        if verbose:
            logger.info("Starting grid search parameter combination" % str(param_combination))
            logger.info("This is combination %d of %d" % (n_so_far, n_total_combs))
        metrics = modeling_function(param_combination)
        report = list(param_combination)
        report.extend( reporting_function(param_combination, metrics) )
        final_report.append(report)
    return final_report


def getPrecisionDegradationFactors(metrics_worse, metrics_better, debug=False):
    ''' helper function to compare how much worse one set of metrics is than another '''
    recalls_worse = metrics_worse['dataset_recall_per_class']
    recalls_better = metrics_better['dataset_recall_per_class']
    precisions_worse = metrics_worse['dataset_precision_per_class']
    precisions_better = metrics_better['dataset_precision_per_class']
    precision_degradation_factors = {'autism_recall': recalls_worse['autism'] / recalls_better['autism'],
                                'autism_precision': precisions_worse['autism'] / precisions_better['autism'],
                                'not_recall': recalls_worse['not'] / recalls_better['not'],
                                'not_precision': precisions_worse['not'] / precisions_better['not']}

    logger.debug("metrics_worse: %s", str(metrics_worse))
    logger.debug("metrics_better: %s", str(metrics_better))
    logger.debug("precision_degradation_factors: %s", str(precision_degradation_factors))

    return precision_degradation_factors


def reverse_one_hot_encoding(dataset, input_columns, fallback_value='Unknown non-ASD', override_column=None):
    '''
    Converts many columns that encode a feature with 1/0 values into a single
    column that reports one of these features as being present (needed for machine learning target).
    In the case where more than one of the input columns has a non zero value the first one
    in the list that is non zero will have priority.

    input_columns is a list of columns '1' or '0' values to combine.
    fallback_value is what should be specified if none of the input columns have a '1'

    if override_column is specified and relevant values are present then those take precedent over anything
    else in this function

    An example of using this function is a situation where you have many variables encoding a condition
    as a 0 or 1 (for example, does child have OCD, does child have ADHD, does child have autism, ...), and you want to
    combine into a single column that summarizes what the most important of those conditions is (if
    any are non-zero), where importance is defined by the order of the columns. It is suggested, however, that
    you make the input columns non-overlapping where possible (for example:
      ['Neurotypical', 'Any Delay not ASD not ADHD', 'ADD/ADHD/OCD not ASD'])
    '''

    def pick_combined_value_for_row(row):
        possible_unknown_values = ['', 'NaN', 'missing']
        combined_value = None
        if override_column is not None:
            if row[override_column] not in possible_unknown_values:
                return row[override_column]
        for possible_combined_value in input_columns:
            if row[possible_combined_value] in possible_unknown_values:
                continue
            if int(row[possible_combined_value])==1:
                combined_value=possible_combined_value
                break
        if combined_value is None:
            combined_value=fallback_value
        return combined_value

    columns_for_combination = input_columns
    if override_column is not None:
        columns_for_combination.append(override_column)

    return dataset[columns_for_combination].apply(pick_combined_value_for_row, axis=1)


def get_general_diagnoses_from_specific_diagnoses(diagnosis_df, diagnosis_categories_mapping):
    '''
    Diagnosis_df is assumed to be one-hot encoded, where each diagnosis is in a different column
	with 0/1 values. This function maps to more general diagnoses which will be created as new
	columns under the one-hot paradigm, using rules defined in diganosis_categories_mapping.
	If no mapping is found child is assumed to be neurotypical

    returns: diagnosis_df, but with additional general category diagnoses filled

	Example of usage (keys are the more general diagnoses which will be created as new columns
	 and values tell you how to interpret the columns that represent specific diagnoses)
    general_diagnosis_categories_mapping = {
        'ASD': {'require_one_of': ['ASD'], 'not': []},
        'ADD/ADHD/OCD': {'require_one_of': ['D1ADHD', 'D1ATTWHY', 'ADD/ADHD', 'OCD', 'ADHD'], 'not': []},
        'Any Delay': {'require_one_of': ['D1SPEDEL', 'D1EXPLAN', 'D1MIXREC', 'D1DELDEV', 'DD', 'ID', 'LD', 'SD', 'Delay'], 'not': []},
        'Unknown non-ASD': {'require_one_of': ['is_non_autism_not_defined'], 'not': ['ASD']}
    }
    general_diagnosis_categories_mapping['ADD/ADHD/OCD not ASD'] = general_diagnosis_categories_mapping['ADD/ADHD/OCD']
    general_diagnosis_categories_mapping['ADD/ADHD/OCD not ASD']['not'] = general_diagnosis_categories_mapping['ASD']['require_one_of']
    general_diagnosis_categories_mapping['Any Delay not ASD not ADHD'] = general_diagnosis_categories_mapping['Any Delay']
    general_diagnosis_categories_mapping['Any Delay not ASD not ADHD']['not'] =\
        general_diagnosis_categories_mapping['ASD']['require_one_of']+\
        general_diagnosis_categories_mapping['ADD/ADHD/OCD']['require_one_of']
    general_diagnoses=general_diagnosis_categories_mapping.keys()+['Neurotypical']


    df = get_general_diagnoses_from_specific_diagnoses(df,
                                    diagnosis_categories_mapping=general_diagnosis_categories_mapping)
    '''

    def check_category_diagnosis(row, cols_to_require, cols_to_forbid):
        # Use this function on sub-diagnosis, and see if any in the
        # Category of diagnosis are present
        required_col_present = False
        for key in cols_to_require:
            if row[key]!='missing' and row[key]!=0 and row[key]!=False and np.isnan(row[key])==False:
                required_col_present = True
        forbidden_col_present = False
        for key in cols_to_forbid:
            if row[key]!='missing' and row[key]!=0 and row[key]!=False and np.isnan(row[key])==False:
                forbidden_col_present = True
        if required_col_present and forbidden_col_present==False:
            return 1
        return 0

    for general_diagnosis_col, sub_col_specs in diagnosis_categories_mapping.iteritems():
        cols_to_require = [col for col in sub_col_specs['require_one_of'] if col in diagnosis_df.columns]
        cols_to_forbid = [col for col in sub_col_specs['not'] if col in diagnosis_df.columns]
        cols_to_consier = cols_to_require+cols_to_forbid
        if len(cols_to_require)==0:
            diagnosis_df[general_diagnosis_col] = [0]*len(diagnosis_df.index)
        else:
            diagnosis_df[general_diagnosis_col] = diagnosis_df[cols_to_consier].apply(check_category_diagnosis, args=(cols_to_require, cols_to_forbid), axis=1)

    diagnosis_categories = diagnosis_categories_mapping.keys()
    diagnosis_df['some_problem_present'] = diagnosis_df[diagnosis_categories].apply(check_category_diagnosis, args=(diagnosis_categories, []), axis=1)
    swap_zeros_and_ones = lambda x: 1 if x == 0 else 0
    diagnosis_df['Neurotypical'] = diagnosis_df['some_problem_present'].apply(swap_zeros_and_ones)

    return diagnosis_df


def get_value_fractions_of_total_in_column(dataset, input_column, values_to_consider, bin_by_column=None):
    '''
    Given an input column (first used with diagnoses), a list of which values to consider in that column,
    and an optional column by which to bin (for example, age_categroy), determine the total fraction of each value in each bin
    '''
    if bin_by_column is None:
        unique_bins = ['all_one_bin']
    else:
        unique_bins = np.unique(dataset[bin_by_column].values)

    list_of_rows = []
    for this_bin in unique_bins:
        if this_bin == 'all_one_bin':
            this_bin_df = cp.deepcopy(dataset)
        else:
            this_bin_df = dataset[dataset[bin_by_column]==this_bin]
        this_row = {}
        for condition in values_to_consider:
            frac_with_condition = float(len([ele for ele in this_bin_df[input_column].values if ele==condition])) /\
                     float(len(this_bin_df.index))
            this_row[condition] = frac_with_condition
        this_row[bin_by_column] = this_bin
        list_of_rows.append(this_row)
    condition_fracs_df = pd.DataFrame(list_of_rows)
    return condition_fracs_df


def get_desired_condition_fracs(app_frac_df,
                                training_set_df,
                                non_target_cols,
                                unknown_non_target_col,
                                assumed_unknown_non_target_fracs,
                                target_frac=0.5,
                                reg_param=0.5,
                                target_key='ASD',
                                debug_level=0):
    '''
    This function is needed to determine the various non-ASD condition breakdowns for purposes of
    balancing them before training

    Function to determine the desired fractions (or weights) of each type of diagnosis,
    given a desired population that we want to configure our algorithms for (specified by app_frac_df).

    The function takes into account the starting fractions of the conditions (specified in training_set_df) and a desired weighting for
    the target_condition (probably want 0.5 to fit this versus everything else). The function also allows for a sample of 'unknown'
    condition in the training set. When this is present it needs a guess for the breakdown of conditions within the unknown population,
    specified by assumed_unknown_non_target_fracs

    The function will do a ridge regression with regularization parameter=reg_param to pick a good set of target weights,
    apply some constraints, and then return a set of target weights that respect target_frac,
    and attempt to bring the fractions as close as possible to the desired app_frac_df without making any condition
    have weights that are too far out of line with their starting values (maximum weight increase of 5x by default)

    All dataframes are assumed to be binned by age of the child. Rows are the child age, columns are different conditions,
    and the values are the fractions of children of that age with a given condition

    Parameters
    -------
    app_frac_df: pandas.DataFrame
        desired fractions of children with each condition, binned by child age (rows).
        Conditions are in the columns and entries are the fractions.

    training_set_df: pandas.DataFrame
        starting fractions of the conditions in the training set. Organized as app_frac_df

    non_target_cols: list of strings
        a list that specifies which columns of the training_set_df refer to conditions
        that are spectators to the target condition and which must have their fractions determined

    app_frac_df: pandas.DataFrame
        desired fractions of children with each condition, binned by child age (rows).
        Conditions are in the columns and entries are the fractions.

    out_filename: string
        purpose of this function is to create this pickled file

    unknown_non_target_col: optional
        If your training data contains a children of "unknown" conditions (must be sure they do not have target condition),
        specify that column here.

    assumed_unknown_non_target_fracs: pandas.DataFrame
        For your children with unknown (but not target) condition, this is your guess of their condition breakdown, organized as a dataframe that is structured like app_frac_df

    target_frac: float
        a fraction to represent what the weighting of the target condition should be (probably want 50% here?)

    reg_param: float
        when running ridge regression, how strong should the regularization be.
        The larger, the less the training data fractions can be shifted.

    target_key: string
        which condition will be considered the target (default is autism)

    debug_level: boolean
        the larger this number, the more verbose the printing in this function
    '''

    def get_X_y_vals_for_diagnosis_frac_fit(app_frac_vals, unknown_frac_vals, initial_training_fracs, target_frac):
        ''' This reformats inputs into the matrices that need to be
        passed to the ridge regression, with proper sklearn format '''
        ########
        # Example X-matrix for 3 non-autism categories should look like this:
        #
        #    1.    0.    0.    f1_unknown
        #    0.    1.    0.    f2_unknown
        #    0.    0.    1.    f3_unknown
        #    1.    1.    1.       1.
        ### The fX_unknowns are fractions of the total non-autistic sample in the desired output
        ### (total including autism after autism frac is set to target_frac)
        ##
        ## initial_training_fracs are used as offsets to make it so that regularization will constrain to change
        ## results as little as possible from default fractions
        # Example y_vals looks like this:
        #    f1_App - f1_train_initial, f2_App-f2_train_initial, f3_App-f3_train_initial, (1-f_autism-sum(f trains))   (but vertical rather than horizontal)
        ########

        # represents the constraint that all non-target + target adds to a total fraction of 1.0
        y_vals = list(app_frac_vals-initial_training_fracs) + [1. - target_frac - np.sum(initial_training_fracs)]
        assert abs(np.sum(y_vals) + (2.*np.sum(initial_training_fracs))  - 1.) < 0.00001
        y_arr = np.array(y_vals).reshape(len(app_frac_vals)+1, 1)

        my_X_list = []
        for irow, row in enumerate(range(len(app_frac_vals))):
            build_this_row = [0.]*(irow) + [1.] + [0.]*(len(app_frac_vals)-(irow+1))
            # Also append the unknown part for this dimension
            build_this_row.append(unknown_frac_vals[irow])
            my_X_list.append(build_this_row)
        my_X_list.append([1.]*(len(app_frac_vals)+1))
        X_arr = np.array(my_X_list)

        return X_arr, y_arr

    def get_tot_frac_non_target(in_df, non_target_cols):
        '''
        Helper function assumes that in_df has non_target_cols filled with
        values that represent fractions of total that include target
        '''
        non_target_df = cp.deepcopy(in_df)
        non_target_df = non_target_df.drop(target_key, 1)
        tot_frac_non_target = cp.deepcopy(in_df[non_target_cols[0]])
        for col in non_target_cols[1:]:
            tot_frac_non_target += in_df[col]

        return tot_frac_non_target

    def enforce_constraints(fracs_dict, target_frac, initial_training_non_target_fracs_dict, max_allowed_weighting=5.):
        # first get rid of negatives:
        for key in fracs_dict.keys():
            if fracs_dict[key] < 0.: fracs_dict[key] = 0.
        # Now enforce maximum weighting
        for key in fracs_dict.keys():
            if key==target_key: continue
            if fracs_dict[key] > max_allowed_weighting*initial_training_non_target_fracs_dict[key]:
                fracs_dict[key] = max_allowed_weighting*initial_training_non_target_fracs_dict[key]
        # Now normalize so that total is 1.0
        expected_non_target_total = 1. - target_frac
        actual_non_target_total = np.sum([value for key, value in fracs_dict.iteritems() if key != target_key])
        correction_factor = expected_non_target_total / actual_non_target_total
        for key in fracs_dict.keys():
            if key == target_key: continue
            fracs_dict[key] *= correction_factor

        return fracs_dict

    age_groups = app_frac_df['age_category'].values
    desired_condition_fracs = {condition: [] for condition in non_target_cols+[unknown_non_target_col]}
    desired_condition_fracs[target_key] = []
    desired_condition_fracs['age_category'] = []
    for age_group in age_groups:
        # Do linear regression of N+1 equations with N+1 unknowns where the N corresponds to the number of non-autism conditions
        # And the +1 represents the unknown category
        app_non_target_app_frac_vals = np.array([app_frac_df[app_frac_df['age_category']==age_group][non_target_condition].values[0] for non_target_condition in non_target_cols])
        desired_non_target_fraction = 1. - target_frac
        desired_non_target_frac_vals = app_non_target_app_frac_vals * (desired_non_target_fraction / np.sum(app_non_target_app_frac_vals))

        if len(training_set_df[training_set_df['age_category']==age_group].index)==0:
            logger.info("No data available for %d" % age_group)
            continue

        training_non_target_frac_vals = np.array([training_set_df[training_set_df['age_category']==age_group][non_target_condition].values[0] for non_target_condition in non_target_cols])
        unknown_frac_vals = [assumed_unknown_non_target_fracs[assumed_unknown_non_target_fracs['age_category']==age_group][non_target_condition].values[0] for non_target_condition in non_target_cols]
        X_arr, y_arr = get_X_y_vals_for_diagnosis_frac_fit(desired_non_target_frac_vals, unknown_frac_vals, initial_training_fracs=training_non_target_frac_vals, target_frac=target_frac)

        # Now do ridge regression
        if debug_level>=2:
            logger.debug("age_group: %s, do fit on: " % str(age_group))
            logger.debug("X: %s" % str(X_arr))
            logger.debug("y: %s" % str(y_arr))

        reg = linear_model.Ridge(alpha=reg_param, fit_intercept=False)
        fit_output = reg.fit(X_arr, y_arr)

        if debug_level>=2:
            logger.debug("fit intercept: %s" % str(fit_output.intercept_))
            logger.debug("fit parameters: %s" % str(fit_output.coef_))
            logger.debug("initial training offsets were: %s" % str(training_non_target_frac_vals))
            logger.debug("(Need to add those back in to get real app values)")

        # Now undo the centering of the fit parameters around the initial values
        output_fracs_dict = {}
        for idx, (non_target_condition, frac) in enumerate(zip(non_target_cols+[unknown_non_target_col], fit_output.coef_[0])):
            if non_target_condition != unknown_non_target_col: frac += training_non_target_frac_vals[idx]
            output_fracs_dict[non_target_condition] = frac
        output_fracs_dict[target_key] = target_frac

        # Now enforce reasonable constraints on output weighting
        if debug_level>=2:
            logger.debug("Raw output_fracs before enforcement of constraints: " % str(output_fracs_dict))

        initial_training_non_target_fracs_dict = {condition: frac for condition, frac in zip(non_target_cols, training_non_target_frac_vals)}
        initial_training_non_target_fracs_dict[unknown_non_target_col] = training_set_df[training_set_df['age_category']==age_group][unknown_non_target_col].values[0]
        output_fracs_dict = enforce_constraints(output_fracs_dict, target_frac, initial_training_non_target_fracs_dict)

        if debug_level>=2:
            logger.debug("output_fracs_ after enforcement of constraints: %s" % str(output_fracs_dict))

        assert abs(np.sum(output_fracs_dict.values()) - 1.) < 0.00001
        for condition, frac in output_fracs_dict.iteritems():
            desired_condition_fracs[condition].append(output_fracs_dict[condition])
        desired_condition_fracs['age_category'].append(age_group)

    desired_fracs_df = pd.DataFrame(desired_condition_fracs)

    if debug_level>=1:
        debug_cols = ['age_category', target_key]+non_target_cols
        logger.debug("Show results for: ")
        logger.debug("target column: %d" % target_key)
        logger.debug("input training fracs: %s" % str(training_set_df[debug_cols+[unknown_non_target_col]]))
        logger.debug("desired App Fracs: %s" % str(app_frac_df[debug_cols]))
        logger.debug("assumed unknown composition: %s" % str(assumed_unknown_non_target_fracs[["age_category"]+non_target_cols]))
        logger.debug("And the results are: %s" % str(desired_fracs_df[debug_cols+[unknown_non_target_col]]))

    return desired_fracs_df
