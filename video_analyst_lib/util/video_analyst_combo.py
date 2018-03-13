'''
API to generate analyst rating combination functions
'''

import numpy as np
import pandas as pd


THE_OTHER_ANALYSTS = 'the_other_analysts'


def evaluate_analyst_reliabilities_and_create_combination_function(in_df,
                                                                   out_filename,
                                                                   reliability_calculation_to_do,
                                                                   questions_to_include,
                                                                   reliability_combination_function=np.mean,
                                                                   reference_analyst=THE_OTHER_ANALYSTS,
                                                                   combination_function=None,
                                                                   use_combination_for_reliability_analysis=True,
                                                                   subject_id_column_name='Clinical Study Id',
                                                                   analyst_id_column_name='Analyst Id',
                                                                   exclude_these_analysts=[],
                                                                   other_groupby_keys=[],
                                                                   prior_analyst_weights_hypothesis_dict=None,
                                                                   how_serialized='pickle',
                                                                   debug=False):
    '''
    Runs full reliability calculation, and outputs pickled combination function.

    Important note on missing values:
    Real missing data should be entered as either np.nan, 'nan', or ''.
    If they are left as some other value such as 0 or 8 then module
    will consider this to be a real category to compare for accuracy.

    For example, if you have do a combined analysis across module 1 and module 2
    versions, and each subject has only been evaluated with one or the other,
    then make sure responses to the non-evaluated questions are one of these three values.


    Parameters
    -------
    in_df: pandas.DataFrame
        an analysis dataframe in a format with one row per combination of subject and analyst,
        and columns representing different question responses. If encoding is desired it must
        be done before calling this function.

    out_filename: string
        purpose of this function is to create this pickled file

    reliability_calculation_to_do: string
        Run this reliability calculation on each question for each analyst
        ==> To get one reliability metric per analzer. Options:
            ** percent agreement
            ** cohen_kappa (improvement on percent agreement)
            ** weighted_cohen_kappa (as cohen_kappa but penalty for disagreement depends on type of disagreement).

        for example, you might want a '2' vs '3' disagreement to be less severe than a '0' vs '3' disagreement.
        In this case a dictionary of weight definitions needs to be specified in **kwargs
        ==> To get one reliability metric over all analysts (reference_column must be None to use this):
        ** fleiss_kappa, or weighted fleiss kappa (also requires weights specified in **kwargs)

    questions_to_include: list of strings
        list of questions that should be considered in reliability calculation

    reliability_combination_function: lambda (DEPRECATED)
        Run this function to combine reliabilities across functions for each analyst

    reference_analyst: string
        If you want your reliability to be with respect to a ground truth analyst (such
        as the thompson ADOS results), specify that column here. In that case use_combination_for_reliability_analysis
        should be set to True. Otherwise use_combination_for_reliability_analysis
        should be False, and reference_analyst should be "combined" to use the results of this combination.

    combination_function: lambda
        The function to use when performing a combination

    use_combination_for_reliability_analysis: boolean
        If you will run a combination and then use it in the reliability analysis, make this flag true
        .... Should be True if reference_analyst set to "combined", otherwise False.
        other parameters only matter if this is true:

    subject_id_column_name: string
        optional if you want to redfine the column of which subjects are which across analysts

    analyst_id_column_name: string
        optional if you want to redefine the column of which analysts are which

    other_groupby_keys: list of strings
        optional, if you want to group by more than just the subjects when doing the initial combination
        (recommended to also group by the "Triton Video Version" since analysts sometimes are inconsistent here)

    prior_analyst_weights_hypothesis_dict: dict
        if you want initial reliability evaluation to use a combination with a prior weight hypothesis
        specify a dict with analyst Ids as keys and weights as values.

    Returns
    -------

    combined_function_and_kwargs: dict
        Dictionary that contains the function and associated kwargs that gets serialized
        (in case user wants to use it immediately).

    '''

    def _weighted_mode_combination(analyst_scores_dict, analyst_weights_dict):
        '''
        Given a single dictionary of analyst responses and the reliability weights of each analyst,
        do a weighted mode combination.
        '''
        for weight in analyst_weights_dict.values():
            if not isinstance(weight, numbers.Number):
                raise TypeError('Error, weighting '+str(weight)+' not numeric')
            if not np.isfinite(weight):
                raise ValueError('Error, weighting '+str(weight)+' invalid')
        value_weight_dict = {}
        for analyst, score in analyst_scores_dict.iteritems():
            if analyst not in analyst_weights_dict:
                raise KeyError('analyst '+str(analyst)+' not recognized')
            if score not in value_weight_dict.keys():
                value_weight_dict[score] = analyst_weights_dict[analyst]
            else:
                # Negative weights can arise from metrics like
                # Kappa if we have analysts who perform worse than
                # Random. These analysts should not be counted, rather
                # than counted as an anti-vote.
                if analyst_weights_dict[analyst]>0:
                    value_weight_dict[score] += analyst_weights_dict[analyst]
        if len(value_weight_dict.values())==0:
            raise ValueError('No valid analysts found to do a weighted mode combination')

        weight_list = value_weight_dict.values()
        score_list = value_weight_dict.keys()
        max_weight = max(weight_list)
        max_weight_index = weight_list.index(max_weight)
        score_weighted_mode = score_list[max_weight_index]
        diagnostic_info = {'max_weight': max_weight}

        return score_weighted_mode, diagnostic_info

    if combination_function is None:
        combination_function = _weighted_mode_combination

    if debug:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Run reliability analysis and create pickled file for the following settings:'
        print 'analysts to evaluate: ', np.unique(analysis_df[analyst_id_column_name].values)
        print 'Ground truth: ', reference_analyst
        print 'Reliability calculation to perform: ', reliability_calculation_to_do
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    # First convert input dataframe to the format that is needed for analysis
    analyst_comparison_df = _create_analyst_comparison_df(in_df,
                                                          questions_to_include=questions_to_include,
                                                          analyst_id_column_name=analyst_id_column_name,
                                                          reference_analysts=[reference_analyst],
                                                          exclude_these_analysts=exclude_these_analysts,
                                                          subject_id_column_name=subject_id_column_name,
                                                          other_groupby_keys=other_groupby_keys,
                                                          prior_analyst_weights_hypothesis_dict=prior_analyst_weights_hypothesis_dict,
                                                          debug=debug)

    print 'Now evaluate reliabilties'
    analyst_reliabilities = rate_analysts_against_reference(analyst_comparison_df,
                                                            rating_method=reliability_calculation_to_do,
                                                            questions_to_include=questions_to_include,
                                                            reference_column_name=reference_analyst,
                                                            subject_id_column_name=subject_id_column_name,
                                                            debug=debug)
    print 'Now save picked file'
    if how_serialized == 'pickle':
        combined_function_and_kwargs = _make_pickled_combination_function(combination_function,
                                                                          out_filename=out_filename,
                                                                          analyst_weights_dict=analyst_reliabilities,
                                                                          debug=debug)
    else:
        raise NotImplementedError('Have not implemented function serialization scheme: '+how_serialized)

    return combined_function_and_kwargs


def get_combination_function(in_filename, how='pickle'):
    if how=='pickle':
        return get_pickled_combination_function(in_filename)
    else:
        raise NotImplementedError('Serialization scheme '+how+' not implemented yet')


def get_pickled_combination_function(in_filename):
    '''
    Extract the combination function and any associated arguments from in_filename.
    It is expected that users will call this function when they want to extract the prebuilt
    pickled file.
    '''
    loaded_params = dill.load(open(in_filename, 'rb'))
    combination_function = loaded_params['function']
    kwargs = {key: value for key, value in loaded_params.iteritems() if key != 'function'}
    return combination_function, kwargs


def _make_pickled_combination_function(combination_function, out_filename, debug=False, **kwargs):
    '''
    Make a pickled file that contains all necessary information to perform a combination.

    Parameters
    -------
    combination_function: lambda
        can refer to any function that performs a combination of
        an analysis dataframe in a format with one row per combination of subject and analyst,
        and columns representing different question responses. If encoding is desired it must
        be done before calling this function.

    out_filename: string
        purpose of this function is to create this pickled file

    **kwargs: dict
        Any necessary arguments to successfully execute combination_function should be passed in the **kwargs.
        They will be pickled as well

    Returns
    -------

    pickle_this: dill object
        pickled object in case user wants to use it immediately

    '''
    pickle_this = {'function': combination_function}
    for key in kwargs.keys():
        pickle_this[key] = kwargs[key]
    if debug:
        print 'pickle_this: ', pickle_this
        print 'out filename: ', out_filename

    dill.dump(pickle_this, open(out_filename, 'wb'))

    return pickle_this
