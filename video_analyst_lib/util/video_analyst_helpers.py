'''
Helper functions for video_analyst_lib.py
'''

import numpy as np
import pandas as pd
import math
import sys
import copy as cp
import numbers
import collections
from sklearn import metrics


ALL_ANALYSTS_EXCEPT = 'all_analysts_except_'


def _create_analyst_comparison_df(analyst_scoresheet_df,
                                  questions_to_include_in_comparison,
                                  analyst_id_column_name,
                                  reference_analysts,
                                  exclude_these_analysts,
                                  subject_id_column_name,
                                  other_groupby_keys=[],
                                  prior_analyst_weights_hypothesis_dict=None,
                                  indeterminate_answer_code=9,
                                  debug=False):
    '''
    This is the main API endpoint to convert an input dataframe into a format
    convenient for analysis by the other API in this lib.

    Parameters
    -------
    analyst_scoresheet_df : pandas.DataFrame
       input dataframe, expected to contain one unique (submission, analyst) per row, and multiple columns, one per question,
       containing question answers in the proper encoding

    reference_analysts : a list
       this is a list of analyst names that should not be used in the analysis
       (presumably because they will be considered as ground truth options to compared with).
       These will not have the analyst naming convention enforced on them.
       This function also enforces a naming convention that every analyst column must have a prefix 'analyst_'.
       The analysis code looks for this prefix to decide which columns are for the analysts that should be evaluated.

    prior_analyst_weights_hypothesis_dict: dictionary
       relative weights per analyst. These are used when combining analyst judgments using majority vote

    Returns
    -------
    analyst_comparison_df : pandas.DataFrame
        converted data frame
    '''

    def _enforce_analyst_naming_convention(name):
        ''' needed so that after conversion to analysts into columns it is clear which columns are for real analysts
        '''
        if name in reference_analysts:
            # # This is a special case
            return name
        name = 'analyst_'+str(name) if 'analyst_' not in str(name) else name
        return name


    def _get_combined_df(in_df,
                         group_keys,
                         questions_to_include_in_comparison,
                         analyst_id_column_name,
                         weights_key=None,
                         new_name=THE_OTHER_ANALYSTS,
                         exclude_analyst_list=[]):
        '''
        Helper function for evaluate_analyst_reliabilities_and_pickle_combination_function
        does an initial combination of the input dataframe before running the reliability analysis

        Parameters
        -------
        in_df: pandas.DataFrame
            the dataframe on which to run the combination

        group_keys: list
            perform the combination in pieces on these groups

        analyst_id_column_name: string
            the column that defines the analysts that are aggregated together

        weights_key: string
            if you have a prior weights assumption, pass it here

        exclude_analyst_list: list of strings
            if you want combination to not include some analysts. Common if you are trying to compare one against others

        Returns
        -------
        out_feature_df: pandas.DataFrame
            a new dataframe that contains only the new rows tagged on with combined values
        '''

        # Make sure to reset indices at beginning to ensure things are not scrambled later
        df = in_df.reset_index()
        if exclude_analyst_list != []:
            df = df[~df[analyst_id_column_name].isin(exclude_analyst_list)]

        if weights_key is None:   #  Treat all rows equally
            weights_key = 'prior_analyst_weights'
            df[weights_key] = [1.]*len(df.index)

        weighted_mode_operation = lambda x: weighted_mode(x, weights=df.loc[x.index, weights_key])

        def weighted_mode(grouped_series, weights):
            # maybe replace this by a call to weighted_mode_combination once you have other pieces working??
            values_to_agg = grouped_series.values
            unique_values = np.unique(values_to_agg)

            # in case of ties want consistent ordered results
            value_weight_dict = collections.OrderedDict()
            max_value = None
            max_weight = None
            for value in unique_values:
                these_weights = weights[values_to_agg==value]
                total_weight = np.sum(weights[values_to_agg==value])
                value_weight_dict[value] = total_weight

                if max_weight is None or max_weight < total_weight:
                    max_value = value
                    max_weight = total_weight
            return max_value

        # Do weighted mode calculation on questions, and just take the first observed value in any non-questions:
        grouped_df = df.groupby(group_keys)
        agg_functions_to_apply = {column_name: weighted_mode_operation for column_name in questions_to_include_in_comparison}
        spectator_column_names = [column_name for column_name in df.columns if column_name not in questions_to_include_in_comparison]
        for spectator_column_name in spectator_column_names:
            agg_functions_to_apply[spectator_column_name] = lambda x: x.values[0]
        out_feature_df = grouped_df.agg(agg_functions_to_apply)
        out_feature_df[analyst_id_column_name] = [new_name]*len(out_feature_df.index)

        return out_feature_df

    def _build_all_permutation_of_exclusions_combined_df(analysis_df,
                                                         group_keys,
                                                         questions_to_include_in_comparison,
                                                         analyst_id_column_name,
                                                         weights_key,
                                                         all_analyst_names):
        '''
        Build all permutations of _get_combined_df with one analyst held out.
        This is the format that will be needed for downstream analysis.

        Parameters
        -------
        analysis_df: pandas.DataFrame
            the dataframe on which to run the combination

        group_keys: list
            perform the combination in pieces on these groups

        questions_to_include_in_comparison: unknown

        analyst_id_column_name: string
            the column that defines the analysts that are aggregated together

        weights_key: string
            if you have a prior weights assumption, pass it here

        all_analyst_names: list of strings
            list of analysts to process

        Returns
        -------
        reliability_analysis_df: pandas.DataFrame
            a new dataframe
        '''

        reliability_analysis_df = None
        for analyst in all_analyst_names:
            this_comb_name = ALL_ANALYSTS_EXCEPT +str(analyst)
            this_analyst_combined_df = _get_combined_df(analysis_df, group_keys=group_keys, questions_to_include_in_comparison=questions_to_include_in_comparison,
                analyst_id_column_name=analyst_id_column_name, weights_key=weights_key, new_name=this_comb_name, exclude_analyst_list=[analyst])
            if reliability_analysis_df is None: reliability_analysis_df = pd.concat([analysis_df, this_analyst_combined_df], ignore_index=True)
            else: reliability_analysis_df = pd.concat([reliability_analysis_df, this_analyst_combined_df], ignore_index=True)

        return reliability_analysis_df

    if exclude_these_analysts == []:
       analysis_df = cp.deepcopy(analyst_scoresheet_df)
    else:
       analysis_df = analyst_scoresheet_df[~analyst_scoresheet_df[analyst_id_column_name].isin(exclude_these_analysts)]

    analysis_df[analyst_id_column_name] = analysis_df[analyst_id_column_name].apply(_enforce_analyst_naming_convention)
    all_analyst_names = np.array([name for name in np.unique(analysis_df[analyst_id_column_name].values) if name.startswith('analyst_')])

    truth_output_column_names = reference_analysts

    group_keys = [subject_id_column_name] + other_groupby_keys
    if prior_analyst_weights_hypothesis_dict is not None:
        analysis_df['analyst_weight'] = full_df[analyst_id_column_name].apply(lambda x: prior_analyst_weights_hypothesis_dict[x])
        analysis_df['prior_analyst_weights'] = [prior_analyst_weights_hypothesis_dict[aid] for aid in analysis_df[analyst_id_column_name].values]
        prior_weights_key = 'prior_analyst_weights'
    else:
        prior_weights_key = None

    reliability_analysis_df = _build_all_permutation_of_exclusions_combined_df(analysis_df,
                                                                               group_keys=group_keys,
                                                                               questions_to_include_in_comparison=questions_to_include_in_comparison,
                                                                               analyst_id_column_name=analyst_id_column_name,
                                                                               weights_key=prior_weights_key,
                                                                               all_analyst_names=all_analyst_names)

    # pivot dataframe to key on "comparison format"
    if debug:
        print 'Now convert data frame to format for reliability analysis'

    analyst_comparison_df = _convert_df_from_questions_in_columns_to_rows(reliability_analysis_df,
                                                                                 questions_to_include=questions_to_include_in_comparison,
                                                                                 subject_id_column_name=subject_id_column_name,
                                                                                 analyst_id_column_name=analyst_id_column_name,
                                                                                 other_groupby_keys=other_groupby_keys)

    # add determinate answer info as boolean columns, one per analyst
    for analyst in all_analyst_names:
        analyst_comparison_df['determinate_answer_by_'+analyst] = np.where(analyst_comparison_df[analyst]==indeterminate_answer_code, 0, 1)

    return analyst_comparison_df


def get_separate_reliabilities_for_all_analysts_and_questions(dataframe,
                                                              calculation_to_do,
                                                              questions_to_include,
                                                              reference_column_name=None,
                                                              debug=True,
                                                              **kwargs):
    '''
    Get reliability metrics for one or more questions for all contained analysts.
    This function is like get_reliabilities_for_given_question, except that there is an
    additional column in the input to specify the question being asked. These questions are
    looped over and get_reliabilities_for_given_question is called for each question

    Parameters
    -------
    dataframe: pandas.DataFrame
        should be encoded in desired format before passing to this function. There should
        be one column for each analyst WHICH MUST USE PREFIX 'analyst_', and one row for
        each subject, with entries representing the responses. Another column should represent
        the question that the response is for.

    calculation_to_do: function
        reliability calculation to perform

    questions_to_include: list
        list of questions to calculate reliabilities on

    reference_column_name: string
        "ground truth" column to calculate reliabilities with respect to (might be a combination of other analysts

    Returns
    -------
    aggregate_reliabilities_df : pandas.DataFrame
        converted data frame
    '''

    each_question_reliabilities_dfs = []
    for question in questions_to_include:
        this_question_df = dataframe[dataframe['question']==question]
        if len(this_question_df.index)==0:
            print 'Question ', question, ' not present in data. Skip it.'
            continue
        this_question_reliabilities_df = _get_reliabilities_for_given_question(this_question_df, calculation_to_do,
                question, reference_column_name, **kwargs)
        each_question_reliabilities_dfs.append(this_question_reliabilities_df)
    aggregate_reliabilities_df = pd.concat(each_question_reliabilities_dfs, ignore_index=True)

    return aggregate_reliabilities_df


def _get_reliabilities_for_given_question(dataframe, calculation_to_do, question, reference_column_name=None, **kwargs):
    '''
    Get reliability metrics for some number of analysts and evaluations on a particular question.

    Parameters
    -------
    dataframe: pandas.DataFrame
        should be encoded in desired format before passing to this function. There should
        be one column for each analyst WHICH MUST USE PREFIX 'analyst_', and one row for
        each subject, with entries representing the responses. Should also have a reference column
        if desired as a ground truth to evaluate analysts against.
        Note: Missing/not answered results should be included as either np.nan, 'nan', or ''.
        All other responses are assumed to be valid.

    calculation_to_do: string
        string to specify which reliability metric to calculate. Suggested options to support:
        ==> To get one reliability metric per analyzer. Options:
          ** percent agreement
          ** cohen_kappa (improvement on percent agreement)
          ** weighted_cohen_kappa (as cohen_kappa but penalty for disagreement depends on type of disagreement).
        for example, you might want a '2' vs '3' disagreement to be less severe than a '0' vs '3' disagreement.
        In this case a dictionary of weight definitions needs to be specified in **kwargs
        ==> To get one reliability metric over all analysts (reference_column must be None to use this):
          ** fleiss kappa, or weighted fleiss kappa (also requires weights specified in **kwargs)

    reference_column_name: string
        name of column that reliability is compared with (not defined for fleiss kappas)

    question: string
        the question being evaluated

    Returns
    -------
    reliability_results_df: pandas.DataFrame
        contains one reliability value per analyst
    '''

    # NOTE: old comment
    # If calculation_to_do specifies that there should be a metric for each analyst, returns a dictionary that
    # contains one reliability value per analyst. If calculation_to_do specifies a single metric over all analysts
    # (such as the fleiss kappa), returns a single float as reliability.

    if 'question' in dataframe.columns:
        ## Make sure he same question is being consistently evaluated
        all_questions = np.unique(dataframe['question'].values)
        if len(all_questions)>1:
            raise ValueError('Should only pass data about a single question to get_reliabilities_for_given_question. '+\
                    ' The following questions were observed: '+str(all_questions))

    analysts_to_evaluate = [column_name for column_name in dataframe.columns if column_name.startswith('analyst_')]
    if len(analysts_to_evaluate)==0:
        raise ValueError('No analysts found in columns of input dataframe. Did you forget to specify "analyst_"'+\
                ' as prefix?')

    if calculation_to_do in ['percent_agreement', 'cohen_kappa', 'weighted_cohen_kappa']:
        separate_reliability_estimate_for_each_analyst = True
    elif calculation_to_do in ['fleiss_kappa', 'weighted_fleiss_cappa']:
        separate_reliability_estimate_for_each_analyst = False
    else:
        raise ValueError('calculation_to_do '+calculation_to_do+' not understood')
    reliability_results = {'question': [], 'reference': [], 'analyst': [], 'calculation': [], 'reliability': []}
    if separate_reliability_estimate_for_each_analyst:
        for analyst in analysts_to_evaluate:
            if reference_column_name == THE_OTHER_ANALYSTS:
                this_reference_column_name = ALL_ANALYSTS_EXCEPT +str(analyst)
            else:
                this_reference_column_name = reference_column_name
            if analyst not in dataframe:
            	raise ValueError('_get_reliabilities_for_given_question(): dataframe should contain analyst column '+analyst+" but doesn't")
            if this_reference_column_name not in dataframe:
             	raise ValueError('_get_reliabilities_for_given_question(): dataframe should contain this_reference_column_name column '+ str(this_reference_column_name) +" but doesn't")

            #HALIM: IF THE separate_reliability_estimate_for_each_analyst is TRUE, AND reference IS NOT THE_OTHER_ANALYSTS ,
            #THEN FILTER dataframe TO EXCLUDE ALL INSTANCES WHERE THE analyst IN QUESTION GAVE AN INDETERMINATE ANSWER
            if (reference_column_name!=THE_OTHER_ANALYSTS and separate_reliability_estimate_for_each_analyst):
                filtered_df = dataframe[dataframe["determinate_answer_by_"+analyst]==1]
            else:
                filtered_df = dataframe.copy(deep=True)

            reliability = _get_reliability_for_given_analyst_and_question(filtered_df[analyst].values, filtered_df[this_reference_column_name].values,
                calculation_to_do, **kwargs)
            reliability_results['analyst'].append(analyst)
            reliability_results['reliability'].append(reliability)
            reliability_results['reference'].append(reference_column_name)
            reliability_results['question'].append(question)
            reliability_results['calculation'].append(calculation_to_do)
    else:
        #raise NotImplementedError('calculation_to_do '+calculation_to_do+' not implemented yet')
        reliability = _get_reliability_given_all_analysts_for_question(dataframe, analysts_to_evaluate,
                calculation_to_do, **kwargs)
        reliability_results['analyst'].append( 'all_analysts_aggregate')
        reliability_results['reliability'].append( reliability)
        reliability_results['reference'].append(reference_column_name)
        reliability_results['question'].append(question)
        reliability_results['calculation'].append(calculation_to_do)
    reliability_results_df = pd.DataFrame(reliability_results)

    return reliability_results_df


def _is_this_value_missing(element, missing_values_to_check=['', np.nan, 'nan']):
    '''
    Helper function to check a given element to see if it is
    of a type we would consider to be "missing"
    '''

    if element is None:
        return True

    for missing_value in missing_values_to_check:
        # Checking nans in a way that doesn't crash on strings is complicated.
        # There must be some better way to do this???
        try:
            if np.isnan(missing_value) and np.isnan(element):
                return True   # Equality check invalid
        except:
            pass # not nan (what's a better way to do this safely?)

        if element == missing_value:
            return True

    return False


def _no_missing_values_in_row(x, analysts_to_evaluate):
    for analyst in analysts_to_evaluate:
        if _is_this_value_missing(x[analyst]):
            return False
    return True


def _get_reliability_given_all_analysts_for_question(dataframe, analysts_to_evaluate, calculation_to_do, **kwargs):
    '''
    Calculate reliability for all analysts for a given question.

    Note: Missing/not answered results should be included as either np.nan, 'nan', or ''.
    All other responses are assumed to be valid.

    Parameters
    -------
    dataframe: pandas.DataFrame
        results for all analysts for a given question.
        Some values may be empty if not all completed.

    calulation_to_do: string
        only "fleiss_kappa" implemented for this approach at the moment

    Returns
    -------
    fleiss_kappa: float
        reliability result
    '''

    def reformat_df_for_fleiss_kappa(in_df):
        '''
        Assumes input is DF with analysts in columns, subjects in rows, and each response
        in the field entries. Want to reformat to dataframe with subjects in rows, response categories in
        columns, and counts of number of analysts responding with that rating for that subject in each entry
        '''

        #  First transpose to get subjects in columns and analysts in rows:
        transpose_df = in_df.transpose()
        #  Now count responses for each subject:
        count_dict = {}
        for column_name in transpose_df.columns:
            count_dict[column_name] = transpose_df[column_name].value_counts()
        count_df = pd.DataFrame(count_dict)
        #  Now have subjects in columns and response categories in rows with counts in entries. Re-transpose.
        result_df = count_df.transpose()
        #  Nan's where values are missing. Replace with zeros.
        result_df = result_df.fillna(0)
        #  Make sure to recover original indexing if any
        result_df.index = in_df.index

        return result_df

    if calculation_to_do not in ['fleiss_kappa']:
        raise ValueError('calculation_to_do '+calculation_to_do+' not defined for all analysts calculation')

    all_completed_df = dataframe[dataframe.apply(_no_missing_values_in_row, axis=1, args=(analysts_to_evaluate,))]

    if calculation_to_do == 'fleiss_kappa':
        refornatted_for_fleiss_df = reformat_df_for_fleiss_kappa(all_completed_df[analysts_to_evaluate])

        print refornatted_for_fleiss_df

        fleiss_kappa = _compute_fleiss_kappa(refornatted_for_fleiss_df.values.tolist())
        return fleiss_kappa


def _get_reliability_for_given_analyst_and_question(analyst_results, reference_results, calculation_to_do, **kwargs):
    '''
    Calculate reliability for a particular analyst.

    Note: Missing/not answered results should be included as either np.nan, 'nan', or ''.
    All other responses are assumed to be valid.

    Parameters
    -------
    analyst_results: numpy.array
        a numpy array of results for one or more analysts.
        Some values may be empty if not all completed.

    reference_results: numpy.array
        the "ground truth" reference numpy array of results.
        Some values may be null if not all completed.

    calulation_to_do: string
        see documentation of get_reliabilities_for_given_question

    Returns
    -------
    kappa_value: float
        reliability result
    '''

    def get_results_completed_by_both(analyst_results, reference_results):
        ''' analyst results and reference results may be missing data. Return arrays that contain only matching values for both arrays '''
        if len(analyst_results) != len(reference_results):
            raise ValueError('Mismatch between analyst_results and reference_result array length')
        ok_analyst_results = []
        ok_reference_results = []
        for idx in range(len(analyst_results)):
            if _is_this_value_missing(analyst_results[idx]) or _is_this_value_missing(reference_results[idx]):
                continue
            else:
                ok_analyst_results.append(analyst_results[idx])
                ok_reference_results.append(reference_results[idx])
        return np.array(ok_analyst_results), np.array(ok_reference_results)

    if calculation_to_do not in ['percent_agreement', 'cohen_kappa', 'weighted_cohen_kappa']:
        raise ValueError('calculation_to_do '+calculation_to_do+' not defined for analyst vs reference calculation')
    valid_analyst_results, valid_reference_results = get_results_completed_by_both(analyst_results, reference_results)
    if calculation_to_do == 'percent_agreement':
        n_results_tot = len(valid_analyst_results)
        n_results_agreeing = len([analyst_value for analyst_value, reference_value in\
            zip(analyst_results, reference_results) if analyst_value == reference_value])
        frac_agreement = np.nan if n_results_tot==0 else float(n_results_agreeing) / float(n_results_tot)
        return frac_agreement

    if calculation_to_do == 'cohen_kappa':
        if len(valid_analyst_results) == 0 or len(valid_reference_results) == 0:
            print 'Not enough common valid results. Abort with value -999 for cohen kappa.'
            return -999
        kappa_value = metrics.cohen_kappa_score(valid_analyst_results, valid_reference_results)
        return kappa_value

    if calculation_to_do == 'weighted_cohen_kappa':
        weights = kwargs['weights'] if 'weights' in kwargs else [1]*len(valid_analyst_results)
        #  sklearn only has weights supported here in version 0.18 and later
        try:
            kappa_value = metrics.cohen_kappa_score(valid_analyst_results, valid_reference_results, weights=weights)
        except Exception as exception_msg:
                print 'Failed to evaluate cohen kappa with weights. Are you using an older version of sklearn (only implemented in version 0.18 and later)?'
                raise ValueError(exception_msg)
        return kappa_value


def _compute_fleiss_kappa(mat, debug=False):
    '''
    Copied  with minor reformatting from https://en.wikibooks.org/wiki/Algorithm_Implementation/Statistics/Fleiss%27_kappa

    Parameters
    -------
    mat: numpy.array
        Matrix[subjects][categories]

    Returns
    -------
    kappa: float
        reliability result
    '''

    def checkEachLineCount(mat):
        '''
        Assert that each line has a constant number of ratings

        Parameters
        -------
        mat: numpy.array
            The matrix checked

        Returns
        -------
        n: int
            The number of ratings

        Raises
        -------
        AssertionError: Error
            If lines contain different number of ratings
        '''

        n = sum(mat[0])

        assert all(sum(line) == n for line in mat[1:]), "Line count != %d (n value)." % n
        return n

    n = checkEachLineCount(mat)   # PRE : every line count must be equal to n
    N = len(mat)
    k = len(mat[0])

    if debug:
        print n, "raters."
        print N, "subjects."
        print k, "categories."

    # Computing p[]
    p = [0.0] * k
    for j in xrange(k):
        p[j] = 0.0
        for i in xrange(N):
            p[j] += mat[i][j]
        p[j] /= N*n

    if debug:
        print "p =", p

    # Computing P[]
    P = [0.0] * N
    for i in xrange(N):
        P[i] = 0.0
        for j in xrange(k):
            P[i] += mat[i][j] * mat[i][j]
        P[i] = (P[i] - n) / (n * (n - 1))

    if debug:
        print "P =", P

    # Computing Pbar
    Pbar = sum(P) / N

    if debug:
        print "Pbar =", Pbar

    # Computing PbarE
    PbarE = 0.0
    for pj in p:
        PbarE += pj * pj

    if debug:
        print "PbarE =", PbarE

    kappa = (Pbar - PbarE) / (1 - PbarE)

    if debug:
        print "kappa =", kappa

    return kappa


def _convert_df_from_questions_in_columns_to_rows(in_df,
                                                  questions_to_include,
                                                  subject_id_column_name,
                                                  analyst_id_column_name,
                                                  spectator_column_names=[],
                                                  other_groupby_keys=[]):
    '''
    Helper function if needed depending on the format of the input dataframe for the reliability analysis

    Assumes input has format with a single column for subject Id, a single column for analyst id, and
    a different column for each question.

    Outputs dataframe in the format the rest of this code expects: one column per analyst, and each row being
    a set of responses for a particular subject and a particular question.

    Parameters
    -------
    in_df: pandas.DataFrame
        pre-converted dataframe. An example format might be:
        Cognoa Id, analyst Id, ados1_a1, ados1_a3, ...
        10          121          0          2 ...
        11          121          0          1 ...
        ...

    questions_to_include: list of strings
        these are the columns that should be considered questions, each of which will become a different row

    spectator_column_names: list of strings
        columns that do not fall into any other category, but which you want to keep in the output dataframe

    other_groupby_keys: list of strings
        when joining output across analysts, what additional matching criteria is there besides subject id and
        question (if any). Common example: triton video version.

    Returns
    -------
    result_df: pandas.DataFrame
        converted format dataframe. An example format might be:
        Cognoa Id, question, analyst_121, analyst_192, ...
        10        ados1_a1         0       2 ...
        10        ados1_a3         2       1 ...
        ...

    '''
    unique_analysts = np.unique(in_df[analyst_id_column_name].values)
    result_df = None
    for analyst in unique_analysts:

        analyst_str = str(analyst)
        this_analyst_df = in_df[in_df[analyst_id_column_name]==analyst]

        #  The analyst_str part of this dictionary will be filled with responses
        results_for_this_analyst = {analyst_str: [], subject_id_column_name: [], 'question': []}
        for column_name in spectator_column_names+other_groupby_keys:
            results_for_this_analyst[column_name] = []
        # this df has columns of questions and rows of subjects. Want to collapse to single
        # for answer with the question name in new column
        for question in questions_to_include:
            subject_responses = this_analyst_df[question].values
            subject_ids = this_analyst_df[subject_id_column_name].values
            questions = [question]*len(subject_ids)

            results_for_this_analyst[analyst_str] += list(subject_responses)
            results_for_this_analyst[subject_id_column_name] += list(subject_ids)
            results_for_this_analyst['question'] += questions
            for column_name in spectator_column_names+other_groupby_keys:
                results_for_this_analyst[column_name] += list(this_analyst_df[column_name].values)

        this_analyst_out_df = pd.DataFrame(results_for_this_analyst)
        if result_df is None:
            result_df = cp.deepcopy(this_analyst_out_df)
        else:
            questions_to_merge_on = [subject_id_column_name, 'question']+other_groupby_keys
            if 'Triton Video Version' in result_df.columns:
                questions_to_merge_on.append('Triton Video Version')
            result_df = result_df.merge(this_analyst_out_df, on=questions_to_merge_on, how='outer')

    return result_df
