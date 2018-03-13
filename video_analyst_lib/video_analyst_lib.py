'''
This module contains functions that accomplish the tasks of determining the reliability
of questions and analysts, as well as the task of combining the results of analysts, and of
pickling and outputing a function that can be used to do the combination
'''

import numpy as np
import pandas as pd

import math
import sys
import copy as cp
import numbers
import collections

from sklearn import metrics
import dill

from util.video_analyst_helpers import *
from util.video_analyst_combo import *


def rate_analysts_against_eachother(scoresheet_df,
                                    analysts_to_rate,
                                    questions_to_rate_over,
                                    rating_aggregation_method=np.mean,
                                    analyst_id_column_name='analyst',
                                    subject_id_column_name='submission'):
    '''
    This function is what you should use to determine the reliabilitiy of analysts over some questions, each compared to the rest

    Functionally, it will perform the rating method on every analyst and for every question.
    Then it will aggregate these ratings across questions using an aggregation method of your choice.

    Parameters
    -------
    scoresheet_df : pandas.DataFrame
       Dataframe of raw data, in format of one row per subject and per question, and one column per analyst

    analysts_to_rate: list of strings
        list of analysts to include in the rating

    questions_to_rate_over: list of strings
        list of questions to calculate reliabilities for and aggregate

    rating_aggregation_method: lambda
        function to combine individual rating calculations

    analyst_id_column_name: string
        Name of the Dataframe column containing analysts
        Default: 'analyst'

    subject_id_column_name: string
        Name of the Dataframe column containing subjects
        Default: 'submission'

    Returns
    -------
    output_dict: dict
        Dictionary containing Dataframes describing results for all questions and per question
    '''

    # make sure we're dealing with lists in case the user passed some other iterable
    analysts_to_rate = list(analysts_to_rate)
    questions_to_rate_over = list(questions_to_rate_over)

    # note: weighted_cohen_kappa, weighted_fleiss_kappa not supported for analyst rating
    supported_rating_methods = ['percent_agreement', 'cohen_kappa']

    # make sure all analysts the user wants to rate actually exist in the passed dataframe
    if analysts_to_rate is None or len(analysts_to_rate)<2:
        raise ValueError("To compare analyst to analyst you need to include at least two of them")

    present_analysts = set(scoresheet_df[analyst_id_column_name])
    for analyst in analysts_to_rate:
        if analyst not in present_analysts:
            raise ValueError("Couldn't find any data for analyst "+analyst+". Make sure the input scoresheet dataframe is complete")

    # every analyst in the scoresheet other than the ones specified by the user will be excluded
    analysts_to_exclude = present_analysts - set(analysts_to_rate)

    # pivot the scoresheet dataset into what is called a "comparison dataframe" format
    analyst_comparison_df = _create_analyst_comparison_df(scoresheet_df,
                                                          questions_to_include_in_comparison=questions_to_rate_over,
                                                          analyst_id_column_name=analyst_id_column_name,
                                                          reference_analysts=[],
                                                          exclude_these_analysts=analysts_to_exclude,
                                                          subject_id_column_name= subject_id_column_name,
                                                          indeterminate_answer_code=9,
                                                          debug=False)

    # compute every supported agreement metric, one at a time
    analyst_reliability_output = {}
    analyst_question_reliability_output = {}

    for rating_method in supported_rating_methods:
        # call on some helper functions to run comparisons on a question and analyst levels
        analyst_question_reliabilities = get_separate_reliabilities_for_all_analysts_and_questions(analyst_comparison_df,
                                                                                                   calculation_to_do=rating_method,
                                                                                                   questions_to_include=questions_to_rate_over,
                                                                                                   reference_column_name=THE_OTHER_ANALYSTS,
                                                                                                   debug=False)

        analyst_question_reliabilities = analyst_question_reliabilities.sort_values(analyst_id_column_name)
        analyst_reliabilities = analyst_question_reliabilities.groupby(analyst_id_column_name).apply(rating_aggregation_method)
        analyst_reliabilities = analyst_reliabilities.sort_index()

        #put the question-specific agreement scores into the output data structure
        analyst_question_reliability_output[rating_method] = {}
        for question in questions_to_rate_over:
            analyst_question_reliability_output[rating_method][question] = [0.0] * len(analysts_to_rate)

        zipped = zip(analyst_question_reliabilities[analyst_id_column_name], analyst_question_reliabilities['question'], analyst_question_reliabilities['reliability'].values)
        for analyst, question, reliability in zipped:
            idx = analysts_to_rate.index(analyst.replace('analyst_',''))
            analyst_question_reliability_output[rating_method][question][idx] = reliability

        #put the question-aggregated agreement scores into the output data structure
        analyst_reliability_output[rating_method] = [0.0] * len(analysts_to_rate)
        zipped = zip(analyst_reliabilities.index, analyst_reliabilities['reliability'].values)
        for analyst, reliability in zipped:
            idx = analysts_to_rate.index(analyst.replace('analyst_',''))
            analyst_reliability_output[rating_method][idx] = reliability

    # now let's evaluate the 'coverage' rate per question per analyst
    analyst_question_coverage_output = {}
    for question in questions_to_rate_over:
        analyst_question_coverage_output[question] = []
        for analyst in analysts_to_rate:
            subset = analyst_comparison_df[analyst_comparison_df['question']==question]
            analyst_question_coverage = rating_aggregation_method(subset['determinate_answer_by_analyst_'+analyst].values)
            analyst_question_coverage_output[question].append(analyst_question_coverage)

    # now let's evaluate the 'coverage' rate per analyst over all questions
    analyst_coverage_output = []
    coverage_df = analyst_comparison_df.groupby('question').apply(rating_aggregation_method)
    for analyst in analysts_to_rate:
        analyst_overall_coverage = rating_aggregation_method(coverage_df['determinate_answer_by_analyst_'+analyst].values)
        analyst_coverage_output.append(analyst_overall_coverage)

    # prepare the agreement over all questions as a separate Dataframe
    all_Qs_output_dictionary = {analyst_id_column_name:analysts_to_rate, 'coverage':analyst_coverage_output}
    for rating_method in supported_rating_methods:
        all_Qs_output_dictionary['agreement_using_'+rating_method] = analyst_reliability_output[rating_method]

    # prepare the agreement per question as a separate Dataframe
    per_Q_output_dictionary = {analyst_id_column_name:analysts_to_rate}
    for question in questions_to_rate_over:
        per_Q_output_dictionary['coverage_on_'+question] = analyst_question_coverage_output[question]
        for rating_method in supported_rating_methods:
            per_Q_output_dictionary['agreement_on_'+question+'_using_'+rating_method] = analyst_question_reliability_output[rating_method][question]

    # format the outputs into a dictionary of DataFrame to return to caller
    output_dict = {}
    output_dict['agreement_over_all_questions'] = pd.DataFrame(all_Qs_output_dictionary)
    output_dict['agreement_per_question'] = pd.DataFrame(per_Q_output_dictionary)

    return output_dict


def rate_analysts_against_reference(scoresheet_df,
                                    analysts_to_rate,
                                    reference,
                                    questions_to_rate_over,
                                    rating_aggregation_method=np.mean,
                                    analyst_id_column_name='analyst',
                                    subject_id_column_name='submission'):

    '''
    This function is what you should use to determine the reliabilitiy of analysts over some questions,
    each compared to a reference ground truth

    Functionally, it will perform the rating method on every analyst and for every question.
    Then it will aggregate these ratings across questions using an aggregation method of your choice.

    Parameters
    -------
    scoresheet_df : pandas.DataFrame
       Dataframe of raw data, in format of one row per subject and per question, and one column per analyst

    analysts_to_rate: list of strings
        list of analysts to include in the rating

    reference: string
        reference analyst or instrument to consider as ground truth

    questions_to_rate_over: list of strings
        list of questions to calculate reliabilities for and aggregate

    rating_aggregation_method: lambda
        function to combine individual rating calculations

    analyst_id_column_name: string
        Name of the Dataframe column containing analysts
        Default: 'analyst'

    subject_id_column_name: string
        Name of the Dataframe column containing subjects
        Default: 'submission'

    Returns
    -------
    output_dict: dict
        Dictionary containing Dataframes describing results for all questions and per question

    '''

    # make sure we're dealing with lists in case the user passed some other iterable
    analysts_to_rate = list(analysts_to_rate)
    questions_to_rate_over = list(questions_to_rate_over)

    # weighted_cohen_kappa currently turned off, and fleiss_kappa/weighted_fleiss_kappa not supported for analyst rating.
    supported_rating_methods = ['percent_agreement', 'cohen_kappa']

    # make sure all analysts the user wants to rate actually exist in the passed dataframe
    if analysts_to_rate is None or len(analysts_to_rate)<2:
            raise ValueError("To compare analyst to analyst you need to include at least two of them")
    present_analysts = set(scoresheet_df[analyst_id_column_name])
    for analyst in analysts_to_rate:
        if analyst not in present_analysts:
            raise ValueError("Couldn't find any data for analyst "+analyst+". Make sure the input scoresheet dataframe is complete")

    # make sure the reference analyst exists in the passed dataframe
    if reference is None:
            raise ValueError("To compare analyst to reference you need to specify a valid analyst or instrument as reference")
    present_analysts = set(scoresheet_df[analyst_id_column_name])
    if reference not in present_analysts:
            raise ValueError("Couldn't find any data for reference "+reference+". Make sure the input scoresheet dataframe is complete")

    # an analyst cannot be chosen as both a reference and as a rated analyst
    if reference in analysts_to_rate:
            raise ValueError("Cannot rate analyst "+reference+" while also using him/her as a reference")

    # every analyst in the scoresheet other than the ones specified by the user will be excluded
    analysts_to_exclude = present_analysts - set(analysts_to_rate+[reference])

    # pivot the scoresheet dataset into what is called a "comparison dataframe" format
    analyst_comparison_df = _create_analyst_comparison_df(scoresheet_df,
                                                          questions_to_include_in_comparison=questions_to_rate_over,
                                                          analyst_id_column_name=analyst_id_column_name,
                                                          reference_analysts=reference,
                                                          exclude_these_analysts=analysts_to_exclude,
                                                          subject_id_column_name= subject_id_column_name,
                                                          indeterminate_answer_code=9,
                                                          debug=False)

    # compute every supported agreement metric, one at a time
    analyst_reliability_output = {}
    analyst_question_reliability_output = {}

    for rating_method in supported_rating_methods:
        # call on some helper functions to run comparisons on a question and analyst levels
        analyst_question_reliabilities = get_separate_reliabilities_for_all_analysts_and_questions(analyst_comparison_df,
                                                                                            calculation_to_do=rating_method,
                                                                                            questions_to_include=questions_to_rate_over,
                                                                                            reference_column_name=reference,
                                                                                            debug=False)

        analyst_question_reliabilities = analyst_question_reliabilities.sort_values(analyst_id_column_name)

        analyst_reliabilities = analyst_question_reliabilities.groupby(analyst_id_column_name).apply(rating_aggregation_method)
        analyst_reliabilities = analyst_reliabilities.sort_index()

        # put the question-specific agreement scores into the output data structure
        analyst_question_reliability_output[rating_method] = {}
        for question in questions_to_rate_over:
            analyst_question_reliability_output[rating_method][question] = [0.0] * len(analysts_to_rate)

        zipped = zip(analyst_question_reliabilities[analyst_id_column_name],
                     analyst_question_reliabilities['question'],
                     analyst_question_reliabilities['reliability'].values)

        for analyst, question, reliability in zipped:
            idx = analysts_to_rate.index(analyst.replace('analyst_',''))
            analyst_question_reliability_output[rating_method][question][idx] = reliability

        # put the question-aggregated agreement scores into the output data structure
        analyst_reliability_output[rating_method] = [0.0] * len(analysts_to_rate)

        zipped = zip(analyst_reliabilities.index,
                     analyst_reliabilities['reliability'].values)

        for analyst, reliability in zipped:
            analyst_reliability_output[rating_method][analysts_to_rate.index(analyst.replace('analyst_',''))] = reliability

    # now let's evaluate the 'coverage' rate per question per analyst
    analyst_question_coverage_output = {}
    for question in questions_to_rate_over:
        analyst_question_coverage_output[question] = []
        for analyst in analysts_to_rate:
            subset = analyst_comparison_df[analyst_comparison_df['question']==question]
            analyst_question_coverage = rating_aggregation_method(subset['determinate_answer_by_analyst_'+analyst].values)
            analyst_question_coverage_output[question].append(analyst_question_coverage)

    # now let's evaluate the 'coverage' rate per analyst over all questions
    analyst_coverage_output = []
    coverage_df = analyst_comparison_df.groupby('question').apply(rating_aggregation_method)
    for analyst in analysts_to_rate:
        analyst_overall_coverage = rating_aggregation_method(coverage_df['determinate_answer_by_analyst_'+analyst].values)
        analyst_coverage_output.append(analyst_overall_coverage)

    # prepare the agreement over all questions as a separate Dataframe
    all_Qs_output_dictionary = {analyst_id_column_name:analysts_to_rate, 'coverage':analyst_coverage_output}
    for rating_method in supported_rating_methods:
        all_Qs_output_dictionary['agreement_with_'+reference+'_using_'+rating_method] = analyst_reliability_output[rating_method]

    # prepare the agreement per question as a separate Dataframe
    per_Q_output_dictionary = {analyst_id_column_name:analysts_to_rate}
    for question in questions_to_rate_over:
        per_Q_output_dictionary['coverage_on_'+question] = analyst_question_coverage_output[question]
        for rating_method in supported_rating_methods:
            per_Q_output_dictionary['agreement_with_'+reference+'_on_'+question+'_using_'+rating_method] = analyst_question_reliability_output[rating_method][question]

    # format the outputs into a dictionary of DataFrame to return to caller
    output_dict = {}
    output_dict['agreement_over_all_questions'] = pd.DataFrame(all_Qs_output_dictionary)
    output_dict['agreement_per_question'] = pd.DataFrame(per_Q_output_dictionary)

    return output_dict


def rate_questions_on_analyst_agreement_with_eachother(scoresheet_df,
                                                       questions_to_rate,
                                                       analysts_to_rate_over,
                                                       rating_aggregation_method=np.mean,
                                                       analyst_id_column_name='analyst',
                                                       subject_id_column_name='submission'):
    '''
    This function is what you should use to determine the analyst agreement with ground truth on a question by question basis

    Functionally, it will perform the rating method on every analyst and for every question.
    Then it will aggregate these ratings across analysts using an aggregation method of your choice.

    Parameters
    -------
    scoresheet_df : pandas.DataFrame
       Dataframe of raw data, in format of one row per subject and per question, and one column per analyst

    questions_to_rate: list of strings
        list of questions to rate

    analysts_to_rate_over: list of strings
        list of analysts to include in the rating calculations

    rating_aggregation_method: lambda
        function to combine individual agreement calculations

    analyst_id_column_name: string
        Name of the Dataframe column containing analysts
        Default: 'analyst'

    subject_id_column_name: string
        Name of the Dataframe column containing subjects
        Default: 'submission'

    Returns
    -------
    output_dict: dict
        Dictionary containing Dataframes describing results for all questions and per question
    '''

    # make sure we're dealing with lists in case the user passed some other iterable
    questions_to_rate = list(questions_to_rate)
    analysts_to_rate_over = list(analysts_to_rate_over)

    # weighted_cohen_kappa, fleiss_kappa, and weighted_fleiss_kappa currently turned off
    supported_rating_methods = ['percent_agreement', 'cohen_kappa']

    # make sure all analysts the user wants to rate over actually exist in the passed dataframe
    if analysts_to_rate_over is None or len(analysts_to_rate_over)<1:
            raise ValueError("To compare questions based on analyst agrement you need to include at least one analyst")
    present_analysts = set(scoresheet_df[analyst_id_column_name])
    for analyst in analysts_to_rate_over:
        if analyst not in present_analysts:
            raise ValueError("Couldn't find any data for analyst "+analyst+". Make sure the input scoresheet dataframe is complete")

    # every analyst in the scoresheet other than the ones specified by the user will be excluded
    analysts_to_exclude = present_analysts - set(analysts_to_rate_over)

    # pivot the scoresheet dataset into what is called a "comparison dataframe" format
    question_comparison_df = _create_analyst_comparison_df(scoresheet_df,
                                                           questions_to_include_in_comparison=questions_to_rate,
                                                           analyst_id_column_name=analyst_id_column_name,
                                                           reference_analysts=[],
                                                           exclude_these_analysts=analysts_to_exclude,
                                                           subject_id_column_name= subject_id_column_name,
                                                           indeterminate_answer_code=9,
                                                           debug=False)

    # compute every supported agreement metric, one at a time
    question_reliability_output = {}

    for rating_method in supported_rating_methods:
        # call on some helper functions to run comparisons on a question and analyst levels
        reliabilities_df = get_separate_reliabilities_for_all_analysts_and_questions(question_comparison_df,
                                                                                     calculation_to_do=rating_method,
                                                                                     questions_to_include=questions_to_rate,
                                                                                     reference_column_name=THE_OTHER_ANALYSTS,
                                                                                     debug=False)
        reliabilities_df = reliabilities_df.sort_values(analyst_id_column_name)
        question_reliabilities = reliabilities_df.groupby('question').apply(rating_aggregation_method)
        question_reliabilities = question_reliabilities.sort_index()
        question_reliability_output[rating_method] = [0.0] * len(questions_to_rate)

        # put the reliability scores into the output data structure
        if rating_method == 'fleiss_kappa':
            for question, reliability in zip(reliabilities_df['question'].values, reliabilities_df['reliability'].values):
                question_reliability_output[rating_method][questions_to_rate.index(question)] = reliability
        else:
            for question, reliability in zip(question_reliabilities.index, question_reliabilities['reliability'].values):
                question_reliability_output[rating_method][questions_to_rate.index(question)] = reliability

    # now let's evaluate the 'coverage' rate per question per analyst
    question_coverage_output = []
    coverage_df = question_comparison_df[[column_name for column_name in question_comparison_df.columns if column_name=='question' or column_name.startswith('determinate_answer_by_analyst_')]].groupby('question').apply(rating_aggregation_method, axis=0).apply(rating_aggregation_method, axis=1)
    for question in questions_to_rate:
        question_overall_coverage = coverage_df[question]
        question_coverage_output.append(question_overall_coverage)

    # format the output into DataFrame to return to caller
    output_df = pd.DataFrame({'question':questions_to_rate, 'coverage':question_coverage_output})
    for rating_method in supported_rating_methods:
        output_df['agreement_using_'+rating_method] = question_reliability_output[rating_method]

    return output_df


def rate_questions_on_analyst_agreement_with_reference(scoresheet_df,
                                                       questions_to_rate,
                                                       analysts_to_rate_over,
                                                       reference,
                                                       rating_aggregation_method=np.mean,
                                                       analyst_id_column_name='analyst',
                                                       subject_id_column_name='submission'):
    '''
    This function is what you should use to determine the analyst inter-agreement on a question by question basis

    Functionally, it will perform the rating method on every analyst and for every question. Then it will
    aggregate these ratings across analysts using an aggregation method of your choice.

    Parameters
    -------
    scoresheet_df : pandas.DataFrame
       Dataframe of raw data, in format of one row per subject and per question, and one column per analyst

    questions_to_rate: list of strings
        list of questions to rate

    analysts_to_rate_over: list of strings
        list of analysts to include in the rating calculations

    reference: string
        reference analyst or instrument to consider as ground truth

    rating_aggregation_method: lambda
        function to combine individual agreement calculations

    analyst_id_column_name: string
        Name of the Dataframe column containing analysts
        Default: 'analyst'

    subject_id_column_name: string
        Name of the Dataframe column containing subjects
        Default: 'submission'

    Returns
    -------
    output_dict: dict
        Dictionary containing Dataframes describing results for all questions and per question
    '''

    # make sure we're dealing with lists in case the user passed some other iterable
    questions_to_rate = list(questions_to_rate)
    analysts_to_rate_over = list(analysts_to_rate_over)

    # weighted cohen kappa, fleiss_kappa, and weighted fleiss kappa currently turned off
    supported_rating_methods = ['percent_agreement', 'cohen_kappa']

    # make sure all analysts the user wants to rate over actually exist in the passed dataframe
    if analysts_to_rate_over is None or len(analysts_to_rate_over)<1:
            raise ValueError("To compare questions based on analyst agrement you need to include at least one analyst")

    present_analysts = set(scoresheet_df[analyst_id_column_name])
    for analyst in analysts_to_rate_over:
        if analyst not in present_analysts:
            raise ValueError("Couldn't find any data for analyst "+analyst+". Make sure the input scoresheet dataframe is complete")

    # make sure the reference analyst exists in the passed dataframe
    if reference is None:
            raise ValueError("To compare analyst to reference you need to specify a valid analyst or instrument as reference")

    if reference not in present_analysts:
            raise ValueError("Couldn't find any data for reference "+reference+". Make sure the input scoresheet dataframe is complete")

    # an analyst cannot be chosen as both a reference and as a rated analyst
    if reference in analysts_to_rate_over:
            raise ValueError("Cannot rate analyst "+reference+" while also using him/her as a reference")

    # every analyst in the scoresheet other than the ones specified by the user will be excluded
    analysts_to_exclude = present_analysts - set(analysts_to_rate_over+[reference])

    # pivot the scoresheet dataset into what is called a "comparison dataframe" format
    question_comparison_df = _create_analyst_comparison_df(scoresheet_df,
                                                                         questions_to_include_in_comparison=questions_to_rate,
                                                                         analyst_id_column_name=analyst_id_column_name,
                                                                         reference_analysts=[reference],
                                                                         exclude_these_analysts=analysts_to_exclude,
                                                                         subject_id_column_name=subject_id_column_name,
                                                                         indeterminate_answer_code=9,
                                                                         debug=False)

    # compute every supported agreement metric, one at a time
    question_reliability_output = {}

    for rating_method in supported_rating_methods:
        # call on some helper functions to run comparisons on a question and analyst levels
        reliabilities_df = get_separate_reliabilities_for_all_analysts_and_questions(question_comparison_df,
                                                                                     calculation_to_do=rating_method,
                                                                                     questions_to_include=questions_to_rate,
                                                                                     reference_column_name=reference,
                                                                                     debug=False)
        reliabilities_df = reliabilities_df.sort_values(analyst_id_column_name)
        question_reliabilities = reliabilities_df.groupby('question').apply(rating_aggregation_method)
        question_reliabilities = question_reliabilities.sort_index()
        question_reliability_output[rating_method] = [0.0] * len(questions_to_rate)

        # put the reliability scores into the output data structure
        if rating_method == 'fleiss_kappa':
            for question, reliability in zip(reliabilities_df['question'].values, reliabilities_df['reliability'].values):
                question_reliability_output[rating_method][questions_to_rate.index(question)] = reliability
        else:
            for question, reliability in zip(question_reliabilities.index, question_reliabilities['reliability'].values):
                question_reliability_output[rating_method][questions_to_rate.index(question)] = reliability

    # now let's evaluate the 'coverage' rate per question per analyst
    question_coverage_output = []
    column_names = [column_name for column_name in question_comparison_df.columns if column_name=='question' or column_name.startswith('determinate_answer_by_analyst_')]

    coverage_df = question_comparison_df[column_names] \
      .groupby('question') \
      .apply(rating_aggregation_method, axis=0) \
      .apply(rating_aggregation_method, axis=1)

    for question in questions_to_rate:
        question_overall_coverage = coverage_df[question]
        question_coverage_output.append(question_overall_coverage)

    # format the output into DataFrame to return to caller
    output_df = pd.DataFrame({'question':questions_to_rate, 'coverage':question_coverage_output})
    for rating_method in supported_rating_methods:
        output_df['agreement_with_'+reference+'_using_'+rating_method] = question_reliability_output[rating_method]

    return output_df
