import collections
import numpy as np
import pandas as pd
import copy as cp
import sys
import logging

from ADOS_mappings import *


DEFAULT_LOG_LEVEL = logging.INFO # or logging.DEBUG or logging.WARNING or logging.ERROR

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.setLevel(DEFAULT_LOG_LEVEL)


def apply_response_transforms(row, response_transform_rules, transform_key, id_key):
    '''
    Runs on a single row, intended to be part of a pandas apply function.
    Transforms responses where needed, but not question itself. response_transform_rules defines
    the transformation that should be applied for each value. If no  transform is defined for a given
    value then returns original value unchanged.

    response_transform_rules should be a dictionary of mappings like this: {'1': '2', '2': ['2', '3']}
    If the mapping is to a list rather than a single value then a deterministic choice is made
    based upon a modulus of the device id, specified by id_key '''

    if row[transform_key] in response_transform_rules.keys():
        transform_instructions = response_transform_rules[row[transform_key]]
        if isinstance(transform_instructions, (list, np.ndarray)):
            ### Get index of output to map to based on last digit in ID if ID is string, or based on % len(list) if int
            number_of_choices = len(transform_instructions)
            try:   ### if ID is string type
                if row[id_key] == '':
                    last_digit_of_id = 'a'  #### just pik something
                else:
                    last_digit_of_id = row[id_key][-1]

                assert isinstance(last_digit_of_id, (str, unicode))
                assert len(last_digit_of_id) == 1
                idx_chosen = ord(last_digit_of_id) % number_of_choices
            except:
                idx_chosen = row[id_key] % number_of_choices

            return transform_instructions[idx_chosen]
        else:
            return response_transform_rules[row[transform_key]]
    else:   ### No instructions for a transformation of this key. Return original.
        return row[transform_key]


def map_df_between_ADOS_versions(in_df, id_col, orig_version='v2', new_version='v1', key_suffix=''):
    ''' Convert between ADOS formats. Can go either v1 => v2, or v2 => v1.
    ... ID column gives user IDs of each row, required for deterministic mapping if multiple plausible answers exist
    ... Returns dataframe with converted columns
    ... Any extra/non ADOS rows are passed through without touching them
    ...... based on 'responses_differ' key in ADOS_v1_vs_v2_mapping_df

    Note: if you are not using standard naming and have a common suffix (such as '_thompson'),
    specify that in the key_suffix variable
    '''

    assert orig_version in ['v1', 'v2']
    assert new_version in ['v1', 'v2']
    conversion_df = cp.deepcopy(ADOS_v1_vs_v2_mapping_df)
    if key_suffix != '':
        conversion_df[KEY_V1] = [None if ele is None else ele+key_suffix for ele in conversion_df[KEY_V1].values]
        conversion_df[KEY_V2] = [None if ele is None else ele+key_suffix for ele in conversion_df[KEY_V2].values]

    orig_key = orig_version + '_key'
    new_key = new_version + '_key'
    out_df = cp.deepcopy(in_df)
    in_cols = cp.deepcopy(list(out_df.columns))
    out_cols = []
    cols_to_drop = []
    for in_col in in_cols:
        out_col = in_col   ### unless we replace it below
        responses_differ = False
        matching_row_idxs = conversion_df[conversion_df[orig_key]==in_col].index.tolist()
        if len(matching_row_idxs) == 0:
            logging.info("No match found for question %s. Do not convert." % str(in_col))
        elif len(matching_row_idxs) == 1:
            idx = matching_row_idxs[0]
            conversion_dict = conversion_df.ix[idx].to_dict()
            matching_col = conversion_dict[new_key]
            if matching_col is None or pd.isnull(matching_col):
                logging.info("No corresponding column for %s. Drop column." % str(in_col))
                cols_to_drop.append(out_col)
            else:   ### ok to replace
                out_col = matching_col
                if in_col != out_col:
                    logger.info("Convert %s to %s" % (str(in_col), str(out_col)))
            response_transform_key = None

            if orig_version=='v1' and new_version == 'v2': response_transform_key = KEY_V1_TO_V2
            if orig_version=='v2' and new_version == 'v1': response_transform_key = KEY_V2_TO_V1
            response_transform_rules = conversion_dict.get(response_transform_key, None)
            if response_transform_rules is not None and isinstance(response_transform_rules, dict):
                out_df[in_col] = out_df[[in_col, id_col]].apply(apply_response_transforms, args=(response_transform_rules,in_col, id_col), axis=1)
        else:
            logger.error("in_col: %s, matching_row_idxs: %s" % (str(in_col), str(matching_row_idxs)))
            logger.error("df: %s" % str(out_df))
            logger.error("Error, could not understand matching for in_col: %s" % str(in_col))
            raise ValueError

        out_cols.append(out_col)

    assert len(out_cols) == len(in_cols)

    out_df.columns = out_cols
    if cols_to_drop != []:
        out_df = out_df.drop(cols_to_drop, axis=1)

    return out_df


def map_between_ADOS_versions_old(in_df, use_strict=False, orig_version='v2', new_version='v1', key_suffix=''):
    '''
    Convert between ADOS formats. Can go either v1 => v2, or v2 => v1.
    ... Returns dataframe with converted columns
    ... Any extra/non ADOS rows are passed through without touching them
    ... If use_strict is True then will drop any columns that are flagged as suspicious
    ...... based on 'responses_differ' key in ADOS_v1_vs_v2_mapping_df

    Note: if you are not using standard naming and have a common suffix (such as '_thompson'),
    specify that in the key_suffix variable
    '''

    assert orig_version in ['v1', 'v2']
    assert new_version in ['v1', 'v2']

    conversion_df = cp.deepcopy(ADOS_v1_vs_v2_mapping_df)

    if key_suffix != '':
        logger.info("v1 values: %s" % str(list(conversion_df[KEY_V1].values)))
        logger.info("v2 values: %s" % str(list(conversion_df[KEY_V2].values)))
        logger.info("key_suffix: %s" % str(key_suffix))

        conversion_df[KEY_V1] = [None if ele is None else ele+key_suffix for ele in conversion_df[KEY_V1].values]
        conversion_df[KEY_V2] = [None if ele is None else ele+key_suffix for ele in conversion_df[KEY_V2].values]

    orig_key = orig_version + '_key'
    new_key = new_version + '_key'
    out_df = cp.deepcopy(in_df)
    in_cols = cp.deepcopy(list(out_df.columns))
    out_cols = []
    cols_to_drop = []
    for in_col in in_cols:
        out_col = in_col   ### unless we replace it below
        responses_differ = False
        matching_row_idxs = conversion_df[conversion_df[orig_key]==in_col].index.tolist()
        if len(matching_row_idxs) == 0:
            logging.info("No match found for question %s. Do not convert." % str(in_col))
        elif len(matching_row_idxs) == 1:
            idx = matching_row_idxs[0]
            matching_col = conversion_df.iloc[idx][new_key]
            if matching_col is None or pd.isnull(matching_col):
                logging.info("No corresponding column for %s. Drop column." % str(in_col))
                cols_to_drop.append(out_col)
            else:   ### ok to replace
                out_col = matching_col
                responses_differ = conversion_df.iloc[idx]['responses_differ']
                if use_strict and responses_differ:
                    logger.info("running in mode use_strict and matching between %s and %s is suspicious. Drop column." % (str(in_col), str(out_col)))
                    cols_to_drop.append(out_col)
                else:
                    if in_col != out_col:
                        logger.info("Convert %s to %s" % (str(in_col), str(out_col)))
        else:
            logger.error("in_col: %s, matching_row_idxs: %s" % (str(in_col), str(matching_row_idxs)))
            logger.error("df: %s" % str(out_df))
            logger.error("Error, could not understand matching for in_col: %s" % str(in_col))
            raise ValueError

        out_cols.append(out_col)

    assert len(out_cols) == len(in_cols)

    out_df.columns = out_cols
    if cols_to_drop != []:
        out_df = out_df.drop(cols_to_drop, axis=1)

    return out_df


def sanity_check_responses(in_df, evaluator):
    '''
    Inputs:
            in_df: dataframe of the data you want to sanity check
            evaluator: 'ADOS1' or 'ADOS2' (add in adirs later)
        methodology:
            for each key in in_df that is part of the evaluator, checks for
            presence or absence of expected responses
        returns:
            missing_responses_dict: an ordered dictionary of questions and valid answers
               that do not appear in the data
               .... Note that this may not be a problem if statistics are low and
               some responses simply were not given
            invalid_responses_dict: an ordered dictionary of questions and responses
               which were never defined (such as an evaluator giving a 3 when only 0, 1, and 2
               have a meaning
            questions_not_asked: a list of questions from the evaluator that are not in the data
            questions_that_may_be_typos: a list of keys in in_df that look like they may have been
               intended to be valid questions but which do not match
    '''

    def convert_response(response):
        ''' If response is in bad format, convert it '''
        try:
            out_response = int(response)
        except:
            out_response = response
        return out_response

    missing_responses_dict = collections.OrderedDict()
    invalid_responses_dict = collections.OrderedDict()

    expected_qs_and_as = VALID_Qs_AND_As[evaluator]
    questions_that_may_be_typos = []

    for column in in_df.columns:
        if column not in expected_qs_and_as.keys():
            continue
        observed_responses = np.unique(in_df[column].values)
        observed_responses_converted = [convert_response(ele) for ele in observed_responses]
        invalid_responses_dict[column] = [ele for ele in observed_responses_converted if ele not in expected_qs_and_as[column]]
        missing_responses_dict[column] = [ele for ele in expected_qs_and_as[column] if ele not in observed_responses_converted]

    questions_not_asked = [question for question in expected_qs_and_as.keys() if question not in in_df.columns]
    questions_that_may_be_typos = [col for col in in_df.columns if evaluator.lower() in col and col not in expected_qs_and_as.keys()]

    return missing_responses_dict, invalid_responses_dict, questions_not_asked, questions_that_may_be_typos


def cross_check_ADOS_consistent_with_new_version(in_df):
    '''
    Quick and dirty check to see whether in_df is more consistent with
    newer or older version

    Newer version has a number of responses that are not in the earlier version,
    and ados2_b12 is a new question. Check to see if these responses and this question are
    present

    Returns boolean true if consistent with new version
    '''

    seems_like_new_version = False

    if 'ados2_b12' in in_df.columns:
        logger.info("ados2_b12 is present in data columns. This is consistent with the newer version.")
        seems_like_new_version = True

    def convert_response(response):
        try:
            out_response = int(response)
        except:
            out_response = response
        return out_response

    question_responses_only_in_new_version = collections.OrderedDict([
            ('ados1_a1', 4),
            ('ados1_b5', 3),
            ('ados1_d1', 3),
            ('ados1_d2', 3),
            ('ados1_e1', 3),
            ('ados1_e2', 3),
            ('ados2_d1', 3),
            ('ados2_d2', 3),
            ('ados2_e3', 3),
        ])

    for question, response in question_responses_only_in_new_version.iteritems():
        value_counts = in_df[question].value_counts()
        if response in [convert_response(ele) for ele in list(value_counts.index)]:
            seems_like_new_version = True

    return seems_like_new_version
