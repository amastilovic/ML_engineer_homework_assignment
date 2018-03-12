import copy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import ds_helper_functions


def get_inconclusive_model(reliability_column,
                           reliability_cutoff,
                           dataset,
                           feature_columns,
                           feature_encoding_map,
                           target_column,
                           model_function,
                           sample_weights,
                           verbose=False,
                           **model_parameters):

    # These priors are only used for 'reallife' metric calculations,
    # but does not make sense for reliability model anyway, so these are just dummy values
    reliability_outcome_class_priors = [(1.0/2.0), (1.0/2.0)]
    reliability_dunno_range = None
    reliability_outcome_classes = ['reliable', 'not']

    # Separate fit to determine reliability
    reliability_model, reliability_features, reliability_y_predicted_without_dunno, reliability_y_predicted_with_dunno, reliability_y_predicted_probs =\
        ds_helper_functions.all_data_model(dataset, feature_columns, feature_encoding_map, target_column, sample_weights, reliability_dunno_range, model_function, **model_parameters)

    dataset['direct_predicted_reliable'] = reliability_y_predicted_without_dunno
    dataset['predicted_reliable'] = ['reliable' if ele[1]>reliability_cutoff else 'not' for ele in reliability_y_predicted_probs]
    dataset['reliability_score'] = [ele[1] for ele in reliability_y_predicted_probs]

    if verbose:
        print 'n_predicted_reliabile: ', len([ele for ele in dataset['predicted_reliable'].values if ele=='reliable'])
        print 'n_predicted_not: ', len([ele for ele in dataset['predicted_reliable'].values if ele=='not'])
        print 'n_Xvalid_correct: ', len([ele for ele in dataset[reliability_column].values if ele=='reliable'])
        print 'n_Xvalid_incorrect: ', len([ele for ele in dataset[reliability_column].values if ele=='not'])

        print 'get reliability metrics'
        reliability_metrics = ds_helper_functions.get_classifier_performance_metrics(reliability_outcome_classes, reliability_outcome_class_priors,
                                dataset[target_column], dataset['predicted_reliable'].values,
                                reliability_y_predicted_with_dunno, reliability_y_predicted_probs
                                                            )
        print 'reliability_metrics: ', reliability_metrics
        print 'formatted reliability metrics are: '
        ds_helper_functions.print_classifier_performance_metrics(reliability_outcome_classes, reliability_metrics)

    ### Now try training/applying with only the reliable ones
    reliable_dataset = dataset[dataset['predicted_reliable']=='reliable']


    reliable_dataset['reliable_sample_weights'] = [weight for reliable, weight in zip(dataset['predicted_reliable'].values, sample_weights) if reliable=='reliable']
    return reliability_model, reliable_dataset, reliability_features


def run_inconclusive_model(model_reliability_structure, model_reliable_only_structure, input_data):
    ''' Helper function to apply the double reliability model '''
    reliability_X,reliability_y = model_reliability_structure['data_prep_function'](input_data, model_reliability_structure)
    for col in model_reliability_structure['features']:
        if col not in reliability_X.columns:
            reliability_X[col] = np.zeros(len(reliability_X.index))
    reliability_output = model_reliability_structure['apply_function'](reliability_X, reliability_y, model_reliability_structure)
    reliability_response = reliability_output['model_response']

    reliable_only_X, reliable_only_y = model_reliable_only_structure['data_prep_function'](input_data, model_reliable_only_structure)
    reliable_only_output = model_reliable_only_structure['apply_function'](reliable_only_X, reliable_only_y, model_reliable_only_structure)
    return reliability_output, reliable_only_output


def run_XValidation_many_times_and_find_agreeing_points(n_expts,
                                                        dataset,
                                                        sample_weights,
                                                        feature_columns,
                                                        feature_encoding_map,
                                                        outcome_column,
                                                        dunno_range,
                                                        n_folds,
                                                        outcome_classes,
                                                        outcome_class_priors,
                                                        model_function,
                                                        model_parameters,
                                                        verbose=False):
    ''' Run n_expts X-validations and do a majority vote of which points end up being in agreement '''
    #verbose = True
    n_matching_arr = None
    for iexpt in range(n_expts):
        if verbose:
            print 'In XValidation matching fn, start iexpt: ', iexpt, ' out of ', n_expts
            print 'outcome_classes: ', outcome_classes
            print 'Number in each class: '
            for this_class in outcome_classes:
                print 'class: ', this_class, ', n children: ', len(dataset[dataset[outcome_column]==this_class].index)
        output = ds_helper_functions.cross_validate_model(dataset, sample_weights, feature_columns, feature_encoding_map, outcome_column, dunno_range,
                       n_folds, outcome_classes, outcome_class_priors, cp.deepcopy(model_function), **model_parameters)
        if verbose:
            print 'Finished XValidate model'
        XValid_good = dataset.index.isin(output['correctly_predicted_sample_indices']).astype(int)
        if iexpt == 0:
            n_matching_arr = XValid_good
        else:
            n_matching_arr += XValid_good
    if verbose:
        print 'In XValidation matching fn, n_matching_arr final is ', list(n_matching_arr), ', out of ', n_expts, ' total pseudoxperiments'
    n_matching_cutoff = int(n_expts / 2.)
    is_reliable_arr = ['reliable' if ele>n_matching_cutoff else 'not' for ele in n_matching_arr]
    if verbose:
        print 'is_reliable_arr: ', is_reliable_arr
    return is_reliable_arr


def run_inconclusive_model_pseudo_experiments(n_expts,
                                              n_XValid_expts,
                                              frac_holdout,
                                              reliability_cutoff,
                                              dataset,
                                              sample_weights,
                                              feature_columns,
                                              feature_encoding_map,
                                              outcome_column,
                                              dunno_range,
                                              n_folds,
                                              outcome_classes,
                                              outcome_class_priors,
                                              base_model,
                                              base_model_args_dict,
                                              base_reliability_model,
                                              reliability_model_args_dict,
                                              verbose=False):
    '''
    Run pseudo experiments to determine benefit of reliability model.
    Inputs:
        n_expts: Number of pseudo-experiments to run and average results
        n_XValid_expts: Number of pseudo-experiments to run inside of each pseudo-experiment
           when doing X-validation majority voting on ok points
        frac_holdout: in each pseudoexperiment hold out this fraction of random
           events to validate performance
        reliability_cutoff: in held-out events, this is the reliability model score
           cutoff that is required in order to consider the event ok to use
        dataset: the dataframe with all training data, before any splitting

        In each experiment will randomly hold out frac_holdout events for validation,
        and will only consider those which are modeled to be deterministic at better than
        reliability_cutoff score.

        samples_weights, feature_columns, feature_encoding_map, outcome_column,
        dunno_range, n_folds, outcome_classes, outcome_class_priors are characteristics
        of the classification model

        The classification model is defined by base_model with parameters from base_model_args_dict

        The reliability model is defined from the reliability_model and reliability_model_args_dict

    Outputs:
        A dataframe with the results of each pseudoexperiment
    '''

    experiment_results = []
    reliability_outcome_column = 'XValid_matches'
    for iexpt in range(n_expts):
        print 'On experiment ', iexpt

        ### Select subset for reliability training
        idxs_for_reliability_training = random.sample(dataset.index, int(len(dataset.index) * frac_holdout))
        reliability_df = dataset.iloc[idxs_for_reliability_training].reset_index(drop=True)
        reliability_sample_weights = sample_weights.iloc[idxs_for_reliability_training].reset_index(drop=True)

        print 'On experiment, ', iexpt, ', run_XValidation many times'
        reliability_df['XValid_matches'] = run_XValidation_many_times_and_find_agreeing_points(n_expts=n_XValid_expts, dataset=reliability_df,
              sample_weights=reliability_sample_weights, feature_columns=feature_columns, feature_encoding_map=feature_encoding_map,
              outcome_column=outcome_column, dunno_range=dunno_range, n_folds=n_folds, outcome_classes=outcome_classes,
              outcome_class_priors=outcome_class_priors, model_function=cp.deepcopy(base_model), model_parameters=base_model_args_dict,
              verbose=False)

        print 'On experiment, ', iexpt, ', get reliability model'
        reliability_model, reliable_dataset, encoded_reliability_features = get_inconclusive_model(reliability_column='XValid_matches',
                reliability_cutoff=reliability_cutoff, dataset=reliability_df, feature_columns=feature_columns,
                feature_encoding_map=feature_encoding_map, target_column=reliability_outcome_column,
                model_function=cp.deepcopy(base_reliability_model), sample_weights=reliability_sample_weights,
                **reliability_model_args_dict)
        n_predicted_reliable = len([ele for ele in reliability_df['predicted_reliable'].values if ele=='reliable'])
        n_predicted_not = len([ele for ele in reliability_df['predicted_reliable'].values if ele=='not'])
        n_Xvalid_correct = len([ele for ele in reliability_df['XValid_matches'].values if ele=='reliable'])
        n_Xvalid_incorrect = len([ele for ele in reliability_df['XValid_matches'].values if ele=='not'])
        reliability_coverage = float(n_predicted_reliable) / float(n_predicted_reliable + n_predicted_not)
        if verbose:
            print 'n_predicted_reliable: ', n_predicted_reliable
            print 'n_predicted_not: ', n_predicted_not
            print 'n_Xvalid_correct: ', n_Xvalid_correct
            print 'n_Xvalid_incorrect: ', n_Xvalid_incorrect

        ### Now compare with validation results
        print 'On experiment, ', iexpt, ', get validation df'
        validation_df = dataset[~dataset.index.isin(idxs_for_reliability_training)].reset_index(drop=True)
        validation_sample_weights = sample_weights[~sample_weights.index.isin(idxs_for_reliability_training)].reset_index(drop=True)

        ### old way of doing things:
        validation_output = ds_helper_functions.cross_validate_model(validation_df, validation_sample_weights, feature_columns,
                                feature_encoding_map, outcome_column, dunno_range, n_folds, outcome_classes,
                                outcome_class_priors, cp.deepcopy(base_model),  **base_model_args_dict)
        ### restrict to only the good stuff:
        X_validate,y_validate,encoded_features_validate = ds_helper_functions.prepare_data_for_modeling(validation_df,
                                    feature_columns, feature_encoding_map, 'outcome', force_encoded_features=encoded_reliability_features)


        validation_df['reliability_prob'] = [ele[1] for ele in reliability_model.predict_proba(X_validate)]
        validation_df['reliability_prediction'] = ['reliable' if ele>reliability_cutoff else 'not' for ele in validation_df['reliability_prob'].values]

        restricted_validation_df = validation_df[validation_df['reliability_prediction']=='reliable']
        restricted_validation_sample_weights = validation_sample_weights[validation_sample_weights.index.isin(restricted_validation_df.index)]
        restricted_validation_output = ds_helper_functions.cross_validate_model(restricted_validation_df, restricted_validation_sample_weights,
                                feature_columns, feature_encoding_map, outcome_column, None, n_folds,
                                outcome_classes, outcome_class_priors, cp.deepcopy(base_model), **base_model_args_dict)

        if verbose:
            print 'For expt ', iexpt, ', X-validation performance with no reliability restriction is:'
            print_classifier_performance_metrics(outcome_classes, validation_output['overall_metrics'])
            print 'For expt ', iexpt, ', X-validation performance with reliability restriction is:'
            print_classifier_performance_metrics(outcome_classes, restricted_validation_output['overall_metrics'])

        if iexpt == 0:
            print 'validation output: ', validation_output['overall_metrics']
            confusion_matrix = validation_output['overall_metrics']['with_dunno']['dataset_confusion']
            print 'Dunno confusion matrix: ', confusion_matrix
            print 'Value at 0, 2: ', confusion_matrix[0,2]

        confusion_matrix_summed_over_rows = np.sum(confusion_matrix, axis=0)
        n_total = np.sum(confusion_matrix_summed_over_rows)
        n_dunno = confusion_matrix_summed_over_rows[2]
        dunno_coverage = float(n_total - n_dunno) / float(n_total)

        print 'n_dunno: ', n_dunno
        print 'n_total: ', n_total
        print 'dunno_coverage: ', dunno_coverage
        print 'chained coverage: ', reliability_coverage
        print 'outcome_classes: ', outcome_classes
        print 'construct results dictionary'

        results_dict = {}
        results_dict['dunno_min'] = dunno_range[0]
        results_dict['dunno_max'] = dunno_range[1]
        results_dict['dunno_coverage'] = dunno_coverage
        results_dict['chained_coverage'] = reliability_coverage

        for outcome_class in outcome_classes:
            results_dict['default_'+outcome_class+'_recall'] = validation_output['overall_metrics']['without_dunno']['dataset_recall_per_class'][outcome_class]
            results_dict['chained_'+outcome_class+'_recall'] = restricted_validation_output['overall_metrics']['without_dunno']['dataset_recall_per_class'][outcome_class]
            results_dict['dunno_'+outcome_class+'_recall'] = validation_output['overall_metrics']['with_dunno']['dataset_recall_per_class'][outcome_class]

        experiment_results.append(results_dict)

    experiment_results_df = pd.DataFrame(experiment_results)

    return experiment_results_df


def visualize_experiment_results(experiment_results_df, title, override_columns_to_visualize=None):
    print 'visualize experiment results for ', title
    mean_vals = experiment_results_df.mean(axis=0)
    print 'mean_vals: ', mean_vals

    if override_columns_to_visualize is None:
        ordered_cols = ['default_autism_recall', 'dunno_autism_recall', 'chained_autism_recall', 'default_not_recall', 'dunno_not_recall', 'chained_not_recall', 'dunno_coverage', 'chained_coverage']
    else:
        ordered_cols = override_columns_to_visualize
    ordered_mean_vals = mean_vals[ordered_cols]
    print 'ordered_mean_vals: ', ordered_mean_vals

    plt.figure(figsize=(12,8))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    xVals = np.arange(len(ordered_mean_vals.index))
    plt.bar(xVals, ordered_mean_vals.values)
    plt.gca().set_xticklabels(ordered_cols, rotation=70, fontsize=20)
    plt.gca().set_ylim([0, 1.2])
    plt.ylabel('Performance value', fontsize=20)
    plt.title(title, fontsize=20)
    plt.show()
