import math
import random
import marshal, pickle, types
import copy as cp
import collections
import itertools
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt

from cognoa_ds_lib.constants import *

# set up logging
DEFAULT_LOG_LEVEL = logging.INFO # or logging.DEBUG or logging.WARNING or logging.ERROR

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.setLevel(DEFAULT_LOG_LEVEL)


def save_model(model,
               class_names,
               dunno_range,
               features,
               feature_columns,
               feature_encoding_map,
               outcome_column,
               data_prep_function,
               apply_function,
               filename):

    data_prep_function_string = marshal.dumps(data_prep_function.func_code)
    apply_function_string = marshal.dumps(apply_function.func_code)

    pickle.dump({
        'model':model,
        'class_names': class_names,
        'dunno_range': dunno_range,
        'features': features,
        'feature_columns': feature_columns,
        'feature_encoding_map': feature_encoding_map,
        'target': outcome_column,
        'data_prep_function': data_prep_function_string,
        'apply_function': apply_function_string
    }, open(filename, "wb"))


def load_model(filename):
    model_structure = pickle.load(open(filename, "rb"))
    data_prep_function_key = 'data_prep_function'
    apply_function_key = 'apply_function'

    model_structure[data_prep_function_key] = types.FunctionType(marshal.loads(model_structure[data_prep_function_key]),
                                                                 globals(),
                                                                 data_prep_function_key)

    model_structure[apply_function_key] = types.FunctionType(marshal.loads(model_structure[apply_function_key]),
                                                             globals(),
                                                             apply_function_key)

    return model_structure


def print_classifier_performance_metrics(class_names, metrics):
    '''
    Pretty printing of outputs that you got by calling get_multi_classifier_performance_metrics
    '''

    print("\n\nDATASET PROFILE:\n\n")

    print("samples\t\t"+str(metrics['number_samples']))
    for class_name in class_names:
        print(class_name+
        " samples\t"+
        str(metrics['samples_per_class'][class_name])+
        "\t["+
        str(100.0*float(metrics['samples_per_class'][class_name]) / float(metrics['number_samples']))
        +"%]"
        )

    print("\n\nPERFORMANCE BEFORE DUNNO LOGIC:\n\n")

    print("AUC\t\t"+str(metrics[KEY_WITHOUT_DUNNO]['auc']))
    print("accuracy [Dataset]\t\t"+str(metrics[KEY_WITHOUT_DUNNO]['dataset_accuracy']))
    print("accuracy [Reallife]\t\t"+str(metrics[KEY_WITHOUT_DUNNO]['reallife_accuracy']))

    print("\n")
    for class_name in class_names:
        print(class_name+" precision [Dataset]\t"+str(metrics[KEY_WITHOUT_DUNNO]['dataset_precision_per_class'][class_name]))
        print(class_name+" precision [Reallife]\t"+str(metrics[KEY_WITHOUT_DUNNO]['reallife_precision_per_class'][class_name]))
        print(class_name+" recall\t"+str(metrics[KEY_WITHOUT_DUNNO]['dataset_recall_per_class'][class_name]))
    print("\n")
    print("Confusion Matrix:")
    print(metrics[KEY_WITHOUT_DUNNO]['dataset_confusion'])

    print("\n\nPERFORMANCE INCLUDING DUNNO LOGIC:\n\n")

    print("classification rate [Dataset]\t"+str(metrics[KEY_WITH_DUNNO]['dataset_classification_rate']))
    print("classification rate [Reallife]\t"+str(metrics[KEY_WITH_DUNNO]['reallife_classification_rate']))
    print("accuracy where classified [Dataset]\t"+str(metrics[KEY_WITH_DUNNO]['dataset_accuracy_where_classified']))
    print("accuracy where classified [Reallife]\t"+str(metrics[KEY_WITH_DUNNO]['reallife_accuracy_where_classified']))

    print("\n")
    for class_name in class_names:
        print(class_name+" precision [Dataset]\t"+str(metrics[KEY_WITH_DUNNO]['dataset_precision_per_class'][class_name]))
        print(class_name+" precision [Reallife]\t"+str(metrics[KEY_WITH_DUNNO]['reallife_precision_per_class'][class_name]))
        print(class_name+" recall\t"+str(metrics[KEY_WITH_DUNNO]['dataset_recall_per_class'][class_name]))
    print("\n")
    for class_name in class_names:
        print(class_name+" precision where classified [Dataset]\t"+str(metrics[KEY_WITH_DUNNO]['dataset_precision_per_class_where_classified'][class_name]))
        print(class_name+" precision where classified [Reallife]\t"+str(metrics[KEY_WITH_DUNNO]['reallife_precision_per_class_where_classified'][class_name]))
        print(class_name+" recall where classified\t"+str(metrics[KEY_WITH_DUNNO]['dataset_recall_per_class_where_classified'][class_name]))
    print("\n")
    print("Confusion Matrix:")
    print(metrics[KEY_WITH_DUNNO]['dataset_confusion'])


# def draw_sanity_overlays(results_dict, feature_columns, presence_means, suspicious_features, title, ylabel, ylims, draw_comp_line=None):
#     plt.figure(figsize=(12,8))
#     plt.grid(True)
#     colors = ['red', 'blue', 'black', 'purple']
#     base_XVals = np.arange(len(feature_columns))+0.5
#     plot_num=0
#     n_plots = len(results_dict.keys())
#     for (leg_label, yVals), color in zip(results_dict.iteritems(), colors):
#         xWidths = 1. / (n_plots + 2.)
#         these_xVals = np.arange(len(feature_columns))+(float(plot_num)*xWidths)
#         plt.bar(these_xVals, yVals, xWidths, color=color, label=leg_label)
#         plot_num+=1

#     if ylims is None:
#         cur_ylims = plt.gca().get_ylim()
#         ylim_range = cur_ylims[1] - cur_ylims[0]
#         ylims = [0, cur_ylims[0]+(ylim_range*1.2)]
#     plt.gca().set_ylim(ylims)
#     if draw_comp_line is not None:
#         xlims = plt.gca().get_xlim()
#         plt.plot(xlims, [draw_comp_line]*2, color='red', linestyle='--', linewidth=2)
#     plt.legend(fontsize=24)
#     plt.xticks(base_XVals, feature_columns, rotation=70, fontsize=18)

#     autism_features = []
#     not_features = []
#     for feature, presence_type in zip(feature_columns, presence_means):
#         if presence_type=='not':
#             not_features.append(feature)
#         elif presence_type=='autism':
#             autism_features.append(feature)
#         else:
#             assert 0

#     logger.info("suspicious_features: %s" % str(suspicious_features))

#     for xtick_label in plt.gca().get_xticklabels():
#         if xtick_label.get_text() in not_features:
#             xtick_label.set_color('red')
#         if xtick_label.get_text() in suspicious_features:
#             logger.info("Set label %s to bold" % xtick_label.get_text())
#             xtick_label.set_weight('bold')

#     plt.ylabel(ylabel, fontsize=20)
#     plt.title(title, fontsize=22)
#     plt.show()
