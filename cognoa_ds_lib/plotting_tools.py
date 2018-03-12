import matplotlib

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

### Importing matplotlib both ways because there are a mix of functions below that use different interfaces.
### Would be good to clean this up at some point.
import numpy as np
import pandas as pd


def _get_list_of_unique_x_values(list_of_x_data_lists):
    '''
    Helper function for functions below. Intended for overlaying bar charts where
    a list of lists or arrays represent the x-values of each dataset that will be overlayed.
    This functions determines the list of the x-values that should be displayed on the
    x-axis
    '''

    list_of_unique_lists = [np.unique(dataset) for dataset in list_of_x_data_lists]
    combined_list_with_duplicates = [item for sublist in list_of_unique_lists for item in sublist]
    list_of_unique_x_values = np.unique(combined_list_with_duplicates)
    return list_of_unique_x_values


def overlay_bar_charts_from_numeric_arrays(list_of_x_data_lists, legend_label_list, plot_options_dict):
    '''
    Intended for situations where there is a small number of often repeated values that
    most of the data might have and you want to compare distributions without them obscuring
    one another

    x_values_list: a list of data lists or arrays
    '''
    list_of_unique_x_values = _get_list_of_unique_x_values(list_of_x_data_lists)
    list_of_x_data_bars = []
    list_of_y_data_bars = []
    for dataset in list_of_x_data_lists:
        x_data_bars = []
        y_data_bars = []
        for x_value in list_of_unique_x_values:
            y_value = len(dataset[dataset==x_value])
            x_data_bars.append(x_value)
            y_data_bars.append(y_value)
        list_of_x_data_bars.append(np.array(x_data_bars))
        list_of_y_data_bars.append(np.array(y_data_bars))

    overlay_bar_charts(list_of_x_data_bars, list_of_y_data_bars, legend_label_list, x_values_are_categorical=False, plot_options_dict=plot_options_dict)


def overlay_bar_charts(list_of_x_data_bars, list_of_y_data_bars, legend_label_list=[''], x_values_are_categorical=True, plot_options_dict=None):
    '''
    Overlay some number of bar charts with values (or categories) list_of_x_data_bars, and y_values list_of_y_data_bars
    ... This can run on categorical or numeric data. If numeric x-axis becomes the value of the numbers, and the bars will
    probably not be equally spaced. The chart can easily get overwhelmed with many bins. This function is mostly useful if only
    a small number of often repeated numeric values are present in the data.
    ... If your x data is categorical then you should have it remapped to equally spaced bars: set x_values_are_categorical to True
    ... The overlaying of many different data bars is accomplished by injecting coordinated offsets into the x-values of different elements
    of the x lists so that they can be seen side-by-side. When interpreting the results (especially if the x-data is numerical rather than
    categorical) it should be remembered that this offset is a necessary artifact of plotting in an understandable way and not indicative of
    a true numerical offset in the data.
    '''

    def get_bar_width(n_bars, x_range):
        bar_density = float(n_bars) / float(x_range[1] - x_range[0])
        bar_width = 0.5 / bar_density   ### want roughly this percent of visual screen to be taken up by bars
        return bar_width

    if 'figshape' in plot_options_dict.keys():
        plt.figure(figsize=plot_options_dict['figshape'])
    else:
        plt.figure(figsize=(12,8))

    if 'grid' in plot_options_dict.keys() and plot_options_dict['grid']==True:
        plt.grid(True)

    n_datasets = len(list_of_x_data_bars)
    assert n_datasets == len(list_of_y_data_bars)
    assert n_datasets == len(legend_label_list)

    xtick_labels = None
    if x_values_are_categorical:
        #### In this case x-values need to be identical in every array, otherwise
        #### Plotted bins will not match up
        for x_values_case in list_of_x_data_bars:
            assert np.array_equal(list_of_x_data_bars[0], x_values_case)
        xtick_labels = list_of_x_data_bars[0]
        x_range = [0, len(xtick_labels)]
        list_of_x_values_for_plotting = [np.arange(len(xtick_labels))]*n_datasets
    else:
        x_range = [min([min(x_data) for x_data in list_of_x_data_bars]), max([max(x_data) for x_data in list_of_x_data_bars])]
        list_of_x_values_for_plotting = list_of_x_data_bars

    n_bars = sum([len(x_data) for x_data in list_of_x_data_bars])
    bar_width = get_bar_width(n_bars, x_range)
    if 'color_ordering' in plot_options_dict:
        colors_list = plot_options_dict['color_ordering']
    else:
        colors_list = ['black', 'red', 'blue', 'yellow', 'green', 'purple', 'orange']

    for plot_index, (x_data_bars, y_data_bars, legend_label, color) in enumerate(zip(list_of_x_values_for_plotting, list_of_y_data_bars, legend_label_list, colors_list)):
        x_offset = bar_width * ((float(plot_index) ) - (0.5 * (n_datasets)))
        this_legend_label = legend_label
        if 'means_in_legend' in plot_options_dict.keys() and plot_options_dict['means_in_legend']==True:
            this_legend_label += ', mean='+str(round(np.average(list_of_x_data_bars[plot_index], weights=list_of_y_data_bars[plot_index]), 3))
        plt.bar(left=x_data_bars+x_offset, height=y_data_bars, width=bar_width, color=color, alpha=0.5, label=this_legend_label)

    if legend_label_list != ['']:
        plt.legend(fontsize=plot_options_dict['legend_fontsize'])

    plt.xlabel(plot_options_dict['xlabel'], fontsize=plot_options_dict['xlabel_fontsize'])
    plt.ylabel(plot_options_dict['ylabel'], fontsize=plot_options_dict['ylabel_fontsize'])
    plt.title(plot_options_dict['title'], fontsize=plot_options_dict['title_fontsize'])

    if x_values_are_categorical:
        ### Increase bottom margin for readability
        plt.xticks(np.arange(len(xtick_labels)), xtick_labels, rotation=50, fontsize=8)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)


def plot_histogram(df, column_name, sort=False):
    '''
    Handy function to plot some histograms of DataFrame columns
    '''
    histo = df[column_name].value_counts()
    if (sort):
        histo = histo.sort_index()
    X = np.array(histo.keys())
    Y = histo.values
    plt.bar(np.arange(len(X)), Y, align='center')
    plt.xticks(np.arange(len(X)), X)
    plt.title("Histogram of "+column_name+" values")
    plt.xlabel(column_name)
    plt.ylabel('Frequency')

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)

    plt.show()


def plot_feature_correlation(df, feature_column_name, sort=False):
    '''
    Plot correlation of categorical feature with outcome variable
    '''
    c = 1.0  - df.groupby(feature_column_name)['outcome'].mean()
    if (sort):
        c = c.sort_index()

    X = np.array(c.keys())
    Y = c.values
    plt.bar(np.arange(len(X)), Y, align='center')
    plt.xticks(np.arange(len(X)), X)
    plt.title("Correlation of outcome variable with "+feature_column_name+" categories")
    plt.xlabel(feature_column_name)
    plt.ylabel('Percent non spectrum')

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.show()


def plot_classifier_profiles(bunch_of_classifier_data, plot_title, default_coverage_to_plot = 0.0, specificity_bin_width = 0.025, ylim=(0., 1.), legend_font_size=16, shaded_sensitivity_zones=True):

    fig = plt.figure(figsize=(20, 6))

    #setup axes
    plt.xlabel('specificity', fontsize=28)
    plt.xticks(np.arange(0.0, 1.1, 0.05), fontsize=16)
    plt.xlim(0.0, 1.0)
    plt.ylabel('sensitivity', fontsize=28)
    plt.yticks(np.arange(0.0, 1.1, 0.05), fontsize=16)
    plt.ylim(ylim)

    #add shaded sensitivity zones if required
    if (shaded_sensitivity_zones):
    	plt.axhspan(0.7, 0.8, edgecolor='none', facecolor='lightyellow', alpha=1.0, zorder=1)
    	plt.axhspan(0.8, 0.9, edgecolor='none', facecolor='orange', alpha=0.3, zorder=1)

    #plot data
    for (classifier_info, sensitivity_specificity_dataframe) in bunch_of_classifier_data:
        print 'Plot for classifier info: ', classifier_info

    	#if we're being asked to plot the optimal point only (as opposed to an ROC curve)
    	if ('type' in classifier_info and classifier_info['type'] == 'optimal_point'):
            label = classifier_info['label'] if 'label' in classifier_info else 'unnamed classifier'
            if sensitivity_specificity_dataframe['coverage']<1.0:
                label = label + ' @ '+"{0:.0f}%".format(100*sensitivity_specificity_dataframe['coverage'])+' coverage'
            size = classifier_info['size'] if 'size' in classifier_info else 400
            linestyle = classifier_info['linestyle'] if 'linestyle' in classifier_info else '-'
            alpha = classifier_info['alpha'] if 'alpha' in classifier_info else 0.75
            fill =  classifier_info['fill'] if 'fill' in classifier_info else True
            edgecolors = classifier_info['color'] if 'color' in classifier_info else None
            if (fill):
                facecolors = classifier_info['color'] if 'color' in classifier_info else None
            else:
                facecolors = 'none'

            label = label + " [ {0:.0f}%".format(100*sensitivity_specificity_dataframe['sensitivity'])+' sens, '
            label = label + "{0:.0f}%".format(100*sensitivity_specificity_dataframe['specificity'])+' spec]'

            plt.scatter([sensitivity_specificity_dataframe['specificity']],
                        [sensitivity_specificity_dataframe['sensitivity']],
                        s=size, alpha=alpha, facecolors=facecolors, edgecolors=edgecolors, label=label, zorder=10)

    	#we default to plotting curves
    	else:
            min_acceptable_coverage = classifier_info['coverage'] if 'coverage' in classifier_info else default_coverage_to_plot
            specificity_sensitivity_values = [(spec, sen) for spec, sen in zip(sensitivity_specificity_dataframe['specificity'].values, sensitivity_specificity_dataframe['sensitivity'].values)]
            plot_color = classifier_info['color'] if 'color' in classifier_info else None
            label = classifier_info['label'] if 'label' in classifier_info else 'unnamed classifier'
            linewidth = classifier_info['linewidth'] if 'linewidth' in classifier_info else 3
            linestyle = classifier_info['linestyle'] if 'linestyle' in classifier_info else '-'

            if 'coverage' not in sensitivity_specificity_dataframe:
                plt.plot(sensitivity_specificity_dataframe['specificity'], sensitivity_specificity_dataframe['sensitivity'], marker=None, linewidth=linewidth, label=label, color = plot_color, linestyle=linestyle)
            else:
                sensitivity_specificity_dataframe['rounded_specificity'] = sensitivity_specificity_dataframe['specificity'].apply(lambda x: 0 if np.isnan(x) else specificity_bin_width*(int(x/specificity_bin_width)) )

                acceptable_coverage_sensitivity_specificity_dataframe = sensitivity_specificity_dataframe[sensitivity_specificity_dataframe.coverage>=min_acceptable_coverage]
                min_sensitivity = acceptable_coverage_sensitivity_specificity_dataframe.groupby('rounded_specificity')['sensitivity'].min()
                max_sensitivity = acceptable_coverage_sensitivity_specificity_dataframe.groupby('rounded_specificity')['sensitivity'].max()

                specificity = acceptable_coverage_sensitivity_specificity_dataframe.groupby('rounded_specificity')['rounded_specificity'].max()

                plt.plot(specificity, max_sensitivity, linewidth=linewidth, label=label, color = plot_color, linestyle=linestyle)


    #add legend
    plt.legend(loc="lower left", prop={'size':legend_font_size})

    #add title
    plt.title(plot_title, fontsize=20, fontweight='bold')

    #let's do it!
    plt.show()
    return plt,fig


def barplot_classifier_profiles(bunch_of_classifier_data, plot_title, sensitivity_low=0.75, sensitivity_high=0.85, min_coverage=0.7):
    '''
    Same as above but plots a simple bar chart instead of complicated ROC curves
    '''

    barplot_data = []

    for (classifier_info, sensitivity_specificity_dataframe) in bunch_of_classifier_data:
        label = classifier_info['label'] if 'label' in classifier_info else 'unnamed classifier'

        if 'coverage' in sensitivity_specificity_dataframe.columns:
            sensitivity_specificity_dataframe = sensitivity_specificity_dataframe[(sensitivity_specificity_dataframe['coverage']>=min_coverage) ]

            sensitivity = sensitivity_specificity_dataframe.groupby('rounded_specificity')['sensitivity'].max()
            specificity = sensitivity_specificity_dataframe.groupby('rounded_specificity')['rounded_specificity'].max()
        else:
            sensitivity = sensitivity_specificity_dataframe.groupby('specificity')['sensitivity'].max()
            specificity = sensitivity_specificity_dataframe.groupby('specificity')['specificity'].max()

        temp = pd.DataFrame(zip(specificity,sensitivity), columns=['specificity', 'sensitivity'])
        temp2 = temp[(temp['sensitivity']>=sensitivity_low) & (temp['sensitivity']<=sensitivity_high)]
        bar_height =  temp2['specificity'].mean()

        barplot_data += [ (classifier_info['label'],bar_height) ]

    fig = plt.figure(figsize=(20, 10))

    barlist = plt.barh( range(len(barplot_data)), [x[1] for x in barplot_data] , align='center', edgecolor = "black", alpha=0.8 )
    plt.yticks(range(len(barplot_data)), [x[0] for x in barplot_data])

    #setup value labels
    for i, v in enumerate( [x[1] for x in barplot_data] ):
        plt.text(v - 0.05,  i-0.1, "{0:.0f}%".format(100*v), color='black', fontsize=24)

    #setup name labels
    for i,v in enumerate ( [x[0]['label'] for x in bunch_of_classifier_data] ):
        plt.text(0.02,  i-0.1, v, color='black', fontsize=18)

    #setup colors
    for i in range(0, len(barlist)):
        classifier_info = bunch_of_classifier_data[i][0]
        color = classifier_info['color'] if 'color' in classifier_info else None
        barlist[i].set(facecolor=color)

    #setup axes
    plt.ylabel('algorithm', fontsize=28)
    plt.yticks([])

    plt.xlabel('specificity', fontsize=28)
    plt.xticks(np.arange(0.0, 1.1, 0.05), fontsize=16)
    plt.xlim(0.0, 1.0)

    #add title
    plt.title(plot_title, fontsize=20, fontweight='bold')

    #let's do it!
    print 'show figure with title ', plot_title
    plt.show()
    return plt,fig
