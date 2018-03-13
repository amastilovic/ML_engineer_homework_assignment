from io.ds_helper_io import *


def inject_proportional_loss_when_presence_encoding(inputDF,
                                                    outcome_key='outcome',
                                                    instructions=None,
                                                    missing_value='missing',
                                                    prior_autism_frac=None,
                                                    module=None,
                                                    validation=True):
    '''
    instructions: if None will be filled with expected default values. Format should be
    a list of dictionaries where each entry tells the association of the feature (whether
    presence means autism, non-autism, or neutral). Example:
    instructions = [{'feature': 'ados1_a1', 'presence_means': 'autism'}, {'feature': 'ados1_a2', 'presence_means': 'not'},
                    {'feature': 'ados1_a3', 'presence_means': 'neutral'}]
    prior_autism_frac: if you want proportionality after reweighting priors then specify prior_autism_frac
    module : if you want to only inject in features of a given module then add this requirement
    Intended to inject loss for autism cases and not autism cases separately, with the goal of
    making it so that the non-presence of a feature is not used to make any important decisions in the
    trees.

    To accomplish this, each feature is considered to be either one where the presence implies the child
    is more likely to be autistic, or less likely to be autistic. If more likely, then the absence of the feature
    is likely to be interpreted as a reduction in the autism probability. To compensate for this missing values
    should be injected in real autism cases until the fraction is in balance with the non autism cases.
    '''

    def get_frac_injection(no_presence_df, presence_df, outcome_key, presence_means, autism_scaling_factor):
        ## How much do we need to inject to achieve balance in the non-presence category??
        n_not_no_presence = float(len(no_presence_df[no_presence_df[outcome_key]=='not'].index))
        n_autism_no_presence = float(len(no_presence_df[no_presence_df[outcome_key]=='autism'].index))
        n_not_presence = float(len(presence_df[presence_df[outcome_key]=='not'].index))
        n_autism_presence = float(len(presence_df[presence_df[outcome_key]=='autism'].index))
        if autism_scaling_factor is not None:
            n_autism_no_presence *= autism_scaling_factor
            n_autism_presence *= autism_scaling_factor

        if presence_means == 'autism':
            # inject into autism to make frac as large as not in non-presence category
            frac_injection = (n_not_no_presence -  n_autism_no_presence) / (n_autism_no_presence + n_autism_presence)
        elif presence_means == 'not':   # inject into not to make frac as large as autism in non-presence category
            frac_injection = (n_autism_no_presence - n_not_no_presence) / (n_not_no_presence + n_not_presence)
        else:
            logging.error("type of injection = %s not understood" % str(presence_means))
            raise ValueError

        if frac_injection < 0:
            logging.warning('Warning: frac_injection < 0, meaning selection feature is not enriching correct outcome. Skip injection.')
            return None
        else:
            return frac_injection

    if instructions is None:
        instructions = [
            {'feature': 'ados1_a1', 'presence_means': 'not'},
            {'feature': 'ados1_a3', 'presence_means': 'autism'},
            {'feature': 'ados1_a7', 'presence_means': 'not'},
            {'feature': 'ados1_a8', 'presence_means': 'not'},
            {'feature': 'ados1_b9', 'presence_means': 'not'},
            {'feature': 'ados1_b10', 'presence_means': 'not'},
            {'feature': 'ados1_b12', 'presence_means': 'not'},
            {'feature': 'ados1_d1', 'presence_means': 'autism'},
            {'feature': 'ados1_d2', 'presence_means': 'autism'},
            {'feature': 'ados1_d4', 'presence_means': 'autism'},
            {'feature': 'ados2_a3', 'presence_means': 'autism'},
            {'feature': 'ados2_a5', 'presence_means': 'autism'},
            {'feature': 'ados2_b1', 'presence_means': 'autism'},
            {'feature': 'ados2_b2', 'presence_means': 'not'},
            {'feature': 'ados2_b6', 'presence_means': 'not'},
            {'feature': 'ados2_d1', 'presence_means': 'autism'},
            {'feature': 'ados2_d2', 'presence_means': 'autism'},
            {'feature': 'ados2_d4', 'presence_means': 'autism'},
            {'feature': 'ados2_e3', 'presence_means': 'autism'},

        	## v1 features
            {'feature': 'ados1_a2', 'presence_means': 'not'},
            {'feature': 'ados1_b1', 'presence_means': 'autism'},
            {'feature': 'ados1_b2', 'presence_means': 'not'},
            {'feature': 'ados1_b5', 'presence_means': 'not'},
            {'feature': 'ados1_c1', 'presence_means': 'not'},
            {'feature': 'ados1_c2', 'presence_means': 'not'},
            {'feature': 'ados2_a8', 'presence_means': 'not'},
            {'feature': 'ados2_b3', 'presence_means': 'not'},
            {'feature': 'ados2_b8', 'presence_means': 'not'},
            {'feature': 'ados2_b10', 'presence_means': 'not'},
        ]

    feature_columns = [instruction['feature'] for instruction in instructions if instruction['feature'] in inputDF.columns]
    feature_encoding_map = {feature: 'presence_of_behavior' for feature in feature_columns}

    # transform the data into presence encoded results
    encoded_df, _, _ = prepare_data_for_modeling(inputDF, feature_columns, feature_encoding_map, target_column=outcome_key)

    # The columns will have a '_behavior_present' suffix. Remove this.
    new_cols = [col[:-len('_behavior_present')] if col.endswith('_behavior_present') else col for col in encoded_df.columns]
    encoded_df.columns = new_cols
    encoded_df[outcome_key] = cp.deepcopy(inputDF[outcome_key])

    autism_prior_scaling_factor = None
    if prior_autism_frac is not None:  # correct values to ensure balancing is to correct priors
        if prior_autism_frac > 0.99999 or prior_autism_frac < 0.000001:
            print 'Error, prior_autism_frac: ', prior_autism_frac, ' not understood'
            raise ValueError
        n_autism_tot = float(len(inputDF[inputDF[outcome_key]=='autism'].index))
        n_not_tot = float(len(inputDF.index) - n_autism_tot)
        autism_prior_scaling_factor = prior_autism_frac * n_not_tot / (n_autism_tot * (1. - prior_autism_frac))

    # First determine what our loss instructions should be
    autism_loss_instructions = {'desc': 'autism_loss_instructions', 'instructions': []}
    not_loss_instructions = {'desc': 'not_loss_instructions', 'instructions': []}
    suspicious_features = []
    for instruct in instructions:
        feature = instruct['feature']
        if (module is not None) and ('ados'+str(module) not in feature):
            continue

        if feature not in encoded_df.columns:
            continue

        presence_means = instruct['presence_means']
        if presence_means == 'neutral':
            continue
        elif presence_means == 'autism':
            # May need to inject missing values into 'not' in order to achieve balance
            no_presence_df = encoded_df[encoded_df[feature]==0][[feature, outcome_key]]
            presence_df = encoded_df[encoded_df[feature]==1][[feature, outcome_key]]
            needed_frac_injection = get_frac_injection(no_presence_df, presence_df, outcome_key, presence_means, autism_prior_scaling_factor)
            if needed_frac_injection is None:
                suspicious_features.append(feature)
            else:
                autism_loss_instructions['instructions'].append({'qType': feature, 'probability': needed_frac_injection})
                if needed_frac_injection>0.5:
                    suspicious_features.append(feature)
            #print 'for feature: ', feature, ', needed_frac_injection: ', needed_frac_injection
        elif presence_means == 'not':
            # May need to inject missing values into 'autism' in order to achieve balance
            no_presence_df = encoded_df[encoded_df[feature]==0][[feature, outcome_key]]
            presence_df = encoded_df[encoded_df[feature]==1][[feature, outcome_key]]
            needed_frac_injection = get_frac_injection(no_presence_df, presence_df, outcome_key, presence_means, autism_prior_scaling_factor)

            if needed_frac_injection is None:
                suspicious_features.append(feature)
            else:
                not_loss_instructions['instructions'].append({'qType': feature, 'probability': needed_frac_injection})
                if needed_frac_injection>0.5:
                    suspicious_features.append(feature)
        else:
            print 'Error, instructions ', instruct, ' not understood. Abort.'
            return -1

    # Now apply our loss instructions
    autism_df = inputDF[inputDF[outcome_key]=='autism']
    not_df = inputDF[inputDF[outcome_key]=='not']
    if len(autism_loss_instructions['instructions']) > 0:   ## apply our loss instructions
        autism_df = injectLoss(autism_df, autism_loss_instructions, missingValue=missing_value, mode='duplicate',
             probKey='probability', scaleToOverwrite=True, exactMatch=True)
    if len(not_loss_instructions['instructions']) > 0:    ## apply our loss instructions
        not_df = injectLoss(not_df, not_loss_instructions, missingValue=missing_value, mode='duplicate',
             probKey='probability', scaleToOverwrite=True, exactMatch=True)

    # Now merge the results and re-shuffle to avoid grouping by autism / not
    output_df = autism_df.append([not_df], ignore_index=True)
    output_df = (output_df.reindex(np.random.permutation(output_df.index))).reset_index()

    if validation:
        do_proportional_injection_sanity_checks(encoded_df, output_df, instructions, feature_columns, suspicious_features, feature_encoding_map, outcome_key, autism_prior_scaling_factor, module)

    return output_df


def do_proportional_injection_sanity_checks(encoded_df, output_df, instructions, feature_columns, suspicious_features, feature_encoding_map, outcome_key, autism_prior_scaling_factor, module):
    autism_scale_factor = 1. if autism_prior_scaling_factor is None else autism_prior_scaling_factor

    # sanity check results
    out_encoded_df, _, _ = prepare_data_for_modeling(output_df, feature_columns, feature_encoding_map, target_column=outcome_key)
    new_cols = [col[:-len('_behavior_present')] if col.endswith('_behavior_present') else col for col in out_encoded_df.columns]
    out_encoded_df.columns = new_cols
    out_encoded_df[outcome_key] = cp.deepcopy(output_df[outcome_key])

    features = []
    presence_means = []
    n_not_dict = collections.OrderedDict([
            ('before', []),
            ('after', []),
             ])
    n_autism_dict = collections.OrderedDict([
            ('before', []),
            ('after', []),
            ('before_weighted', []),
            ('after_weighted', []),
             ])
    autism_frac_dict = collections.OrderedDict([
            ('before', []),
            ('after', []),
            ('before_weighted', []),
            ('after_weighted', []),
    ])

    for instruct in instructions:
        feature = instruct['feature']
        if (module is not None) and ('ados'+str(module) not in feature):
            continue

        if feature not in encoded_df.columns:
            continue

        features.append(feature)
        presence_means.append(instruct['presence_means'])
        no_presence_df = encoded_df[encoded_df[feature]==0][[feature, outcome_key]]
        out_no_presence_df = out_encoded_df[out_encoded_df[feature]==0][[feature, outcome_key]]

        logger.info("For instructions %s, no_presence_df: %s" % (str(instruct), str(no_presence_df)))

        n_not = float(len(no_presence_df[no_presence_df[outcome_key]=='not'].index))
        n_autism = float(len(no_presence_df[no_presence_df[outcome_key]=='autism'].index))

        logger.info("n_not: %s, n_autism: %s" % (str(n_not), str(n_autism)))

        n_autism_weighted = autism_scale_factor*n_autism
        autism_frac = n_autism / (n_not + n_autism)
        autism_frac_weighted = n_autism_weighted / (n_autism_weighted + n_not)
        n_not_out = float(len(out_no_presence_df[out_no_presence_df[outcome_key]=='not'].index))
        n_autism_out = float(len(out_no_presence_df[out_no_presence_df[outcome_key]=='autism'].index))
        n_autism_out_weighted = autism_scale_factor*n_autism_out
        autism_out_frac = n_autism_out / (n_autism_out + n_not_out)
        autism_out_frac_weighted = n_autism_out_weighted / (n_autism_out_weighted + n_not_out)

        n_not_dict['before'].append(n_not)
        n_not_dict['after'].append(n_not_out)
        n_autism_dict['before'].append(n_autism)
        n_autism_dict['before_weighted'].append(n_autism_weighted)
        n_autism_dict['after'].append(n_autism_out)
        n_autism_dict['after_weighted'].append(n_autism_out_weighted)
        autism_frac_dict['before'].append(autism_frac)
        autism_frac_dict['before_weighted'].append(autism_frac_weighted)
        autism_frac_dict['after'].append(autism_out_frac)
        autism_frac_dict['after_weighted'].append(autism_out_frac_weighted)

    draw_sanity_overlays(n_not_dict, features, presence_means, suspicious_features, title='Number of not autism results', ylabel='Number of children when feature not present', ylims=None)
    draw_sanity_overlays(n_autism_dict, features, presence_means, suspicious_features, title='Number of autism results', ylabel='Number of children when feature not present', ylims=None)
    draw_sanity_overlays(autism_frac_dict, features, presence_means, suspicious_features, title='Autism frac results', ylabel='Autism fraction when feature not present', ylims=[0., 1.4], draw_comp_line=0.5)


def injectLoss(inputDF,
               lossSpecifications,
               missingValue=999,
               mode='duplicate',
               probKey='probability',
               scaleToOverwrite=False,
               exactMatch=False):
    '''
    Function to inject loss simulating unknowable results in an imperfect survey

    Example epected input format:
        lossSpecifications = {'desc': '10pLoss, 'instructions': [{'qType': 'ados2_a', 'probability': 0.1},
                                                                 {'qType': 'ados2_b', 'probability': 0.1},
                                                                 ...]}
    Method: for every event, throws random number between 0 and 1 for each element of instructions
    If result is less than probability, all questions matching qType description will have their values
    Reverted to a missing value

    'missingValue': either a value to replace with, or a 'rand', which represents a random integer
    from [0,1,2,3,4]

    mode can be 'duplicate' or 'overwrite'
    ... duplicate means to make a copy of the row with the values missing and append it
    ...... If scaleToOverwrite is set to True then the probability will be scaled up so that
    ...... The final number of missing values is the same as expected from the overwrite option
    .......... Caveat: obviously since the max probability is 1.0 no probability value over 0.5
    .......... can work with this option unless multiple copies of the missing events are used
    .......... and in that case it becomes ambiguous when multiple missing values with tdiferent probabilities
    .......... are specified in the instructions.
    ... overwrite means overwrite the current row with the new one that has the missing values

    Returns data frame rebuilt with appropriate missing values
    '''

    desc = lossSpecifications['desc']
    instructions = lossSpecifications['instructions']
    if instructions is None:
        return inputDF

    ## What is probability of having a row lost??
    pGivenRowNotLost = 1.
    for instruct in instructions:
        pGivenRowNotLost *= 1. - max(instruct[probKey], 0.9999999)
    pGivenRowLost = 1. - pGivenRowNotLost

    nRows = len(inputDF.index)
    outputDF = pd.DataFrame()
    colsAlreadyDone = []
    colsToReset = []
    resetDF = cp.deepcopy(inputDF)
    resetDF['colsToReset'] = [[]]*len(resetDF.index)
    def markColsForReset(row, colsToReset):
        return row['colsToReset'].append(colsToReset)

    ## Loop over the instructions for loss types to apply
    for instr in instructions:
        instrPLoss = instr[probKey]

        if scaleToOverwrite:
            # Correct for the duplication factor
            assert mode == 'duplicate'
            instrPLoss = instrPLoss / (1. - min(pGivenRowLost, 0.5))
            if instrPLoss > 1.:
                logger.warning('Warning, hit physical ceiling')
                instrPLoss = 1.   ## hit physical ceiling
            ## Now instrPLoss has been scaled to
            ## value that will make the output
            ## missing rate equal this probability
            ## after non-missing duplicates are factored in
        #applyLossToTheseCols = [ele for ele in inputDF.columns if instr['qType'] in ele and instr['except'] not in ele]
        matchCheck = lambda x : instr['qType'] in ele
        if exactMatch:
              matchCheck = lambda x : instr['qType'] == ele

        if 'except' in instr.keys():
            applyLossToTheseCols = [ele for ele in inputDF.columns if matchCheck(ele) and instr['except'] not in ele]
        else:
            applyLossToTheseCols = [ele for ele in inputDF.columns if matchCheck(ele)]

        # Check that none of the new columns have already had instructions
        overlapCols = np.intersect1d(colsAlreadyDone, applyLossToTheseCols)
        if len(overlapCols) != 0:
            logger.error("Error, %s have conflicting instructions. Abort." % str(overlapCols))
            raise ValueError

        colsAlreadyDone += applyLossToTheseCols

        ## get which rows need to have this subset of columns reset
        rowsToReset = (np.random.rand(nRows) < instrPLoss)

        ## expand the list of columns that do need a reset for each row:
        resetDF['colsToReset'] = [list(curContent) + applyLossToTheseCols if doThisRow else list(curContent)\
                  for curContent, doThisRow in zip(resetDF['colsToReset'].values, rowsToReset)]

    # Define the pandas operation that will do the reset
    def doResets(row):
        outRow = row
        for key in row['colsToReset']:
            if type(missingValue) == str and missingValue == 'random':
                randValToUse = int(5.*np.random.rand(1)[0])
                outRow[key] = randValToUse
            else:
                outRow[key] = missingValue
        return outRow

    # inject some tracking information to enable reconstruction of original later
    resetDF['original_index'] = resetDF.index
    resetDF['status'] = ['original']*len(resetDF.index)

    # Now actually apply the resets
    if mode == 'overwrite':
        outDF = resetDF.apply(doResets, axis=1)

    if mode == 'duplicate':
        # In this case we keep a copy of the original rows in the DF
        appendDF = resetDF[np.array([ele != [] for ele in resetDF['colsToReset']])]
        appendDF = appendDF.apply(doResets, axis=1)
        appendDF['status'] = ['duplicate']*len(appendDF.index)
        outDF = resetDF.append([appendDF], ignore_index=True)

    return outDF
