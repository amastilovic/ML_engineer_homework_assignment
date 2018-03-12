import numpy as np
import pandas as pd


def get_missing_instructions_from_SQL(featuresToDo=None, minProb=0.01, desc='default', module=1):
    '''
    featuresToDo: this should be an array with indexes for module 1 and module 2 with contents
    as the feature names missing values should be added to.

	If you set desc to 'clinical' then you can just take default values. No need to enter anything for featuresToDo.
    '''

    ### These were determined approximately from comparisons of missing data (when presence encoded!) compared with Thompson data
	### For average analyzer in https://docs.google.com/presentation/d/1dNm721rsGDPhPCbH_zUhEv5SR8LrNlIrWUQiSnEgVys/edit#slide=id.g17cb8fb81d_0_59
	### Only actually used questions encoded here, others left blank
	####### Note: there are big uncertainties on this technique when the denominator (the rate of presence encoding in clinical setting) is small
    if desc == 'clinical_presence_encoded':
        lossInstructions = [
            {},
            {
                'desc': desc,
                'instructions': [
                     {'qType': 'ados1_a1', 'probability': max(minProb, 1. - (0.5 / 0.8))},
                     #{'qType': 'ados1_a2', 'probability': max(minProb, 0)},
                     {'qType': 'ados1_a3', 'probability': max(minProb, 1. - (0.4 / 0.8))},
                     {'qType': 'ados1_a7', 'probability': max(minProb, 1. - (0.1 / 0.55))},
                     {'qType': 'ados1_a8', 'probability': max(minProb, 1. - (0.25 / 0.7))},
                     # {'qType': 'ados1_b1', 'probability': max(minProb, 0.))},
                     #{'qType': 'ados1_b5', 'probability': max(minProb, 0.)},
                     {'qType': 'ados1_b9', 'probability': max(minProb, 0.)},    ### was good average agreement
                     {'qType': 'ados1_b10', 'probability': max(minProb, 1. - (0.35 / 0.4))},
                     {'qType': 'ados1_b12', 'probability': max(minProb, 1. - (0.2 / 0.3))},
                     {'qType': 'ados1_d1', 'probability': max(minProb, 1. - (0.35 / 0.55))},
                     {'qType': 'ados1_d2', 'probability': max(minProb, 1. - (0.25 / 0.55))},
                     {'qType': 'ados1_d4', 'probability': max(minProb, 1. - (0.4 / 0.95))}]
            },
            {
                'desc': desc,
                'instructions': [
                    {'qType': 'ados2_a3', 'probability': max(minProb, 1. - (0.6 / 0.9))},
                    {'qType': 'ados2_a5', 'probability': max(minProb, 1. - (0.2 / 0.25))},
                    #{'qType': 'ados2_a8', 'probability': max(minProb, 1. - ()},
                    {'qType': 'ados2_b1', 'probability': max(minProb, 1. - (0.45 / 0.9))},
                    {'qType': 'ados2_b2', 'probability': max(minProb, 1. - (0.8 / 0.95))},
                    #{'qType': 'ados2_b3', 'probability': max(minProb, 0.)},
                    {'qType': 'ados2_b6', 'probability': max(minProb, 1. - (0.6 / 0.75))},
                    #{'qType': 'ados2_b8', 'probability': max(minProb, 0.)},
                    {'qType': 'ados2_d1', 'probability': max(minProb, 1. - (0.2 / 0.4))},
                    {'qType': 'ados2_d2', 'probability': max(minProb, 1. - (0.25 / 0.4))},
                    {'qType': 'ados2_d4', 'probability': max(minProb, 1. - (0.3 / 0.9))},
                    {'qType': 'ados2_e3', 'probability': max(minProb, 0.)}]    #### was good average agreeemnt
            }
        ]

        return lossInstructions

    ####  These were determined in late summer (August?) 2016, using the rate of missing values in clinical data
    elif desc == 'clinical':
        #### Clinical probabilities to use (from clinical study values):
        lossInstructions = [
            {},
            {
                'desc': desc,
                'instructions': [
                    {'qType': 'ados1_a2', 'probability': max(minProb, 0.)},
                    {'qType': 'ados1_a1', 'probability': max(minProb, 0.509090909090909)},
                    {'qType': 'ados1_a3', 'probability': max(minProb, 0.8181818181818182)},
                    {'qType': 'ados1_a7', 'probability': max(minProb, 0.)},
                    {'qType': 'ados1_a8', 'probability': max(minProb, 0.012121212121212121)},
                    {'qType': 'ados1_b1', 'probability': max(minProb, 3. / (3. + 47. + 113.))},
                    {'qType': 'ados1_b5', 'probability': max(minProb, 0.)},
                    {'qType': 'ados1_b10', 'probability': max(minProb, 0.024242424242424242)},
                    {'qType': 'ados1_b12', 'probability': max(minProb, 0.006060606060606061)},
                    {'qType': 'ados1_b9', 'probability': max(minProb, 0.006060606060606061)},
                    {'qType': 'ados1_d1', 'probability': max(minProb, 0.)},
                    {'qType': 'ados1_d2', 'probability': max(minProb, 0.)},
                    {'qType': 'ados1_d4', 'probability': max(minProb, 0.)}]
            },
            {
                'desc': desc,
                'instructions': [
                    {'qType': 'ados2_a3', 'probability': max(minProb, 0.)},
                    {'qType': 'ados2_a5', 'probability': max(minProb, 0.02631578947368421)},
                    {'qType': 'ados2_a8', 'probability': max(minProb, 0.5)},
                    {'qType': 'ados2_b1', 'probability': max(minProb, 0.13157894736842105)},
                    {'qType': 'ados2_b2', 'probability': max(minProb, 0.02631578947368421)},
                    {'qType': 'ados2_b3', 'probability': max(minProb, 0.)},
                    {'qType': 'ados2_b6', 'probability': max(minProb, 0.27631578947368424)},
                    {'qType': 'ados2_b8', 'probability': max(minProb, 0.)},
                    {'qType': 'ados2_d1', 'probability': max(minProb, 0.)},
                    {'qType': 'ados2_d2', 'probability': max(minProb, 0.)},
                    {'qType': 'ados2_d4', 'probability': max(minProb, 0.)},
                    {'qType': 'ados2_e3', 'probability': max(minProb, 0.)}]
            }
        ]

        return lossInstructions

    ### Otherwise use psql
    ### SQL probabilities to use:
    ### See https://docs.google.com/spreadsheets/d/1nNsaMR2jKlCspy7hgpx18XdKj2PM3T58EcrwJnoq74I/edit
    ados_instructions = [None,None,None]
    nADOS1_old = 110.
    nADOS1_new = 1490.
    nADOS1 = nADOS1_old + nADOS1_new
    ados_probs = [None, {}, {}]
    ados_probs[1]['a'] = np.array([(0.+3.) / nADOS1])
    ados_probs[1]['b'] = np.array([(3.+29.) / nADOS1, (4. + 67.) / nADOS1, (3. + 20.) / nADOS1, (2. + 49.) / nADOS1, (14.+112.) / nADOS1])
    ados_probs[1]['c'] = np.array([(16.+211.) / nADOS1, (26.+591.) / nADOS1])
    ados_probs[1]['d'] = np.array([])
    ados_probs[1]['e'] = np.array([])
    ados_probs[1]['all'] = np.array(list(ados_probs[1]['a']) + list(ados_probs[1]['b']) + list(ados_probs[1]['c']))

    ados_probs[2]['a'] = np.array([3. / 916., 662. / 916.])
    ados_probs[2]['b'] = np.array([25. / 196., 5. / 916, 112. / 916., 5. / 916., 5. / 916.])
    ados_probs[2]['c'] = np.array([])
    ados_probs[2]['d'] = np.array([0. / 916, 1. / 916])
    ados_probs[2]['e'] = np.array([])
    ados_probs[2]['all'] = np.array(list(ados_probs[2]['a']) + list(ados_probs[2]['b']) + list(ados_probs[2]['d']))

    for module in [1,2]:
        if module == 1:
            listOfInstructions = [
                {'qType': 'ados1_b9', 'probability': max(minProb, (2.+49.) / nADOS1) },
                {'qType': 'ados1_b10', 'probability': max(minProb, (14.+112.) / nADOS1) }
            ]
        else:
            listOfInstructions = [
                {'qType': 'ados2_a5', 'probability': max(minProb, 3. / 916.)},
                {'qType': 'ados2_a8', 'probability': max(minProb, 662. / 916.)},
                {'qType': 'ados2_b1', 'probability': max(minProb, 25. / 916.)},
                {'qType': 'ados2_b6', 'probability': max(minProb, 112. / 916.)},
                {'qType': 'ados2_d2', 'probability': max(minProb, 0. / 916.)},
                {'qType': 'ados2_d4', 'probability': max(minProb, 1. / 916.)}
            ]

        for feature in featuresToDo[module]:
            newFeature = True
            for oldFeature in listOfInstructions:
                if feature == oldFeature['qType']:
                    newFeature = False
            if newFeature == False: continue
            featureLetter = feature[6]
            assert featureLetter in ['a', 'b', 'c', 'd', 'e']
            catData = ados_probs[module].get(featureLetter, ados_probs[module]['all'])

#            catData = ados_probs[module][featureLetter]
            if len(catData) == 0:
                catData = ados_probs[module]['all']

            listOfInstructions.append({'qType': feature, 'probability': max(minProb, np.mean(catData))})
        ados_instructions[module] = pd.DataFrame(listOfInstructions)

    #### Make an error range choice
    ados_probs_fluct_90per_downVals = [{}, {}, {}]
    ados_probs_fluct_90per_upVals = [{}, {}, {}]
    ados_probs_fluct_oneSigma_downVals = [{}, {}, {}]
    ados_probs_fluct_oneSigma_upVals = [{}, {}, {}]
    for inner_module in [1,2]:
        for key in ados_probs[inner_module].keys():
            #### One sigma hypothesis
            ados_probs_fluct_90per_downVals[inner_module][key] = np.nan if len(ados_probs[inner_module][key]) == 0 else\
                                                       np.percentile(ados_probs[inner_module][key], 10.0)
            ados_probs_fluct_90per_upVals[inner_module][key] = np.nan if len(ados_probs[inner_module][key]) == 0 else\
                                                       np.percentile(ados_probs[inner_module][key], 90.0)
            ados_probs_fluct_oneSigma_downVals[inner_module][key] = np.nan if len(ados_probs[inner_module][key]) == 0 else\
                                                       np.percentile(ados_probs[inner_module][key], 31.73)
            ados_probs_fluct_oneSigma_upVals[inner_module][key] = np.nan if len(ados_probs[inner_module][key]) == 0 else\
                                                       np.percentile(ados_probs[inner_module][key], 68.27)

    #### Convert to "errors". If fewer than "3" available in category then use "all".
    ados_probs_fluct_downErr = [{}, {}, {}]
    ados_probs_fluct_upErr = [{}, {}, {}]
    for inner_module in [1,2]:
        for key in ados_probs[inner_module].keys():
            if len(ados_probs[inner_module][key]) >= 3:
                ados_probs_fluct_downErr[inner_module][key] = np.mean(ados_probs[inner_module][key]) - ados_probs_fluct_oneSigma_downVals[inner_module][key]
                ados_probs_fluct_upErr[inner_module][key] = ados_probs_fluct_oneSigma_upVals[inner_module][key] - np.mean(ados_probs[inner_module][key])
            else:
                ados_probs_fluct_downErr[inner_module][key] = np.mean(ados_probs[inner_module]['all']) - ados_probs_fluct_90per_downVals[inner_module]['all']
                ados_probs_fluct_upErr[inner_module][key] = ados_probs_fluct_90per_upVals[inner_module]['all'] - np.mean(ados_probs[inner_module]['all'])

    #### Determine final lower and upper bounds on missing values
    ados_probs_downVals = [{}, {}, {}]
    ados_probs_upVals = [{}, {}, {}]
    for inner_module in [1,2]:
        qCategories = ados_instructions[inner_module]['qType'].apply(lambda x : x[6:7])
        qErrsUp = np.array([max(0.05, ados_probs_fluct_upErr[inner_module][ele]) for ele in qCategories.values])
        qErrsDown = np.array([max(0.05, ados_probs_fluct_downErr[inner_module][ele]) for ele in qCategories.values])
        ados_instructions[inner_module]['probUp'] = np.minimum(1., ados_instructions[inner_module]['probability'].values + qErrsUp)
        ados_instructions[inner_module]['probDown'] = np.maximum(0., ados_instructions[inner_module]['probability'].values - qErrsDown)

    return [
        {},
        {'desc': desc, 'instructions': ados_instructions[1].T.to_dict().values()},
        {'desc': desc, 'instructions': ados_instructions[2].T.to_dict().values()}
    ]
