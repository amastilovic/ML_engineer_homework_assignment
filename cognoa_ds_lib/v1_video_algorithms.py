### Algorithms taken from https://github.com/cognoa/cognoa/tree/develop/app/calculators,
### Converted from ruby

import numpy as np


def branch(score, fulcrum, low, high):
    if score < fulcrum:
        return low
    else:
        return high


class Cog1Calculator(object):
    def __init__(self):
        pass

    def is_high_risk(self, risk_level):
        return 1 if risk_level == 'high_risk' else 0

    def get_risk_level(self, row):
        score = row['cog1_response']
        age_months = row['age_months']
        if age_months<48:
            if score < -9:
                return 'high_risk'
            elif score < -6:
                return 'clear_risk'
            elif score < 5:
                return 'watchful_waiting'
            else:
                return 'low_risk'
        else:
            if score < -8:
                return 'high_risk'
            elif score < -3:
                return 'clear_risk'
            elif score < 8:
                return 'medium_risk_asd'
            elif score < 23:
                return 'medium_risk_learning_delay'
            else:
                return 'low_risk'

    def compute_raw_score_on_row(self, row):
        calc_so_far = -1.823
        try:
            vA2 = row['ados1_a5']
            vB1 = row['ados1_b1']
            vB2 = row['ados1_b3']
            vB5 = row['ados1_b5']
            vB9 = row['ados1_b8']
            vB10 = row['ados1_b9']
            vC1 = row['ados1_c3']
            vC2 = row['ados1_c4']
        except:
            print 'Missing data in row ', row
            return np.nan
        calc_so_far = self.B10(vB10, vB2, vC2, calc_so_far)
        calc_so_far = self.B1(vB1, vA2, calc_so_far)
        calc_so_far = self.B5(vB5, calc_so_far)
        calc_so_far = self.B9(vB9, calc_so_far)
        calc_so_far = self.C1(vC1, calc_so_far)

        return calc_so_far

    def B10(self, vB10, vB2, vC2, init):
        if vB10<1.5:
            init+=0.254
            init = self.B2(vB2,vC2,init)
        elif vB10>=1.5:
            init+=-1.345

        if vB10<0.5:
            init+=0.735
        elif vB10>=0.5:
            init+=-0.47

        return init

    def B2(self, vB2,vC2,init):
        if vB2<1.5:
            init+=0.545
            init=self.C2(vC2,init)
        elif vB2>=1.5:
            init+=-0.487

        return init

    def C2(self, vC2, init):
        init += branch(vC2, 1.5, 0.404, -0.108)
        return init

    def B1(self, vB1, vA2, init):
        if vB1<1:
            init+=0.99
        elif vB1>=1:
            init+=-0.532
            init = self.A2(vA2, init)
        return init

    def B5(self, vB5, init):
        init += branch(vB5, 0.5, -0.245, -1.144)
        return init

    def B9(self, vB9, init):
        init += branch(vB9, 0.5, 0.184, -0.276)
        init += branch(vB9, 1.987, -1.015, 2.643)
        return init

    def A2(self, vA2, init):
        init += branch(vA2, 0.5, -0.705, -0.954)
        return init

    def C1(self, vC1, init):
        init += branch(vC1, 0.5, 0.365, +0.974)
        return init

    def compute_raw_scores_on_df(self, input_df):
        ''' Assumes that all rows in input_df are cog1 based '''
        input_df['cog1_response'] = input_df.apply(self.compute_raw_score_on_row, axis=1)
        input_df['risk_level'] = input_df[['age_months', 'cog1_response']].apply(self.get_risk_level, axis=1)
        input_df['is_high_risk'] = input_df['risk_level'].apply(self.is_high_risk)
        return input_df


class Cog2Calculator(object):
    def __init__(self):
        pass

    def is_high_risk(self, risk_level):
        return 1 if risk_level == 'high_risk' else 0

    def get_risk_level(self, score):
        if score >= 0.6:
            return 'high_risk'
        elif score >= 0.3:
            return 'medium_risk'
        else:
            return 'low_risk'

    def clean_column(self, value):
        if value in [7,8]:
            return 0
        elif value in [0,1,2,3,4]:
            return value
        raise ValueError('unexpected score '+str(value))

    def compute_raw_scores_on_df(self, input_df):
        questions = ['ados2_a5', 'ados2_a8', 'ados2_b1', 'ados2_b3', 'ados2_b6', 'ados2_b8', 'ados2_b10', 'ados2_d2', 'ados2_d4']
        df_for_calculation = input_df[questions]
        ### Clean it
        for question in questions:
            df_for_calculation[question] = df_for_calculation[question].apply(self.clean_column)
            log_odds = (-15.8657 + 2.2539 * input_df['ados2_a5'] + 3.0323 * input_df['ados2_a8'] +\
                            3.8820 * input_df['ados2_b1'] + 4.3625 * input_df['ados2_c4'] +\
                            5.0750 * input_df['ados2_b8'] + 4.0215 * input_df['ados2_b8'] +\
                            3.8299 * input_df['ados2_b9'] + 3.4053 * input_df['ados2_d2'] + 2.6616 * input_df['ados2_c3'])
            probability_of_diagnosis = 1./(1. + np.exp(-1 * log_odds))
            input_df['cog2_response'] = probability_of_diagnosis
            input_df['risk_level'] = input_df['cog2_response'].apply(self.get_risk_level)
            input_df['is_high_risk'] = input_df['risk_level'].apply(self.is_high_risk)
            return input_df
