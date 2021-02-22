import pandas as pd
import numpy as np

'''
methods for implementing functions from https://arxiv.org/pdf/1906.00285.pdf
'''

def read_and_annotate(path, threshhold):
    '''
    reads in a CSV of 311 request data as dataframe, annotates it with a new column, 'LABEL'
    'LABEL' = 0 or 1 depending on if the request was completed in more than or less than the thresshold time
    (so 1 = quicker response)
    and returns the dataframe
    :param path: filepath to read from
    :param threshhold: a number of seconds
    :return: a pandas dataframe
    '''

    df = pd.read_csv(path, index_col=0,     dtype = {'CITY': np.str, 'STATE': np.str, 'ZIP_CODE': np.str,
             'STREET_NUMBER': np.str, 'LEGACY_SR_NUMBER': np.str,
             'PARENT_SR_NUMBER': np.str, 'SANITATION_DIVISION_DAYS': np.str,
             'BLOCK_GROUP': np.str}) #get a bunch of warnings unless you specify the datatype for these guys

    df['LABEL'] = df.apply(lambda row: 1 if row['DELTA'] >= threshhold else 0, axis=1)

    return df

def percent_from_demographic(alpha, df_311, df_census):
    '''
    method for finding P(A=alpha) from the paper, used in equation (7).
    estimates the percent of requests from a given demographic
    :param alpha: a *set* of census codes (not necessarily just one) describing the demographic group (i.e. {B03002003, B03002013} for hispanic and non-hispanic white people)
    :param df_311: the dataframe of 311 data to look through
    :param df_census: the dataframe of census data (called "demographics_table" in the box repository)
    :return: the estimated percent of 311 records from people of that demographic.
    '''


    # get a new vector indexed by block groups, containing the number of requests from each block group

    bg_counts = df_311['BLOCK_GROUP'].value_counts()

    # get a vector 'bg_ratios' decribing the % of people in each block group from the demographic in question
    # B03002001 is the census count for total

    def ratio_calc(row):
        total = row['B03002001 - count']
        if total == 0: return 0
        return sum([row[a + ' - count'] for a in alpha]) / total

    bg_ratios = df_311.apply(lambda row: ratio_calc(row), axis=1)

    #now the dot product of bg_counts and bg_ratios is the approx number of requests from the demographic alpha

    return bg_counts.dot(bg_ratios)

def mu(w, alpha, df_311, df_census):
    '''
    implements the function mu(alpha; w) from eq. (7) of the paper
    :param w: another function, called within mu
    :param alpha: a demographic, and a parameter for w. expressed as a set of census codes.
    :param df_311: dataframe of 311 data
    :param df_census: dataframe of census data
    :return: mu(alpha; w)
    '''

    # two parts: first, take the expectation of mu_alpha(Y hat, Z) * Y hat over all Y hat and Z
    # then divide that by P(A = alpha)

    # part 1.

    # first, get a list of all Z (all block groups)

    Z = list(df_311['BLOCK_GROUP'].value_counts().index)

    # now, the expectation is the sum over all Y hat and Z of P(Y hat, Z) * (w_alpha(Y hat, Z) * Y hat)
    # but since we're multiplying by Y hat at the end, that means that when Y hat = 0 we're guaranteed to not contribute anything to the sum
    # so we really just need to sum over all Z, keeping Y hat fixed as 1.

    expectation = 0
    total_records = len(df_311)
    for z in Z:
        w_result = w(alpha, 1, z)

        P_Yhat_Z = len(df_311.loc[df_311['BLOCK_GROUP'] == z].loc[df_311['LABEL'] == 1]) / total_records

        expectation += w_result * P_Yhat_Z

    # part 2. P(A = alpha) is handled by another function
    P_A_equals_alpha = percent_from_demographic(alpha, df_311, df_census)

    return expectation / P_A_equals_alpha
