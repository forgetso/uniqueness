import pandas as pd
from backend.settings import NGRAMS_DATA
import numpy as np
import logging
from ast import literal_eval

NGRAM_PERIODS = [10, 20, 30, 40]


class Ngrams:

    def __init__(self, words=None):
        ngrams_data = pd.read_csv(NGRAMS_DATA.absolute())
        to_drop = [x for x in ['Unnamed: 0', 'parent', 'type'] if x in ngrams_data.columns]
        ngrams_data.drop(columns=to_drop, inplace=True, axis=1)
        if words is not None:
            ngrams_data = ngrams_data[ngrams_data['word'].isin(words)]
        self.data = ngrams_data.drop_duplicates()
        ngrams_data.reset_index(inplace=True)

    def subset(self, words, col):
        coldf = pd.DataFrame(
            words
        ).merge(
            self.data,
            left_on=col,
            right_on="ngram",
            how="left",
        ).rename(columns={'timeseries': '{}_timeseries'.format(col.split('_')[0])}).drop(['ngram'], axis=1)
        return coldf


def calculate_ts_angle(ts1, ts2, period=10):
    angle = np.nan
    if ts1 is not np.nan and ts2 is not np.nan:
        vector1 = calculate_moving_average_vector(ts1, period=period)
        vector2 = calculate_moving_average_vector(ts2, period=period)
        angle = calculate_cosine_angle(vector1, vector2)

    return angle


def calculate_cosine_angle(vector1, vector2):
    try:
        return np.rad2deg(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))
    except RuntimeWarning as e:
        return np.nan


def calculate_moving_average_vector(ts, period=10):
    arr = np.array(ts)
    if len(arr) < period + 1:
        raise ValueError('Array is too short to calculate moving average with {} periods'.format(period))
    # e.g. if period is 10 years vector is (avg(2000-2009), avg(2010-2019))
    vector = (np.average(arr[:-period][-period:]), np.average(arr[:-period]))
    return vector


def literal_return(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val


def calculate_min_and_max_yearly_diffs(df):
    diffcol = 'ngram_diff'
    df[diffcol] = df.apply(lambda x: np.subtract(x['cue_timeseries'], x['response_timeseries']), axis=1)
    # TODO work out how to get this to be an int instead of a list
    df['{}_min_idx'.format(diffcol)] = df[diffcol].apply(min_index)
    df['{}_max_idx'.format(diffcol)] = df[diffcol].apply(max_index)
    return df


def min_index(arr):
    try:
        return list(arr).index(np.min(arr))
    except (ValueError, TypeError):
        return np.nan


def max_index(arr):
    try:
        return list(arr).index(np.max(arr))
    except (ValueError, TypeError):
        return np.nan


# TODO what if there are loads of crossovers, only want significant ones
# def crossover(arr):
#     sign = np.sign(arr)
#     item_count = 0
#     xovers = []
#     while item_count < len(sign):
#         to_check = sign[item_count:]
#         ne = np.where(to_check != to_check[item_count])
#         if len(ne[0]) == len(to_check) or len(ne[0]) == 0:
#             print('no crossover')
#             break
#         else:
#             xovers.append(ne[0][0])
#             item_count += 1
#     return xovers


def attach_ngrams(df):
    ngrams = Ngrams()
    ngrams_df = ngrams.data

    ngrams_df['ngram'] = ngrams_df['ngram'].astype(str)
    ngrams_df['timeseries'] = ngrams_df['timeseries'].apply(literal_return)

    # get a subset of the ngrams, one set for the cues and one for the responses and then join the two sets together
    # TODO make this work with multi word responses
    cue_ngrams = ngrams.subset(df.index.get_level_values("cue_clean").str.decode("utf8"), 'cue_clean')
    response_ngrams = ngrams.subset(df.index.get_level_values("response_clean").str.decode("utf8"), 'response_clean')
    ngrams_df = pd.concat((cue_ngrams, response_ngrams), axis=1)

    ngrams_df['cue_clean'] = ngrams_df['cue_clean'].astype(str).str.encode('utf8')
    ngrams_df['response_clean'] = ngrams_df['response_clean'].astype(str).str.encode('utf8')
    ngrams_df.set_index(['cue_clean', 'response_clean'], inplace=True)
    df = df.join(ngrams_df)

    # calculate the angle between the two ngram timeseries
    for period in NGRAM_PERIODS:
        angle_col = 'ngram_angle_{}'.format(period)
        df[angle_col] = df.apply(lambda x: calculate_ts_angle(x['cue_timeseries'], x['response_timeseries'], period),
                                 axis=1)

    # work out when cue is most and least different in popularity from the response
    df = calculate_min_and_max_yearly_diffs(df)

    # TODO mean of ngram diffs? crossover points? etc.

    df.drop(['cue_timeseries', 'response_timeseries', 'ngram_diff'], inplace=True, axis=1)
    return df
