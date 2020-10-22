from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from backend.db.db import get_cue_response_pairs_coded
import pandas as pd
import argparse
import numpy as np
import time
import logging
from backend.settings import LOG_LEVEL, DATA_FOLDER
from sklearn.feature_selection import VarianceThreshold
from pathlib import Path
import os
import math

SAMPLE_SIZE = 8
TO_PREDICT = 'user_id'
MIN_RESPONSES = 50 * SAMPLE_SIZE
logging.basicConfig(level=LOG_LEVEL)
USER_COLUMNS = ['nativelanguage', 'gender', 'age', 'city', 'country', 'education']
C_param_range = [0.001, 0.01, 0.1, 1, 10, 100]
# Magic numbers
# Without attractrepel
# CATEGORIES = [
#     {'name': 'synonym', 'mean': 0.39497, 'stdev': 0.09099, 'ratio': 12.57},
#     {'name': 'lexical_set', 'mean': 0.36623, 'stdev': 0.08235, 'ratio': 21.98},
#     {'name': 'other_conceptual', 'mean': 0.26388, 'stdev': 0.08276, 'ratio': 20.50},
#     {'name': 'cue_response_collocation', 'mean': 0.27928, 'stdev': 0.08203, 'ratio': 21.02},
#     {'name': 'response_cue_collocation', 'mean': 0.29624, 'stdev': 0.08674, 'ratio': 14.95},
#     {'name': 'affix_manipulation', 'mean': 0.57684, 'stdev': 0.05269, 'ratio': 1.53},
#     {'name': 'form_based', 'mean': 0.20946, 'stdev': 0.08244, 'ratio': 1.24},
#     {'name': 'two_step', 'mean': 0.21379, 'stdev': 0.07663, 'ratio': 3.11},
#     {'name': 'erratic', 'mean': 0.16325, 'stdev': 0.09704, 'ratio': 3.11},
# ]
# With attractrepel
CATEGORIES = [
    {'name': 'synonym', 'mean': 0.40205, 'stdev': 0.08037, 'ratio': 12.57},
    {'name': 'lexical_set', 'mean': 0.34135, 'stdev': 0.09045, 'ratio': 21.98},
    {'name': 'other_conceptual', 'mean': 0.23517, 'stdev': 0.09622, 'ratio': 20.50},
    {'name': 'cue_response_collocation', 'mean': 0.24524, 'stdev': 0.10412, 'ratio': 21.02},
    {'name': 'response_cue_collocation', 'mean': 0.26062, 'stdev': 0.10947, 'ratio': 14.95},
    {'name': 'affix_manipulation', 'mean': 0.57744, 'stdev': 0.04565, 'ratio': 1.53},
    {'name': 'form_based', 'mean': 0.18898, 'stdev': 0.08481, 'ratio': 1.24},
    {'name': 'two_step', 'mean': 0.18407, 'stdev': 0.09391, 'ratio': 3.11},
    {'name': 'erratic', 'mean': 0.13200, 'stdev': 0.10930, 'ratio': 3.11},
]
AGE_BRACKETS = [(0, 25), (50 , 120)]


def main(predict=None, sample_size=None, source=None, columns=None, k_value=None, reload=None):
    if predict is None:
        predict = TO_PREDICT

    try:
        sample_size = int(sample_size)
    except TypeError:
        sample_size = SAMPLE_SIZE
    start = time.perf_counter()
    logging.debug('Getting data')

    min_responses = MIN_RESPONSES
    label_encoder = preprocessing.LabelEncoder()

    coded_file = Path(
        os.path.join(DATA_FOLDER, 'coded_data_{source}_{min_responses}_{columns}.csv'.format(source=source,
                                                                                             min_responses=min_responses,
                                                                                             columns=columns)))
    if not Path.is_file(coded_file) or reload:
        data = getData(source=source, min_responses=min_responses, columns=columns, label_encoder=label_encoder,
                       predict=predict)
        # create more features to try and increase the accuracy
        data = feature_creation(data)

        # filter out empty columns and apply any user defined filtering
        data = filterData(data, predict, columns)

        # avoid calling the DB every time
        data.to_csv(coded_file.absolute())
    else:
        data = pd.read_csv(coded_file)
        data.drop(columns=['Unnamed: 0'], inplace=True, axis=1)
    if data is None:
        raise (ValueError('No data returned from getData!'))

    batched_data = data
    if sample_size > 1:
        batched_data = batch_data(data, predict, sample_size)

    logging.info('Predicting for {} using {}'.format(predict, batched_data.columns))
    # the data to be predicted
    y = batched_data[predict]
    x = batched_data.drop(predict, axis=1)

    # split the data before replacing NaNs with mean to avoid leaking information into the test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)

    # feature normalization

    x_train_norm = feature_normalization(x_train)
    x_test_norm = feature_normalization(x_test)

    # replace nans with zeros and inf with very large numbers
    x_train = replace_nan_with_mean(x_train_norm)
    x_test = replace_nan_with_mean(x_test_norm)

    # Principal Components Analysis
    # TODO This could be done with Incremental PCA in sklearn and then it would also handle the batching
    # pca_train = PCA(n_components=20)
    # pca_train.fit(x_test)
    # x_train = pca_train.transform(x_train)
    #
    # pca_test = PCA(n_components=20)
    # pca_test.fit(x_test)
    # x_test = pca_test.transform(x_test)

    # create combined arrays for X and Y
    x_combined = np.vstack((x_train, x_test))
    y_combined = np.hstack((y_train, y_test))

    # feature selection
    if k_value is not None:
        x_train, col_indices = feature_selection(x_train, y_train, k_value=k_value)
        x_test = x_test[:, col_indices]
        x_combined = x_combined[:, col_indices]
        # get the column/feature names that were included
        # TODO this is not right as predict column was dropped
        df_k_best = batched_data.iloc[:, col_indices]
    else:
        df_k_best = batched_data

    logging.debug('including features: {}'.format(df_k_best.columns))

    # logistic regression
    accuracy_cols = ['Model', 'C_parameter', 'Accuracy']
    accuracies = pd.DataFrame(columns=accuracy_cols)
    accuracies['C_parameter'] = C_param_range
    accuracy_row = 0
    for c_param in C_param_range:
        logging.debug('Logistic Regression')
        logmodel, y_predictions = logisticRegression(x_train, y_train, x_test, y_test, c_param, label_encoder)

        # Saving accuracy score in table
        accuracies.iloc[accuracy_row, 2] = accuracy_score(y_test, y_predictions)
        accuracies.iloc[accuracy_row, 0] = 'logistic'
        accuracy_row += 1
    logging.debug('Naive Bayes')
    logmodel, y_predictions = naiveBayes(x_train, y_train, x_test, y_test)
    # Saving accuracy score in table
    accuracies = accuracies.append(
        pd.DataFrame([['naive-bayes', None, accuracy_score(y_test, y_predictions)]], columns=accuracy_cols))

    print(accuracies)
    end = time.perf_counter()
    logging.debug('time taken {}'.format(end - start))
    return logmodel


def feature_creation(df):
    # (0.1, 0.2) band reduces the accuracy
    distance_bands = ((0.2, 0.3), (0.301, 0.4), (0.401, 1))
    for distance_band in distance_bands:
        bands = [{'distance_lower': distance_band[0], 'distance_upper': distance_band[1], 'upper': 1, 'lower': 0.301},
                 {'distance_lower': distance_band[0], 'distance_upper': distance_band[1], 'upper': 0.3, 'lower': 0.201},
                 {'distance_lower': distance_band[0], 'distance_upper': distance_band[1], 'upper': 0.2, 'lower': 0.101},
                 {'distance_lower': distance_band[0], 'distance_upper': distance_band[1], 'upper': 0.1, 'lower': -1},
                 ]
        for idx, band in enumerate(bands):
            df['band_{}_{}_{}'.format(distance_band[0], distance_band[1], idx)] = np.logical_and(
                np.logical_or.reduce((
                    (df['dict2vec_similarity'].subtract(df['s2v_similarity'])).abs().between(
                        band['distance_lower'],
                        band['distance_upper']),
                    (df['dict2vec_similarity'].subtract(df['wikipedia_similarity'])).abs().between(
                        band['distance_lower'],
                        band['distance_upper']),
                    (df['dict2vec_similarity'].subtract(df['crawl_similarity'])).abs().between(
                        band['distance_lower'],
                        band['distance_upper'])
                )),
                df['dict2vec_similarity'].between(band['lower'], band['upper'])
            )

    df['model_average'] = df[
        ['dict2vec_similarity', 's2v_similarity', 'wikipedia_similarity', 'crawl_similarity']].mean(axis=1)

    gaussian_cols = []
    for category in CATEGORIES:
        cat_gauss = ''.join([category['name'], '_gaussian'])
        df[cat_gauss] = df['model_average'].apply(gaussian, args=(category['mean'], category['stdev']))
        df[cat_gauss] = df[cat_gauss] * category['ratio']
        gaussian_cols.append(cat_gauss)

    for category in CATEGORIES:
        cat_gauss = ''.join([category['name'], '_gaussian'])
        df[category['name']] = df[cat_gauss] / df[gaussian_cols].sum(axis=1)

    df = df.drop(columns=gaussian_cols)

    result = df.drop(columns=['dict2vec_similarity', 's2v_similarity', 'wikipedia_similarity', 'crawl_similarity'])

    return result


def gaussian(x, mean, sd):
    # return norm(mu, sig).pdf(x)
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom
    # return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def replace_nan_with_mean(arr):
    # Obtain mean of columns as you need, nanmean is convenient.
    col_mean = np.nanmean(arr, axis=0)

    # Find indices that you need to replace
    inds = np.where(np.isnan(arr))

    # Place column means in the indices. Align the arrays using take
    arr[inds] = np.take(col_mean, inds[1])
    return arr


def feature_normalization(df):
    x = preprocessing.robust_scale(df)
    # x = preprocessing.scale(df)
    return x


def feature_selection(x, y, k_value=None):
    if not k_value:
        k_value = 5
    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=k_value)
    # apply feature selection
    X_selected = fs.fit_transform(x, y)
    # get selected column indices
    col_indices = fs.get_support(indices=True)
    return X_selected, col_indices


def logisticRegression(x_train, y_train, x_test, y_test, C_param, label_encoder):
    # create an instance and fit the model
    logmodel = LogisticRegression(C=C_param)

    logmodel.fit(x_train, y_train)
    # predictions
    predictions = logmodel.predict(x_test)
    label_index = [int(x) for x in logmodel.classes_.tolist()]
    # target_strings = [str(x) for x in label_encoder.inverse_transform(label_index)]
    print(classification_report(y_test, predictions))  # , target_names=target_strings))
    return logmodel, predictions


def naiveBayes(x_train, y_train, x_test, y_test):
    # create an instance and fit the model
    model = GaussianNB()
    model.fit(x_train, y_train)
    # predictions
    predictions = model.predict(x_test)
    label_index = [int(x) for x in model.classes_.tolist()]
    # target_strings = [str(x) for x in label_encoder.inverse_transform(label_index)]
    print(classification_report(y_test, predictions))  # , target_names=target_strings))
    return model, predictions


def filterData(df, predict, columns=None):
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(df)
    constant_columns = [column for column in df.columns
                        if column not in
                        df.columns[constant_filter.get_support()]]
    if len(constant_columns):
        logging.debug('Following columns are CONSTANT {} '.format(constant_columns))

        logging.debug('Dropping columns')
        df = df.drop(constant_columns, axis=1)

    if columns:
        columns_list = columns.split(',')
        columns_list.append(predict)
        to_drop = [x for x in df.columns if x not in columns_list]
        logging.debug('Dropping {} from df - keeping {}'.format(to_drop, columns_list))
        df.drop(columns=to_drop, inplace=True)
        # df = df.loc[df['left_hemi_distance'].notnull()]
        # print(df)
        # df = df.dropna(how='all')
    return df


def get_age_bracket(age):
    result = None
    for bracket in AGE_BRACKETS:
        bracket_name = '-'.join([str(bracket[0]), str(bracket[1])])
        if bracket[0] < age <= bracket[1]:
            result = bracket_name
            break
    return result


def getData(label_encoder, predict, source=None, min_responses=None, columns=None, keep_pairs=None):
    # TODO make this multiprocess the formatting
    min_responses = min_responses or MIN_RESPONSES

    cur = get_cue_response_pairs_coded(source=source)
    df = pd.DataFrame([x for x in cur], columns=[desc[0] for desc in cur.description])
    if predict == 'age_bracket':
        df['age_bracket'] = df['age'].apply(get_age_bracket)
    df.drop(columns=['rowid'], inplace=True)
    logging.info('there are {} records'.format(len(df)))
    # full wordssoc table
    predict_group = df.groupby(by=predict).agg(['count'])
    logging.debug('min responses set to {}'.format(min_responses))
    min_responses_indices = predict_group['response_clean'].gt(min_responses)
    groups_with_min_responses = predict_group.index[min_responses_indices['count']].values
    if not len(groups_with_min_responses):
        exit('not enough responses')
    df = df[df[predict].isin(groups_with_min_responses)]
    if not keep_pairs:
        df = df.drop(
            columns=['cue_clean', 'response_clean',
                     'cue_raw', 'response_raw',
                     'cue_correction', 'response_correction',
                     'cue_sense', 'response_sense', 'source'])
    user_columns = USER_COLUMNS
    if predict in USER_COLUMNS:
        user_columns.remove(predict)

    if TO_PREDICT != predict:
        df.drop(TO_PREDICT, inplace=True, axis=1)

    df = df.drop(columns=user_columns)

    # converts True/False to 1/0
    df = df * 1
    # drop any columns with only zeroes
    df = df.loc[:, (df != 0).any(axis=0)]

    logging.debug('replacing nans with strings in str columns')
    df['phonology'] = df['phonology'].replace({np.nan: ''})
    df['syntagmatic'] = df['syntagmatic'].replace({np.nan: ''})
    logging.debug('encoding non-numeric columns')

    df['phonology'] = label_encoder.fit_transform(df['phonology'])
    df['syntagmatic'] = label_encoder.fit_transform(df['syntagmatic'])
    if predict == 'age_bracket':
        df['age_bracket'] = label_encoder.fit_transform(df['age_bracket'])

    df[predict] = label_encoder.fit_transform(df[predict])
    print(df)
    if not keep_pairs:
        df = df.astype(float)
    df = df.sort_values(by=predict)
    logging.debug('filtering columns to remove constant columns and perform requested filtering: {}'.format(columns))

    logging.debug('columns left in dataframe {}'.format(sorted(df.columns)))
    return df


def batch_data(df, predict, size):
    # GROUP BY variable to predict
    # GROUP BY n rows and calculate mean
    logging.info('There are {} samples to batch'.format(len(df)))
    print(df.columns)
    groups = {key: df.loc[value] for key, value in df.groupby(predict).groups.items()}
    batched = pd.DataFrame()
    logging.debug('batching data...')
    for key, group in groups.items():
        logging.debug('batching {}'.format(key))
        group.reset_index(drop=True, inplace=True)
        group = group.groupby(group.index // size).mean()
        logging.info('There are {} samples for key {}'.format(len(group), key))
        batched = pd.concat([batched, group])
    batched = batched.loc[:, ~df.columns.duplicated()]
    return batched


def drop_wordnet_features(df):
    df = df.drop(columns=['derivationally_related_forms', 'pertainyms',
                          'instance_hypernyms', 'hyponyms', 'instance_hyponyms',
                          'member_holonyms', 'substance_holonyms', 'part_holonyms',
                          'member_meronyms', 'substance_meronyms', 'part_meronyms',
                          'topic_domains', 'region_domains', 'usage_domains', 'attributes',
                          'entailments', 'causes', 'also_sees', 'verb_groups', 'similar_tos',
                          'hypernyms', 'root_hypernyms'])
    return df


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    parser.add_argument('--predict', default=None, help="choose variable to predict")
    parser.add_argument('--source', default=None, help="code responses from a particular source only")
    parser.add_argument('--samplesize', default=None, help="choose the number of responses to batch as sample")
    parser.add_argument('--columns', default=None, help="predict based on specific columns")
    parser.add_argument('--k_value', default=None, help="select K best features for regression", type=int)
    parser.add_argument('--reload', default=None, action="store_true", help="reload data from the DB")
    args = parser.parse_args()
    main(predict=args.predict, source=args.source, sample_size=args.samplesize, columns=args.columns,
         k_value=args.k_value, reload=args.reload)


if __name__ == "__main__":
    setup()
