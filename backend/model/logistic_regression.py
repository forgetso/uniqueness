from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from backend.db.layer import get_cue_response_pairs_coded
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
import hashlib
from collections import defaultdict


SAMPLE_SIZE = 8
TO_PREDICT = 'user_id'
MIN_RESPONSES = 50 * SAMPLE_SIZE
logging.basicConfig(level=LOG_LEVEL)
USER_COLUMNS = ['nativelanguage', 'gender', 'age', 'city', 'country', 'education']
C_PARAM_RANGE = [0.001, 0.01, 0.1, 1, 10, 100]
# Magic numbers with attractrepel
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
CATEGORICAL_LIMIT = 20
AGE_BRACKETS = [(0, 25), (60, 100)]


# AGE_BRACKETS = [(0, 20), (21, 30), (31, 40), (41, 50), (51, 60)]


def main(predict=None, sample_size=None, source=None, columns=None, k_value=None, reload=None, exclude=None,
         model=None):
    X, y, X_train, y_train, X_test, y_test, label_encoder = prepare(predict, sample_size, source, columns, reload,
                                                                     exclude, k_value)

    accuracy_cols = ['Model', 'C_parameter', 'Accuracy']
    accuracies = pd.DataFrame(columns=accuracy_cols)
    accuracies['C_parameter'] = C_PARAM_RANGE
    accuracy_row = 0
    best_model = None
    best_score = 0
    model_fn = get_model_fn(model)
    for c_param in C_PARAM_RANGE:
        logging.debug(model_fn)
        logmodel, y_predictions = model_fn(X_train, y_train, X_test, y_test, c_param, label_encoder)
        # Saving accuracy score in table
        score = accuracy_score(y_test, y_predictions)
        if score >= best_score:
            best_model = logmodel
        accuracies.iloc[accuracy_row, 2] = score
        accuracies.iloc[accuracy_row, 0] = 'logistic'
        accuracy_row += 1
    logging.debug('Naive Bayes')
    logmodel, y_predictions = naiveBayes(X_train, y_train, X_test, y_test)
    # Saving accuracy score in table
    accuracies = accuracies.append(
        pd.DataFrame([['naive-bayes', None, accuracy_score(y_test, y_predictions)]], columns=accuracy_cols))
    logging.debug(accuracies)
    return X, y, best_model


def prepare(predict, sample_size, source, columns, reload, exclude, one_hot=True):
    if predict is None:
        predict = TO_PREDICT

    try:
        sample_size = int(sample_size)
    except TypeError:
        sample_size = SAMPLE_SIZE

    logging.debug('Getting data')

    min_responses = MIN_RESPONSES

    # filename can get too long
    coded_file = get_coded_filename(source, min_responses, columns)

    if not Path.is_file(coded_file) or reload:
        df, _user_data = getData(source=source, min_responses=min_responses, columns=columns, predict=predict)
        # create more features to try and increase the accuracy
        df = feature_creation(df)

        # avoid calling the DB every time
        df.to_csv(coded_file.absolute(), index=False)
    else:
        df = pd.read_csv(coded_file)

        try:
            df.drop(columns=['Unnamed: 0'], inplace=True, axis=1)
        except KeyError:
            pass
    if df is None:
        raise (ValueError('No data returned from getData!'))

    df = df.drop(
        columns=['cue_clean', 'response_clean',
                 'cue_raw', 'response_raw', 'source',
                 'cue', 'response', 'cue_lstem', 'cue_sstem',
                 'response_lstem', 'response_sstem'])

    if 'index' in df.columns:
        df.drop('index', inplace=True, axis=1)

    # filter out empty columns and apply any user defined filtering
    logging.debug(
        'filtering columns to remove constant columns and perform requested filtering: {}'.format(columns))
    logging.debug('columns left in dataframe: {}'.format(len(df.columns)))

    # encode any string columns here
    cat_cols = df.select_dtypes(include='object').columns
    d = defaultdict(preprocessing.LabelEncoder)
    df[cat_cols] = df[cat_cols].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
    df = filterData(df, predict, columns, exclude)

    df = balance_samples(df, predict)
    logging.debug(df[predict].value_counts())
    cat_bool_cols = get_category_boolean_columns(df, predict)

    X, X_train, X_test, y, y_train, y_test = prep_data(df, predict, sample_size, cat_bool_cols)

    # split out category columns into one-hot columns
    if one_hot:
        X_train = pd.get_dummies(X_train, columns=cat_bool_cols)
        X_test = pd.get_dummies(X_test, columns=cat_bool_cols)

    # following line just keeps the sample size sane for SVM
    X_train_random_indices = np.random.choice(X_train.shape[0], size=10000, replace=False).tolist()
    X_test_random_indices = np.random.choice(X_test.shape[0], size=10000, replace=False).tolist()
    X_train = X_train.iloc[X_train_random_indices, :]
    X_test = X_test.iloc[X_test_random_indices, :]
    y_train = y_train.iloc[X_train_random_indices]
    y_test = y_test.iloc[X_test_random_indices]
    return X, y, X_train, y_train, X_test, y_test, d


def balance_samples(df, predict):
    vc = df[predict].value_counts()
    factor = 2
    # check if there is a sample that has more than 3 times what the other samples have
    small_samples = (df[predict].value_counts().max() / vc >= factor).sum()
    while small_samples > 0:
        max_category = vc.index[vc == vc.max()].values.tolist()[0]
        items = df.index[df[predict] == max_category]
        # use the second highest count
        keep = np.random.choice(items, vc[1], replace=False)
        remove = items[~items.isin(keep)]
        df = df[~df.index.isin(remove)]
        vc = df[predict].value_counts()
        small_samples = (df[predict].value_counts().max() / vc >= factor).sum()
    return df


def get_model_fn(model):
    fn = None
    if model == 'logistic':
        fn = logisticRegression
    if model == 'svm':
        fn = svc
    if model == 'bayes':
        fn = naiveBayes
    return fn


def get_category_boolean_columns(df, predict):
    unique_vals = df[df.dtypes[(df.dtypes == np.float) | (df.dtypes == np.int)].index.values].nunique()
    cat_bool_cols = unique_vals[unique_vals <= CATEGORICAL_LIMIT].index.tolist()
    try:
        cat_bool_cols.remove(predict)
    except ValueError:
        pass
    return cat_bool_cols


def prep_data(df, predict, sample_size, cat_bool_cols):
    continuous_cols = [x for x in df.columns if x not in cat_bool_cols and x != predict]

    logging.info('Predicting for {}'.format(predict))
    # the data to be predicted
    y = df[predict]
    X = df.drop(predict, axis=1)

    # get rid of np.nan and np.inf etc.
    X = pd.DataFrame(np.nan_to_num(X), columns=X.columns)
    y = pd.Series(np.nan_to_num(y), name=predict)

    # split the data before replacing NaNs with mean to avoid leaking information into the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    # fill nans with the mean
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0

    # take the log of the data
    X_train = log_transform(X_train, continuous_cols)
    X_test = log_transform(X_test, continuous_cols)

    # feature normalization
    X_train = feature_normalization(X_train, continuous_cols)
    X_test = feature_normalization(X_test, continuous_cols)

    if sample_size > 1:
        exit('not implemented')

    logging.debug('including features: {}'.format(X_train.columns))

    return X, X_train, X_test, y, y_train, y_test


def get_coded_filename(source, min_responses, columns):
    hash = hashlib.md5('{source}_{min_responses}_{columns}'.format(source=source,
                                                                   min_responses=min_responses,
                                                                   columns=columns).encode('utf-8')).hexdigest()
    return Path(os.path.join(DATA_FOLDER, 'coded_data_{}.csv'.format(hash)))


def log_transform(df, continuous_cols):
    # https://statmodeling.stat.columbia.edu/2019/08/21/you-should-usually-log-transform-your-positive-data/
    n = int(0.05 * len(df))
    for col in continuous_cols:
        # make sure there are no negatives in the column
        if np.all(df[col] >= 0):
            # check ratio between the highest and the lowest 5% is > 1.1 (close to 1 implies not much variance)
            if np.divide(df.nsmallest(n, col).mean(), df.nlargest(n, col).mean())[col] > 1.1:
                # TODO this doesn't really work for length, syllables, frequency
                #   they should instead use something like the below
                #   [length, syllables].apply(lambda x:abs(0.5*np.log((2+x**5)/(2-x**5))))
                #   [freq].apply(lambda x:abs(0.5*np.log((2+x**0.05)/(2-x**0.05))))
                # df[col] = abs(np.log(1 / (1 + df[col] ** 0.5)))
                df[col] = np.log(1 + df[col])

    df = df.replace([np.inf, -np.inf], 0)
    return df


def remove_outliers(df):
    # remove any outliers more than + or -3 times the standard deviation of mean, for each column
    from scipy import stats
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    return df


def get_boolean_columns(df):
    bool_cols = []
    for column in df.columns:
        if df[column].isin([0, 1]).all():
            bool_cols.append(column)
    return bool_cols


def feature_creation(df):
    # TODO move all of this into coding OR a separate process between coding and modelling
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

    multiply = ['concreteness_measure', ('valence', 'arousal', 'dominance'), 'age_of_acquisition', 'lg10wf']
    c_multiply = ['cue_{}'.format(x) for x in multiply]
    r_multiply = ['response_{}'.format(x) for x in multiply]

    # Norms of valence, arousal, and dominance for 13,915 English lemmas
    # Valence arousal and dominance interact with concreteness, age of acquisition, and frequency
    # By multiplying we will hopefully increase the visibility of these interactions
    # https://link.springer.com/article/10.3758/s13428-012-0314-x
    # for m1 in c_multiply:
    #     for m2 in r_multiply:
    #         if m1 == m2:
    #             continue
    #         if isinstance(m1, str):
    #             if isinstance(m2, str):
    #                 col = '{}.{}'.format(m1, m2)
    #                 if col not in df.columns:
    #                     df['{}.{}'.format(m1, m2)] = df[m1] * df[m2]
    #             if isinstance(m2, tuple):
    #                 for m3 in m2:
    #                     col = '{}.{}'.format(m1, m3)
    #                     if col not in df.columns:
    #                         df[col] = df[m1] * df[m3]

    df = df.drop(columns=gaussian_cols)

    # TODO add test for character intersection and probably move it to it's own function
    df["cue_chars"] = df["cue_chars"].apply(set)
    df["response_chars"] = df["response_chars"].apply(set)
    df["intersection"] = df.apply(
        lambda row: set.intersection(row["cue_chars"], row["response_chars"]), axis=1
    )
    df["intersection"] = df["intersection"].apply(lambda x: list(x))
    df["char_intersect"] = df["intersection"].str.len() / df["cue_clean"].str.len()
    df = df.drop(columns=["cue_chars", "response_chars", "intersection"])

    result = df.drop(columns=['dict2vec_similarity', 's2v_similarity', 'wikipedia_similarity', 'crawl_similarity'])

    return result


def gaussian(x, mean, sd):
    # return norm(mu, sig).pdf(x)
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom
    # return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def feature_normalization(df, cols_to_normalize):
    # TODO which type of scaling to use here?
    old_index = df.index
    df.reset_index(inplace=True, drop=True)
    cols = [c for c in cols_to_normalize if c in df.columns]
    x = df[cols]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    n_test = pd.DataFrame(x_scaled, columns=cols)
    l_test = df.drop(cols, axis=1)
    result = pd.concat([n_test, l_test], axis=1)
    result.index = old_index
    return result


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


def logisticRegression(x_train, y_train, x_test, y_test, C_param, label_encoders):
    # create an instance and fit the model
    logmodel = LogisticRegression(C=C_param, max_iter=2000)

    logmodel.fit(x_train, y_train)
    # predictions
    predictions = logmodel.predict(x_test)
    label_index = [int(x) for x in logmodel.classes_.tolist()]
    # target_strings = [str(x) for x in label_encoders[predict].inverse_transform(label_index)]
    print(classification_report(y_test, predictions))  # , target_names=target_strings))
    return logmodel, predictions


def svc(x_train, y_train, x_test, y_test, C_param, label_encoders):
    # create an instance and fit the model
    # The implementation is based on libsvm.
    # The fit time scales at least quadratically with the number of samples and
    # may be impractical beyond tens of thousands of samples.

    svmmodel = svm.SVC(C=C_param)

    svmmodel.fit(x_train, y_train)
    # predictions
    predictions = svmmodel.predict(x_test)
    label_index = [int(x) for x in svmmodel.classes_.tolist()]
    # target_strings = [str(x) for x in label_encoders[predict].inverse_transform(label_index)]
    print(classification_report(y_test, predictions))  # , target_names=target_strings))
    return svmmodel, predictions


def naiveBayes(x_train, y_train, x_test, y_test):
    # create an instance and fit the model
    model = GaussianNB()
    model.fit(x_train, y_train)
    # predictions
    predictions = model.predict(x_test)
    label_index = [int(x) for x in model.classes_.tolist()]
    # target_strings = [str(x) for x in label_encoders[predict].inverse_transform(label_index)]
    print(classification_report(y_test, predictions))  # , target_names=target_strings))
    return model, predictions


def filterData(df, predict, columns=None, exclude=None):
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

    if exclude:
        # only try and exclude columns that exist
        exclude = [e for e in exclude.split(',') if e in df.columns]
        df.drop(columns=exclude, inplace=True)
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
    df = df.replace('NaN', np.nan)
    df = df.apply(pd.to_numeric, errors='ignore')
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

    user_columns = USER_COLUMNS
    if predict in USER_COLUMNS:
        user_columns.remove(predict)

    if TO_PREDICT != predict:
        df.drop(TO_PREDICT, inplace=True, axis=1)

    user_data = df[user_columns]
    df = df.drop(columns=user_columns)

    # converts True/False to 1/0
    df = df * 1
    # drop any columns with only zeroes
    df = df.loc[:, (df != 0).any(axis=0)]

    df = df.apply(pd.to_numeric, errors='ignore')

    # if keep_pairs:
    #     df = pd.concat(df, user_data, axis=1)
    df = df.sort_values(by=predict)

    return df, user_data


def batch_data(df, predict, size):
    # GROUP BY variable to predict
    # GROUP BY n rows and calculate mean
    logging.info('There are {} samples to batch'.format(len(df)))
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


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    parser.add_argument('--predict', default=None, help="choose variable to predict")
    parser.add_argument('--source', default=None, help="code responses from a particular source only")
    parser.add_argument('--samplesize', default=None, help="choose the number of responses to batch as sample")
    parser.add_argument('--columns', default=None, help="predict based on specific columns")
    parser.add_argument('--k_value', default=None, help="select K best features for regression", type=int)
    parser.add_argument('--reload', default=None, action="store_true", help="reload data from the DB")
    parser.add_argument('--exclude', default=None, help="exclude specific columns")
    parser.add_argument('--model', default='logistic', help="type of model to run")
    args = parser.parse_args()
    main(predict=args.predict, source=args.source, sample_size=args.samplesize, columns=args.columns,
         k_value=args.k_value, reload=args.reload, model=args.model)


if __name__ == "__main__":
    setup()
