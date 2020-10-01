from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SAMPLE_SIZE = 8
TO_PREDICT = 'user_id'
MIN_RESPONSES = 3 * SAMPLE_SIZE
logging.basicConfig(level=LOG_LEVEL)
USER_COLUMNS = ['nativelanguage', 'gender', 'age', 'city', 'country', 'education']
C_param_range = [0.001, 0.01, 0.1, 1, 10, 100]
#C_param_range = [0.1]


def main(predict=None, sample_size=None, source=None, columns=None, k_value=None, reload=None):
    if predict is None:
        predict = TO_PREDICT
    try:
        sample_size = int(sample_size)
    except TypeError:
        sample_size = SAMPLE_SIZE
    start = time.perf_counter()
    logging.debug('Getting data')

    min_responses = 50 * sample_size
    label_encoder = preprocessing.LabelEncoder()

    coded_file = Path(
        os.path.join(DATA_FOLDER, 'coded_data_{source}_{min_responses}_{columns}.csv'.format(source=source,
                                                                                             min_responses=min_responses,
                                                                                             columns=columns)))
    if not Path.is_file(coded_file) or reload:
        data = getData(source=source, min_responses=min_responses, columns=columns, label_encoder=label_encoder,
                       predict=predict)
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

    # feature normalization
    x = batched_data.drop(predict, axis=1)
    x_normalized = feature_normalization(x)

    # split the data before replacing NaNs with mean to avoid leaking information into the test set
    x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.30, random_state=101)

    # replace nans with zeros and inf with very large numbers
    x_train = replace_nan_with_mean(x_train)
    x_test = replace_nan_with_mean(x_test)

    # Principal Components Analysis
    # TODO This could be done with Incremental PCA in sklearn and then it would also handle the batching
    pca_train = PCA(n_components=20)
    pca_train.fit(x_test)
    x_train = pca_train.transform(x_train)

    pca_test = PCA(n_components=20)
    pca_test.fit(x_test)
    x_test = pca_test.transform(x_test)

    # create combined arrays for X and Y
    x_combined = np.vstack((x_train, x_test))
    y_combined = np.hstack((y_train, y_test))

    # feature selection
    x_train, col_indices = feature_selection(x_train, y_train, k_value=k_value)
    x_test = x_test[:, col_indices]
    x_combined = x_combined[:, col_indices]

    # get the column/feature names that were included
    df_k_best = batched_data.iloc[:, col_indices]
    # df_k_best = batched_data

    logging.debug('including features: {}'.format(df_k_best.columns))

    # logistic regression
    accuracies = pd.DataFrame(columns=['C_parameter', 'Accuracy'])
    accuracies['C_parameter'] = C_param_range
    accuracy_row = 0
    for c_param in C_param_range:
        logmodel, y_predictions = logisticRegression(x_train, y_train, x_test, y_test, c_param, label_encoder)
        # Saving accuracy score in table


        accuracies.iloc[accuracy_row, 1] = accuracy_score(y_test, y_predictions)
        accuracy_row += 1

        # Printing decision regions
        # plt.subplot(3, 2, accuracy_row)
        # plt.subplots_adjust(hspace=0.4)
        # print(x_combined)
        # plot_decision_regions(X=x_combined
        #                       , y=y_combined
        #                       , classifier=logmodel
        #                       , test_idx=range(105, 150))
        # plt.xlabel('Sepal length')
        # plt.ylabel('Sepal width')
        # plt.title('C = %s' % c_param)
        # plt.show()
    print(accuracies)
    end = time.perf_counter()
    logging.debug('time taken {}'.format(end - start))
    return logmodel


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
    #x = preprocessing.scale(df)
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
    bracket = None
    if age < 25:
        bracket = '<25'
    elif 25 <= age < 35:
        bracket = '25-34'
    elif 35 <= age < 45:
        bracket = '35-44'
    elif 45 <= age < 55:
        bracket = '45-54'
    elif 55 <= age < 65:
        bracket = '55-64'
    elif age >= 65:
        bracket = '>=65'
    return bracket


def getData(label_encoder, predict, source=None, min_responses=None, columns=None):
    # TODO make this multiprocess the formatting
    min_responses = min_responses or MIN_RESPONSES

    cur = get_cue_response_pairs_coded(source=source)
    df = pd.DataFrame([x for x in cur], columns=[desc[0] for desc in cur.description])
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
    df['age_bracket'] = label_encoder.fit_transform(df['age_bracket'])

    df[predict] = label_encoder.fit_transform(df[predict])
    df = df.astype(float)
    df = df.sort_values(by=predict)
    logging.debug('filtering columns to remove constant columns and perform requested filtering: {}'.format(columns))
    df = filterData(df, predict, columns)
    logging.debug('columns left in dataframe {}'.format(sorted(df.columns)))
    return df


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


def drop_wordnet_features(df):
    df = df.drop(columns=['derivationally_related_forms', 'pertainyms',
                          'instance_hypernyms', 'hyponyms', 'instance_hyponyms',
                          'member_holonyms', 'substance_holonyms', 'part_holonyms',
                          'member_meronyms', 'substance_meronyms', 'part_meronyms',
                          'topic_domains', 'region_domains', 'usage_domains', 'attributes',
                          'entailments', 'causes', 'also_sees', 'verb_groups', 'similar_tos',
                          'hypernyms', 'root_hypernyms'])
    return df


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Initialise the marker types and colors
    markers = ('s', 'x', 'o', '^', 'v', '4')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    color_Map = ListedColormap(colors[:len(np.unique(y))])  # we take the color mapping correspoding to the
    # amount of classes in the target data

    # Parameters for the graph and decision surface
    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contour(xx1, xx2, Z, alpha=0.4, cmap=color_Map)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot samples
    X_test, Y_test = X[test_idx, :], y[test_idx]

    #print(len(y), X.shape)
    for idx, cl in enumerate(np.unique(y)):
        print(idx)
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=color_Map(idx),
                    marker=markers[idx], label=cl
                    )


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
