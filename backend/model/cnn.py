from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import argparse
import numpy as np
import time
import logging
from backend.settings import LOG_LEVEL, DATA_FOLDER
from pathlib import Path
import os
from backend.model.logistic_regression import getData, feature_creation, feature_normalization, feature_selection, \
    filterData, batch_data, replace_nan_with_mean, SAMPLE_SIZE, TO_PREDICT, MIN_RESPONSES, C_param_range
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
logging.basicConfig(level=LOG_LEVEL)


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

    accuracy = keras_cnn(x_train, x_test, y_train, y_test)
    print(accuracy)

    end = time.perf_counter()
    logging.debug('time taken {}'.format(end - start))
    return


def keras_cnn(x_train, x_test, y_train, y_test):
    trainX = np.dstack(x_train).T
    trainy = np.vstack(y_train)
    testX = np.dstack(x_test).T
    testy = np.vstack(y_test)
    print(testX.shape, trainX.shape, trainy.shape, testy.shape)
    verbose, epochs, batch_size = 0, 10, 1
    n_records, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_records, n_features)))
    #model.add(Dropout(0.5))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    #model.add(Dense(200, activation='relu'))
    #model.add(Dense(n_outputs, activation='sigmoid'))
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # answers are either 0 or 1
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy


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
