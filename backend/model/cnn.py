from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import argparse
import numpy as np
import time
import logging
from backend.settings import LOG_LEVEL
from pathlib import Path
from backend.model.logistic_regression import getData, feature_creation, \
    filterData, SAMPLE_SIZE, TO_PREDICT, MIN_RESPONSES, C_param_range, \
    get_coded_filename, prep_data
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, Conv2D, Activation
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

    coded_file = get_coded_filename(source, min_responses, columns)

    if not Path.is_file(coded_file) or reload:
        df, label_encoder = getData(source=source, min_responses=min_responses, columns=columns,
                                    label_encoder=label_encoder,
                                    predict=predict)
        # create more features to try and increase the accuracy
        df = feature_creation(df)

        # filter out empty columns and apply any user defined filtering
        df = filterData(df, predict, columns)

        # avoid calling the DB every time
        df.to_csv(coded_file.absolute())
    else:
        df = pd.read_csv(coded_file)
    if df is None:
        raise (ValueError('No data returned from getData!'))

    X, X_train, X_test, y, y_train, y_test = prep_data(df, predict, sample_size, k_value)

    # logistic regression
    accuracy_cols = ['Model', 'C_parameter', 'Accuracy']
    accuracies = pd.DataFrame(columns=accuracy_cols)
    accuracies['C_parameter'] = C_param_range
    accuracy_row = 0

    accuracy = keras_cnn(X_train, X_test, y_train, y_test)
    print(accuracy)

    end = time.perf_counter()
    logging.debug('time taken {}'.format(end - start))
    return


def keras_cnn(x_train, x_test, y_train, y_test):
    from keras.constraints import maxnorm
    from keras.optimizers import SGD
    trainX = np.dstack(x_train).T
    trainy = np.vstack(y_train)
    testX = np.dstack(x_test).T
    testy = np.vstack(y_test)
    print(testX.shape, trainX.shape, trainy.shape, testy.shape)
    verbose, epochs, batch_size = 0, 10, 1
    n_records, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Dense(512, input_shape=(n_records, n_features), activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(n_outputs, activation='sigmoid'))
    sgd = SGD(lr=0.1, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
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
