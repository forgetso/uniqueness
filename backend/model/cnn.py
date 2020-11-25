from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import argparse
import numpy as np
import time
import logging
from backend.settings import LOG_LEVEL
from backend.model.logistic_regression import prepare, C_PARAM_RANGE
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, Conv2D, Activation
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
logging.basicConfig(level=LOG_LEVEL)


def main(predict=None, sample_size=None, source=None, columns=None, k_value=None, reload=None, exclude=None):
    X, y, X_train, y_train, X_test, y_test, label_encoder = prepare(predict, sample_size, source, columns, reload,
                                                                    exclude, k_value)

    # logistic regression
    accuracy_cols = ['Model', 'C_parameter', 'Accuracy']
    accuracies = pd.DataFrame(columns=accuracy_cols)
    accuracies['C_parameter'] = C_PARAM_RANGE
    accuracy_row = 0

    accuracy = keras_cnn(X_train, X_test, y_train, y_test)
    print(accuracy)

    end = time.perf_counter()
    return


def keras_cnn(X_train, X_test, y_train, y_test):
    from keras.constraints import maxnorm
    from keras.optimizers import SGD
    print(X_train.shape, X_test.shape)
    trainX = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    trainy = np.vstack(y_train)
    testX = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
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
    sgd = SGD(lr=0.01, momentum=0.9)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
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
