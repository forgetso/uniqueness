import os
from inspect import currentframe
import logging


# helpers for debugging dataframes
def dfdebug(df, column, search_term, line_no):
    if column in df.columns:
        if len(df.loc[df[column] == search_term]):
            logging.debug('process ID {} line number {} contains search term {}'.format(os.getpid(), line_no, search_term))
            logging.debug(df.loc[df[column] == search_term])
        else:
            logging.debug('process ID {} line number {} MISSING search term {} in COLUMN'.format(os.getpid(), line_no, search_term))
            pass
    else:
        try:
            location = df.index.get_loc(search_term)
            logging.debug(
                'process ID {} line number {} contains search term {} in the index'.format(os.getpid(), line_no, search_term))
            logging.debug('present on row {}'.format(str(location)))
        except KeyError:
            logging.debug('process ID {} line number {} MISSING search term {}'.format(os.getpid(), line_no, search_term))
            pass


def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno
