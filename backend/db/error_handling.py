# import sys to get more detailed Python exception info
import sys

# import the connect library for psycopg2
from psycopg2 import connect

# import the error handling libraries for psycopg2
from psycopg2 import OperationalError, errorcodes, errors

# import the psycopg2 library's __version__ string
from psycopg2 import __version__ as psycopg2_version
# !/usr/bin/python3
# -*- coding: utf-8 -*-

# import sys to get more detailed Python exception info
import sys
import logging


logging.basicConfig(level=logging.DEBUG)

# import the psycopg2 library's __version__ string
from psycopg2 import __version__ as psycopg2_version

# print the version string for psycopg2
logging.info('{}{}'.format("psycopg2 version:", psycopg2_version))

# define a function that handles and parses psycopg2 exceptions
def print_psycopg2_exception(err):
    # get details about the exception
    err_type, err_obj, traceback = sys.exc_info()

    # get the line number when exception occured
    line_num = traceback.tb_lineno

    # print the connect() error
    logging.error('{}{}{}{}'.format("psycopg2 ERROR:", err, "on line number:", line_num))
    logging.error('{}{}{}{}'.format("psycopg2 traceback:", traceback, "-- type:", err_type))

    # psycopg2 extensions.Diagnostics object attribute
    logging.error(''.join(["extensions.Diagnostics:", err.diag]))

    # print the pgcode and pgerror exceptions
    logging.error(''.join(["pgerror:", err.pgerror]))
    logging.error(''.join(["pgcode:", err.pgcode]))
