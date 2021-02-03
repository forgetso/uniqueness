import numpy as np


def unpack_lists(lst, length=None):
    # return a list of individual words from columns of word lists and include the original column index of the list
    unpacked = None
    # we need every item in the list to have the length attribute. empty list satisfies this condition
    #lst.replace(np.nan, [], inplace=True)
    for row in lst.loc[lst.isnull()].index:
        lst.at[row] = []
    if not length:
        length = 2
    try:
        unpacked = np.vstack([np.c_[x, [idx] * len(x)] for idx, x in enumerate(lst) if len(x) >= length])
    except (ValueError):
        # the array may be empty if there are no multi word strings, causing concatenate to fail with
        # "ValueError: need at least one array to concatenate"
        pass
    return unpacked
