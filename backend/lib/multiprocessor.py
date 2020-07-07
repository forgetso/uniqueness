import multiprocessing
import logging
import numpy as np
import pandas as pd
from math import ceil
from multiprocessing import shared_memory


# split a list into evenly sized chunks
def chunks(lst, n):
    if n < 1:
        n = 1
    result = [x for x in np.array_split(lst, n) if len(x)]
    return result


def amend_groups(slices, groupby):
    # Takes a PRE-SORTED list of dataframes and makes sure that each inner list is grouped according to groupby
    # If we have [[{'w': 'ar'}, {'w': 'at'}, {'w': 'az'}], [{'w': 'as'}, {'w': 'bc'}]]
    # and groupby equal to ['w', 0] we need to remove the 'as' record from the second internal list and
    # append it to the first giving [[{'w': 'ar'}, {'w': 'at'}, {'w': 'az'}, {'w': 'as'}], [{'w': 'bc'}]]
    slice_count = len(slices)
    assert isinstance(slices[0], pd.DataFrame)
    assert groupby[0] in slices[0].columns
    if slice_count > 1:
        for cur_slice_idx, _ in enumerate(slices):
            slice_x = slices[cur_slice_idx]
            slice_x_index = len(slice_x) - 1
            slice_x_value = get_groupby_val(slice_x.iloc[slice_x_index, :], groupby)
            if cur_slice_idx < slice_count - 1:
                nxt_slice_idx = cur_slice_idx + 1
                slice_y = slices[nxt_slice_idx]
                startswith_filter = slice_y[groupby[0]].str.startswith(slice_x_value)
                slice_x = slice_x.append(slice_y[startswith_filter])
                slice_y = slice_y.drop(slice_y[startswith_filter].index)
                slices[cur_slice_idx] = slice_x
                slices[nxt_slice_idx] = slice_y
                if len(slices[nxt_slice_idx]) == 0:
                    # move this slice to the start of the list to maintain the list length
                    slices = [slices[nxt_slice_idx]] + slices[:nxt_slice_idx] + slices[nxt_slice_idx + 1:]

    slices = [x for x in slices if len(x)]
    return slices


def get_groupby_val(slice_val, groupby):
    group_val = slice_val[groupby[0]]
    if len(groupby) > 1:
        for group in groupby[1:]:
            group_val = group_val[group]
    return group_val


def split_on(df, col, job_number):
    rows = df.index[df[col] == True].values
    # splitting on col
    df_len = len(df)
    lst = [df.iloc[row:rows[i + 1]] if i + 1 < len(rows) else df.iloc[row:] for i, row in enumerate(rows)]
    # make sure the number of dataframes in lst is equal to the number of jobs
    if len(lst) > job_number:
        # force the split result into the smaller number of jobs
        indices = get_indices_for_combining_list_of_lists(lst=lst, num=job_number)
        # use the indices to combine our list of lists into the right number
        lst = [pd.concat([lst[r] for r in x]) for x in indices]
        # check we have the right number of jobs
        assert (len(lst) == job_number)
        # check we have the right number of total elements
        assert (sum([len(x) for x in lst]) == df_len)
    return lst


def get_indices_for_combining_list_of_lists(lst, num, addon=0):
    n = len(lst) / num
    combine_n = ceil(n)
    num_groups = int(min(len(lst) / combine_n, num))
    loop_max = combine_n * num_groups
    remainder = len(lst) - combine_n * num_groups
    remainder_groups = num - num_groups
    indices = [range(i + addon, min(combine_n + i, len(lst)) + addon) for i in range(0, loop_max, combine_n)]
    if remainder:
        extra_indices = get_indices_for_combining_list_of_lists(lst[len(lst) - remainder:], remainder_groups,
                                                                addon=loop_max + addon)
        indices.extend(extra_indices)
    return indices


def job_pool(data, job_number, job_to_do, groupby=None, split_col=None, **kwargs):
    cleanup = None
    if split_col:
        slices = split_on(data, split_col, job_number)
    else:
        slices = chunks(data, job_number)
    logging.debug('got {} chunks'.format(len(slices)))
    if groupby:
        logging.debug('ensuring slices adhere to grouping rules.... this is very slow as only 1 CPU')
        slices = amend_groups(slices, groupby)
    if len(slices) < job_number:
        job_number = len(slices)
    shares = []
    memories = []
    if 'shared' in kwargs is not None and isinstance(kwargs['shared'], list):
        for df, dtypes in kwargs['shared']:
            shm, arr, dtypes_sorted = create_shared_block(df, dtypes)
            # pass the name of the shared memory, the shape of the array, the columns and the dtypes to the child processes
            shares.append([shm.name, arr.shape, dtypes_sorted])
            memories.append(shm)
        kwargs['shared'] = shares

    pool = multiprocessing.Pool(processes=job_number)
    arg = [(job_to_do, slice, kwargs) for slice in slices]
    jobs = pool.map(worker_wrapper, arg)

    for share in memories:
        share.close()
        share.unlink()
    return jobs


def worker_wrapper(arg):
    job_to_do, slice, kwargs = arg
    return job_to_do(slice, **kwargs)


def create_shared_block(to_share, dtypes):
    # float64 can't be pickled
    for col, dtype in to_share.dtypes.items():
        if dtype == 'float64':
            to_share[col] = pd.to_numeric(to_share[col], downcast='float')
    # make the dataframe a numpy array
    to_share.reset_index(inplace=True)
    dtypes_sorted = sort_dtypes(to_share, dtypes)
    values = [tuple(x) for x in to_share.values.tolist()]
    to_share = np.array(values, dtype=(dtypes_sorted))
    shm = shared_memory.SharedMemory(create=True, size=to_share.nbytes)
    # Now create a NumPy array backed by shared memory
    np_array = np.ndarray(to_share.shape, dtype=dtypes_sorted, buffer=shm.buf)
    np_array[:] = to_share[:]  # Copy the original data into shared memory
    return shm, np_array, dtypes_sorted


def sort_dtypes(df, dtypes):
    # category is a pandas dtype, not numpy
    string_types = ('category', 'object', '|S')
    dtypes = [(x, '|S{}'.format(df[x].str.len().max())) if y in string_types else (x, y) for x, y in dtypes if
              x in df.columns]
    # build a lookup
    dtypes_dict = {x: y for x, y in dtypes}
    # fix the order
    dtypes_sorted = [(x, dtypes_dict[x]) for x in df.columns]
    return dtypes_sorted
