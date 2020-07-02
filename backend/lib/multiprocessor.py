import multiprocessing
import logging
import numpy as np


# split a list into evenly sized chunks
def chunks(lst, n):
    if n < 1:
        n = 1
    return [x.tolist() for x in np.array_split(lst, n) if len(x)]


def amend_groups(slices, groupby):
    # Takes a PRE-SORTED list of lists of dicts and makes sure that each inner list is grouped according to groupby
    # If we have [[{'w': 'ar'}, {'w': 'at'}, {'w': 'az'}], [{'w': 'as'}, {'w': 'bc'}]]
    # and groupby equal to ['w', 0] we need to remove the 'as' record from the second internal list and
    # append it to the first giving [[{'w': 'ar'}, {'w': 'at'}, {'w': 'az'}, {'w': 'as'}], [{'w': 'bc'}]]
    slice_count = len(slices)
    if slice_count > 1:
        for cur_slice_idx, _ in enumerate(slices):
            slice_x = slices[cur_slice_idx]
            if cur_slice_idx < slice_count - 1:
                nxt_slice_idx = cur_slice_idx + 1
                slice_y = slices[nxt_slice_idx]
                slice_x_index = len(slice_x) - 1
                if slice_x_index > -1:
                    slice_x_value = get_groupby_val(slice_x[slice_x_index], groupby)
                    slice_y_len = len(slice_y)
                    slice_y_index = 0
                    modify = False
                    while slice_x_value == get_groupby_val(slice_y[slice_y_index], groupby):
                        slice_y_index = slice_y_index + 1
                        modify = True
                        if slice_x_index < 0 or slice_y_index == slice_y_len:
                            break
                    if modify:
                        # taking a slice 0:0 does not work so we need this if statement
                        if slice_y_index == 0:
                            slices[cur_slice_idx].append(slice_y[0])
                        else:
                            slices[cur_slice_idx].extend(slice_y[0:slice_y_index])
                        slices[nxt_slice_idx] = slice_y[slice_y_index:]
                        if len(slices[nxt_slice_idx]) == 0:
                            # move this slice to the start of the list to maintain the list length
                            slices = [slices[nxt_slice_idx]] + slices[:nxt_slice_idx] + slices[nxt_slice_idx + 1:]
    slices = [x for x in slices if x]
    return slices


def get_groupby_val(slice_val, groupby):
    group_val = slice_val[groupby[0]]

    if len(groupby) > 1:
        for group in groupby[1:]:
            if group_val == 'i':
                print('found')
                break
            group_val = group_val[group]
    return group_val


def job_pool(data, job_number, job_to_do, groupby=None, **kwargs):
    total = len(data)
    chunk_size = total / job_number
    slices = chunks(data, int(chunk_size))
    if groupby:
        logging.debug('ensuring slices adhere to grouping rules....')
        slices = amend_groups(slices, groupby)
    if len(slices) < job_number:
        job_number = len(slices)
    pool = multiprocessing.Pool(processes=job_number)
    arg = [(job_to_do, slice, kwargs) for slice in slices]
    jobs = pool.map(worker_wrapper, arg)
    return jobs


def worker_wrapper(arg):
    job_to_do, slice, kwargs = arg
    return job_to_do(slice, **kwargs)


def jobs_running(jobs):
    running = False
    for job in jobs:
        # job.join(timeout=0)
        if job.is_alive():
            running = True
            break
    return running
