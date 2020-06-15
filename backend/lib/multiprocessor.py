import multiprocessing
from itertools import repeat


# split a list into evenly sized chunks
def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def dispatch_jobs(data, job_number, job_to_do, groupby=None, **kwargs):
    total = len(data)
    chunk_size = total / job_number
    slices = chunks(data, int(chunk_size))
    if groupby:
        slices = amend_groups(slices, groupby)
    jobs = []

    for slice in slices:
        j = multiprocessing.Process(target=job_to_do, args=(slice,), kwargs=kwargs)
        jobs.append(j)
    for j in jobs:
        j.start()
    return jobs


def amend_groups(slices, groupby):
    slice_count = len(slices)
    if slice_count > 1:
        for i, slice in enumerate(slices):
            if i < slice_count - 1:
                slice_y = slices[i + 1]
                slice_x_index = len(slice) - 1
                slice_x_value = slice[slice_x_index][groupby]
                slice_y_len = len(slices[i + 1])
                slice_y_index = 0
                modify = False
                while slice_x_value == slice_y[slice_y_index][groupby]:
                    slice_y_index = slice_y_index + 1
                    print(slice_x_index)
                    print(slice_y_index)
                    modify = True
                    if slice_x_index < 0 or slice_y_index == slice_y_len:
                        break
                if modify:
                    slices[i].extend(slice_y[0:slice_y_index])
                    slices[i + 1] = slice_y[slice_y_index + 1:]
    return slices


def job_pool(data, job_number, job_to_do, groupby=None, **kwargs):
    total = len(data)
    chunk_size = total / job_number
    slices = chunks(data, int(chunk_size))
    if groupby:
        slices = amend_groups(slices, groupby)
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
