import multiprocessing
from itertools import repeat


# split a list into evenly sized chunks
def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def dispatch_jobs(data, job_number, job_to_do, **kwargs):
    total = len(data)
    chunk_size = total / job_number
    slices = chunks(data, int(chunk_size))
    jobs = []

    for slice in slices:
        j = multiprocessing.Process(target=job_to_do, args=(slice,), kwargs=kwargs)
        jobs.append(j)
    for j in jobs:
        j.start()
    return jobs


def job_pool(data, job_number, job_to_do, **kwargs):
    total = len(data)
    chunk_size = total / job_number
    slices = chunks(data, int(chunk_size))
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
