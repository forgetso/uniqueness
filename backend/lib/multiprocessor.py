import multiprocessing


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


def jobs_running(jobs):
    running = False
    for job in jobs:
        # job.join(timeout=0)
        if job.is_alive():
            running = True
            break
    return running
