from multiprocessing import Process, cpu_count, shared_memory, Process, set_start_method, Pipe, Pool
from backend.lib.multiprocessor import job_pool
import numpy as np
from backend.tmp.shared_memory_worker import work_with_shared_memory
import spacy
import random
import pandas as pd
import time

set_start_method('fork', force=True)
words = ['dog', 'cat', 'fox', 'chicken', 'cow', 'mouse']


def run_processes(shm, shape, dtype, count):
    processes = []
    for _unused in range(1, count):
        p = Process(target=work_with_shared_memory, args=(shm.name, shape[0], shape[1], str(dtype)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    return processes


def get_shared_array(shm_name: str, shape: tuple, dtype: str):
    print('get_shared_array {}'.format(shm_name))
    shm = shared_memory.SharedMemory(name=shm_name)
    print('shm buf is {} size {}'.format(shm.buf, shm.buf.nbytes))
    data = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    print(data)
    return data


def main():
    # Load the spacy model - a shared memory for vectors will be created
    nlp = spacy.load('/home/chris/dev/uniqueness/data/crawl')
    shape, dtype = nlp.vocab.vectors.data.shape, nlp.vocab.vectors.data.dtype
    shm = nlp.vocab.vectors.shm
    process_count = int(cpu_count() / 2)
    work = []
    for _ in range(0, process_count * 10000):
        work.append((random.choice(words), random.choice(words)))
    start = time.time()
    processes = job_pool(data=work,
                         job_number=process_count,
                         job_to_do=work_with_shared_memory,
                         shm_name=shm.name,
                         rows=shape[0],
                         cols=shape[1],
                         dtype=dtype)
    end = time.time()
    result = pd.concat([pd.DataFrame(x) for x in processes])
    print('got {} similarities from {} jobs'.format(len(result), process_count))
    print('time taken {}s'.format(start - end))
    nlp.vocab.vectors.close_shared_memory()
    nlp.vocab.vectors.unlink_shared_memory()


if __name__ == "__main__":
    main()
