from multiprocessing import shared_memory
import numpy as np
import argparse
from get_shm import main as get_shm
from multiprocessing import Process
from numpy import dtype

def main(cython=None, use_spacy=None):
    arr_share = np.load('/home/chris/dev/uniqueness/data/crawl/vocab/vectors')
    shape, dtype = arr_share.shape, arr_share.dtype
    # Create a shared memory of size arr_share.nbytes
    shm = shared_memory.SharedMemory(create=True, size=arr_share.nbytes)
    # Create array using the buffer of shm
    shm_np_array = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    # Copy the data into the shared memory
    np.copyto(shm_np_array, arr_share)
    print('shared name {}'.format(shm.name))
    print('shared size {}'.format(arr_share.nbytes))
    print(arr_share)
    run_processes(shm, shape, dtype, cython, 16, use_spacy)
    shm.close()
    shm.unlink()


def run_processes(shm, shape, dtype, cython, count, use_spacy):
    processes = []
    print(shm, shape, dtype, cython, count)
    for _unused in range(1, count):
        p = Process(target=get_shm, args=(shm.name, shape[0], shape[1], str(dtype)), kwargs={'cython': cython, 'use_spacy': use_spacy})
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def setup():
    parser = argparse.ArgumentParser(description='Shared mem getter')
    parser.add_argument('--cython', action='store_true', help='Use cython to get shared mem')
    parser.add_argument('--spacy', action='store_true', help='Use spacy to get shared mem')
    args = parser.parse_args()
    main(cython=args.cython, use_spacy=args.spacy)


if __name__ == "__main__":
    setup()
