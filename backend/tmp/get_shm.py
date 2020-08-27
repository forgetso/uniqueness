from multiprocessing.shared_memory import SharedMemory
import numpy as np
import argparse
import sys
from c_get_shm import Vectors
import time
import spacy


def main(name, rows, cols, dtype, cython=None, use_spacy=None):
    shape = (rows, cols)
    val = None
    if cython:
        print('getting vector using cython - {}'.format(name))
        vectors = Vectors(name, shape, dtype)
        print(hex(id(vectors.data)))
        vectors.data.base[0][0] = vectors.data.base[0][0] + 1
        val = vectors[0][0]
        time.sleep(1)
    elif use_spacy:
        nlp = spacy.load('/home/chris/dev/uniqueness/data/crawl',
                         shared={'vectors': {'name': name, 'shape': shape, 'dtype': dtype}})
        nlp.vocab.vectors.data.base[0][0] = nlp.vocab.vectors.data.base[0][0] + 1
        val = nlp.vocab.vectors.data.base[0][0]
        time.sleep(60)
    else:
        shm = SharedMemory(name=name)
        np_array = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        print('getting vector using python - {}'.format(name))
        val = np_array[0][0]
        shm.close()
    print('val is {}'.format(val))
    sys.stdout.flush()
    return


def setup():
    parser = argparse.ArgumentParser(description='Shared mem getter')
    parser.add_argument('name', type=str, help='Shared memory name')
    parser.add_argument('rows', type=str, help='Shared memory shape (rows)')
    parser.add_argument('cols', type=str, help='Shared memory shape (cols)')
    parser.add_argument('dtype', type=str, help='Shared memory dtype')
    parser.add_argument('--cython', action='store_true', help='Use cython to get shared mem')
    parser.add_argument('--spacy', action='store_true', help='Use spacy to get shared mem')
    args = parser.parse_args()
    main(name=args.name, rows=args.rows, cols=args.cols, dtype=args.dtype, cython=args.cython, use_spacy=args.spacy)


if __name__ == "__main__":
    setup()
