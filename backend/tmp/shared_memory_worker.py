import tracemalloc
import spacy
import sys
from multiprocessing import current_process
import argparse
import time
words = ['dog', 'cat', 'fox', 'chicken', 'cow', 'mouse']


def work_with_shared_memory(work, shm_name, rows, cols, dtype):
    tracemalloc.start()
    shape = (rows, cols)
    current, peak = tracemalloc.get_traced_memory()

    shared = {'vectors': {'name': shm_name, 'shape': shape, 'dtype': dtype}}
    nlp = spacy.load('/home/chris/dev/uniqueness/data/crawl', shared=shared)
    print(f"Current Process: {current_process().name=} memory usage {current / 1e6}MB; Peak: {peak / 1e6}MB SharedName: {nlp.vocab.vectors.vectors_shared_name}")
    time.sleep(3)
    sims = []
    for pair in work:
        # data is sent as numpy.str...
        w1 = nlp(str(pair[0]))
        w2 = nlp(str(pair[1]))
        sim = w1.similarity(w2)
        sims.append(sim)
    sys.stdout.flush()
    tracemalloc.stop()
    nlp.vocab.vectors.close_shared_memory()
    return sims


def get_nlp_similarity(w1, w2, model):
    sim = None
    if w1 and w2:
        w1v = model(w1)
        w2v = model(w2)
        sim = w1v.similarity(w2v)
    return sim


def setup():
    parser = argparse.ArgumentParser(description='Shared mem getter')
    parser.add_argument('shm_name', type=str, help='Shared memory name')
    parser.add_argument('rows', type=int, help='Shared memory rows')
    parser.add_argument('cols', type=int, help='Shared memory cols')
    parser.add_argument('dtype', type=str, help='Shared memory dtype')
    args = parser.parse_args()
    work_with_shared_memory(shm_name=args.shm_name, rows=args.rows, cols=args.cols, dtype=args.dtype)


if __name__ == "__main__":
    setup()
