#!python
#cython: language_level=3, boundscheck=False
from multiprocessing import shared_memory
cimport numpy as np
import time

np.import_array()

cdef class Vectors:

    cdef public object shm_name
    cdef public float[:, :] data
    cdef public object shm

    def __init__(self, shm_name, shape, dtype):
        print('Vectors Class for getting shared memory - {}'.format(shm_name))
        self.__get_shared_array(shm_name, shape, dtype)
        print('data is {}, {}'.format(self.data, self.data.shape))
        #print(self.data.base[0])

    def __setitem__(self, i, vector):
        self.data.base[i] = vector

    def __getitem__(self, i):
        return self.data.base[i]

    def __get_shared_array(self, shm_name: str, shape: tuple, dtype: str):
        print('get_shared_array {}'.format(shm_name))
        self.shm = shared_memory.SharedMemory(name=shm_name)
        print('shm buf is {} size {}'.format(self.shm.buf, self.shm.buf.nbytes))
        time.sleep(30)
        self.data = np.ndarray(shape=shape, dtype=dtype, buffer=self.shm.buf)

