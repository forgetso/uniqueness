from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("c_get_shm", ["backend/tmp/c_get_shm.pyx"],
              libraries=['rt'])
]
setup(
    ext_modules=cythonize(extensions, gdb_debug=True)
)
