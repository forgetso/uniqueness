import numpy as np
from backend.settings import SEMANTIC_SPACE_FOLDER, SHARE_PREFIX
from backend.db.layer import get_affixes, get_all_words
from backend.lib.measures import get_all_measures
import pandas as pd
from multiprocessing import shared_memory
import re
import gc
import logging
import time

FLOAT64 = np.dtype(np.float64).str
FLOAT32 = np.dtype(np.float32).str


class SharedManager:
    def __init__(
            self,
            word_filter: bool = None,
            rebuild: bool = None,
            from_shared: bool = None,
            from_disk: bool = None,
            shares: dict = None,
    ):
        try:
            self.rebuild = rebuild
            self.word_filter = word_filter
            self.data = {}
            self.shares = {}
            if shares:
                self.shares = shares
            if from_shared:
                self.load_from_shared()
            elif from_disk:
                self.load_from_disk()
                self.share_data()
                # we don't need the original copy of the data in the parent process
                del self.data
                gc.collect()
        except Exception as e:
            logging.error("Error in SharedManager, attempting cleanup - {}".format(e))
            self.close_shared_memories()
            self.unlink_shared_memories()

    def load_from_disk(self):
        self.data["affixes"] = np.array(list({x.get("affix") for x in get_affixes()}))
        self.data["all_words"] = pd.DataFrame(get_all_words(), columns=["word"])
        self.data["all_measures"] = get_all_measures(rebuild=self.rebuild)

    def share_data(self):
        for name in self.data:
            if self.data[name] is not None:
                shm, np_array, type = self.create_shared_block(name, self.data[name])
                self.shares[name] = {
                    "name": shm.name,
                    "shape": np_array.shape,
                    "dtype": np_array.dtype,
                    "type": type,
                }

    def load_from_shared(self):
        for name in self.shares:
            self.get_shared_memory(name, self.shares[name])
        return

    def create_shared_block(self, shm_name, to_share):
        to_share = self.downcast_float64(to_share)
        type = "numpy"
        # make the dataframe a numpy array
        if isinstance(to_share, pd.DataFrame):
            to_share = self.pandas_to_numpy(to_share)
            type = "pandas"
        # size = max(to_share.nbytes, sys.getsizeof(to_share))
        shm = shared_memory.SharedMemory(
            name="{}_{}".format(SHARE_PREFIX, shm_name),
            create=True,
            size=to_share.nbytes,
        )
        # Now create a NumPy array backed by shared memory
        np_array = np.ndarray(to_share.shape, dtype=to_share.dtype, buffer=shm.buf)
        np_array[:] = to_share[:]  # Copy the original data into shared memory
        return shm, np_array, type

    def pandas_to_numpy(self, df):
        # make the dataframe a numpy array
        if df.index.name is not None:
            df.reset_index(inplace=True)
        dtypes = self.make_dtypes_numpy_compatible(df)
        values = [tuple(x) for x in df.values.tolist()]
        return np.array(values, dtype=dtypes)

    def get_shared_memory(self, name: str, share: dict):
        sh_m = shared_memory.SharedMemory(name=share["name"])

        np_array = np.ndarray(
            shape=share["shape"], dtype=share["dtype"], buffer=sh_m.buf
        )
        # this means we have a list of columns and dtypes, otherwise it is a single dtype for a numpy array

        # create a pandas dataframe
        if share["type"] == "pandas":
            columns = [col for col in np_array.dtype.fields]
            self.data[name] = pd.DataFrame(np_array, columns=columns)
            for coldtype in np_array.dtype.fields.items():
                col = coldtype[0]
                dtype = coldtype[1][0]
                self.data[name][coldtype[0]] = self.data[name][col].astype(dtype)
        # if column names are not passed then we just pass back the numpy array
        else:
            self.data[name] = np_array

    def close_shared_memories(self, shared=None):
        if not shared:
            shared = self.shares
        for share in shared.values():
            try:
                sh_m = shared_memory.SharedMemory(name=share['name'])
                sh_m.close()
            except FileNotFoundError:
                pass

    def unlink_shared_memories(self, shared=None):
        if not shared:
            shared = self.shares
        for share in shared.values():
            try:
                sh_m = shared_memory.SharedMemory(name=share['name'])
                sh_m.close()
            except FileNotFoundError:
                pass

    @staticmethod
    def downcast_float64(arr):
        # float64 can't be pickled
        if hasattr(arr, "dtypes"):
            for col, dtype in arr.dtypes.items():
                if dtype == FLOAT64:
                    arr[col] = pd.to_numeric(arr[col], downcast="float")
        elif hasattr(arr, "dtype"):
            if arr.dtype == FLOAT64:
                arr = arr.astype(FLOAT32)
        return arr

    @staticmethod
    def make_dtypes_numpy_compatible(df):
        # category is a pandas dtype, not numpy
        # object should be string
        # str, |S should be fixed to the max length of string in the column
        string_types = re.compile("^(category|object|\|S|str|\|O)")
        dtypes_columns = [
            (col, df.dtypes.to_list()[i].str) for i, col in enumerate(df.columns)
        ]
        dtypes_columns = [
            (x, "|S{}".format(df[x].str.len().max()))
            if re.match(string_types, y)
            else (x, y)
            for x, y in dtypes_columns
        ]
        return dtypes_columns
