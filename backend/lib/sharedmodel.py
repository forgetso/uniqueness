from backend.lib.sharedmanager import SharedManager
from backend.settings import DATA_FOLDER
import logging
import spacy
from typing import Union, List, Tuple
import numpy as np
from backend.lib.sense2vec.sense2vec import Sense2Vec
from backend.settings import SENSE2VEC_FOLDER
from pathlib import Path


class SharedModelManager(SharedManager):
    def __init__(
            self,
            from_shared: bool = None,
            from_disk: bool = None,
            shares: dict = None,
            models: Union[List, Tuple] = None,
    ):
        self.parent_process = from_disk and not from_shared
        self.models = []
        self.data = {}
        if models is not None:
            self.models = models
        super().__init__(from_shared=from_shared, from_disk=from_disk, shares=shares)

    def load_from_disk(self):
        for name in self.models:
            logging.debug("getting {} from shared memory".format(name))
            model_path = Path(DATA_FOLDER).joinpath(name).joinpath("vocab/vectors")
            self.data[name] = np.load(model_path)

    def get_shared_memory(self, name: str, share: dict):
        # take the share and put it in the right format for spacy
        shared = {"vectors": share}
        path = Path(DATA_FOLDER).joinpath(name)
        model = spacy.load(path.absolute(), shared=shared)

        # TODO allow models to have a type (spacy or sense2vec ... etc.)
        if name == 's2v':
            path = Path(SENSE2VEC_FOLDER)
            self.data[name] = Sense2Vec(model=model).from_disk(path.absolute())
        else:
            self.data[name] = model
