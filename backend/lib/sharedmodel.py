from backend.lib.sharedmanager import SharedManager
from backend.settings import DATA_FOLDER
import logging
import spacy
from typing import Union, List, Tuple
import numpy as np


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
            model_path = "".join([DATA_FOLDER, "/{}/vocab/vectors".format(name)])
            self.data[name] = np.load(model_path)

    def get_shared_memory(self, name: str, share: dict):
        # take the share and put it in the right format for spacy
        shared = {"vectors": share}
        self.data[name] = spacy.load("".join([DATA_FOLDER, "/", name]), shared=shared)
