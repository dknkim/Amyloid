import json
import random
import contextlib
import os
import glob

from pathlib import Path
from typing import Union, Dict, List, Optional
import h5py
from copy import deepcopy

import numpy as np
import nibabel as nib

from .datasetinstance import DatasetInstance
from .datasetentry import DatasetEntry

@contextlib.contextmanager
def temp_seed(seed):
    """
    Source:
    https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)

class BaseContainer(object):

    def __init__(self, entries: List[DatasetInstance]) -> None:
        self.entries = entries

    def __getitem__(self, index):
        return self.entries[index]

    def __len__(self):
        return len(self.entries)

    def __delitem__(self, index):
        del self.entries[index]

    def __str__(self):
        return str(self.to_dict())
    
    def __add__(self, other):
        new_container = deepcopy(self)
        new_container.entries += deepcopy(other.entries)
        return new_container

    def order(self):
        self.order_entries()
        self.order_instances()

    def order_entries(self):
        # date format should be year-month-day which makes this sorting work
        self.entries.sort(key=lambda x: x.date)

    def order_instances(self):
        for entry in self.entries:
            entry.order_instances()

    def shuffle(self, seed=None):
        """
        Shuffles the entries, used for random training
        Args:
            seed (int): The seed used for the random shuffle
        """
        with temp_seed(seed):
            random.shuffle(self.entries)

    def add_entry(self, entry: DatasetEntry):
        """
        Append DataseEntry
        Args:
            info (DatasetEntry): The DatasetEntry to be appended
        """
        self.entries.append(deepcopy(entry))

    def keys(self):
        return self.to_dict().keys()

    def to_dict(self):
        """
        returns:
            dict version of DatasetContainer
        """
        container_dict = dict()
        container_dict['entries'] = [entry.to_dict() for entry in self.entries]
        return container_dict
    
    def from_dict(self, in_dict):
        """
        Appends data into DatasetContainer from dict
        Args:
            in_dict (dict): Dict to append data from, meant to be used to recover when loading from file
        """
        for entry in in_dict['entries']:
            self.entries.append(DatasetEntry().from_dict(entry))

    def to_json(self, path: Union[str, Path]):
        """
        Save DatasetContainer as json file
        Args:
            path (str, Path): path where DatasetContainer is saved
        """
        path = Path(path)
        suffix = path.suffix
        if suffix != '.json':
            raise NameError('The path must have suffix .json not, ', suffix)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as outfile:
            json.dump(self.to_dict(), outfile, ensure_ascii=False, indent=4)

    @classmethod
    def from_json(cls, path: Union[str, Path]):
        """
        Load DatasetContainer from file
        Args:
            path (str, Path): Path to load from
        returns:
            The DatasetContainer from file
        """
        with open(path) as json_file:
            data = json.load(json_file)
        new_container = deepcopy(cls())
        new_container.from_dict(data)
        return new_container