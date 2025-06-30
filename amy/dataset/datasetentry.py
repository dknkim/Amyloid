from pathlib import Path
from typing import Union, Dict, List, Optional
import h5py
from copy import deepcopy
from datetime import datetime

import numpy as np
import nibabel as nib

from .datasetinstance import DatasetInstance



class DatasetEntry(object):
    """
    A class used to store information about the different training objects
    This class works like a regular dict
    """

    def __init__(
            self,
            instances: List[DatasetInstance] = None,
            date: int = None,
            amyloid: float = None,
            tracer: str = None,
            patient_id: str = None,
            centiloid: bool = None,
            dataset: str = None,
            age: float=None,
            sex: int = None,
            apoe: int = None,
            edu: int = None,
            mmse: int =None,
            status: str = None,
            # dx: str=None,
            # note: str=None,
            ):
        """
        Args:
            image_path (str, Path, list): The path where the data is stored
            datasetname (str): The name of the dataset the data is from
            dataset_type (str): What kind of data the data is
            shape (tuple): The shape of the data
        """

        self.instances = instances if instances is not None else list()

        self.date = date
        self.amyloid = amyloid
        self.tracer = tracer
        self.patient_id = patient_id
        self.centiloid = centiloid
        self.dataset = dataset
        self.age = age
        self.sex = sex
        self.apoe = apoe
        self.edu = edu
        self.mmse = mmse
        self.status = status
        # self.dx=dx
        # self.note=note

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.instances[key]
        return self.to_dict()[key]

    def __delitem__(self, index):
        del self.instances[index]

    def __str__(self):
        return str(self.to_dict())

    def __len__(self):
        return len(self.instances)

    def __repr__(self):
        return self.__str__()
    
    def centiloid_threshold(self):
        if self.amyloid <=0:
            amyloid=0
        elif self.amyloid >=100:
            amyloid=100
        else:
            amyloid = self.amyloid
        return amyloid

            

    
    def convert_to_centiloid(self):
        if not self.centiloid:
            if self.dataset == "ADNI" and self.tracer is not None and self.amyloid is not None:
                if self.tracer == "AV45":
                    amyloid = 196.9*self.amyloid - 196.03
                    centiloid = True
                elif self.tracer == "FBB":
                    amyloid = 156.08*self.amyloid - 151.65
                    centiloid = True
                elif self.tracer == "PIB":
                    amyloid = 100*(self.amyloid - 1.012)/(2.081 - 1.012)
                    centiloid = True
            if self.dataset == "HABS" and self.tracer is not None and self.amyloid is not None:
                if self.tracer == "PIB":
                    # Use the one defined in the centiloid paper
                    amyloid = 100*(self.amyloid - 1.009)/(1.067)
                    centiloid = True
            if self.dataset == "Stanford" and self.tracer is not None and self.amyloid is not None:
                if self.tracer == "FBB":
                    amyloid = 156.08*self.amyloid - 151.65
                    centiloid = True
        else:
            amyloid = self.amyloid
            centiloid = self.centiloid

        return amyloid, centiloid


    def order_instances(self):
        self.instances.sort(key=lambda x: x.sequence_type)

    def add_instance(self, entry: DatasetInstance):
        """
        Append DataseEntry
        Args:
            info (DatasetEntry): The DatasetEntry to be appended
        """
        self.instances.append(deepcopy(entry))

    def keys(self):
        """
        dict keys of class
        """
        return self.to_dict().keys()

    def to_dict(self) -> dict:
        """
        returns:
            dict format of this class
        """
        entry_dict = {
            "date": self.date,
            "amyloid": self.amyloid,
            "tracer": self.tracer,
            "patient_id": self.patient_id,
            "centiloid": self.centiloid,
            "dataset": self.dataset,
            "age": self.age,
            "sex": self.sex,
            "apoe": self.apoe,
            "edu": self.edu,
            "mmse": self.mmse,
            "status": self.status,
            # "dx": self.dx,
            # "note": self.note,
            }

        entry_dict['instances'] = [instance.to_dict() for instance in self.instances]
        return entry_dict

    def from_dict(self, in_dict: dict):
        """
        Args:
            in_dict: dict, dict format of this class
        Fills in the variables from the dict
        """
        if isinstance(in_dict, dict):
            self.date = in_dict['date']
            self.amyloid = in_dict['amyloid']
            self.tracer = in_dict['tracer']
            self.patient_id = in_dict['patient_id']
            self.centiloid = in_dict['centiloid']
            self.dataset = in_dict['dataset']
            self.age=in_dict['age']
            self.sex=in_dict['sex']
            self.apoe=in_dict['apoe']
            self.edu=in_dict['edu']
            self.mmse=in_dict['mmse']
            self.status=in_dict['status']
            # self.dx=in_dict['dx']
            # self.note=in_dict['note']

            for instance in in_dict['instances']:
                self.instances.append(DatasetInstance().from_dict(instance))

        return self
    
    def to_bids(self, bids_dir: Union[Path, str], date, patient_num):
        """
        Save the data in bids format
        """
        for instance in self.instances:
            instance.to_bids(bids_dir, date, patient_num)
        