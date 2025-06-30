from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional

import torch
import torch.nn as nn
import numpy as np
import monai

from .datasetcontainer import DatasetContainer

class Loader(object):
    CUTOFF = {
        "A4": {"AV45": 18, "FBP": 18},  # centiloid
        "Stanford": {"FBB": 1.1},
        "ADNI": {"AV45": 18, "PIB": 9, "FBB": 12},
        "OASIS": {"PIB": 27.2, "PIB_RSF": 16.4, "AV45": 21.9, "AV45_RSF": 20.6},  # centiloid
        "HABS": {"PIB":24.6}, # centiloid
        # "ADNI": {"AV45": 1.11, "PIB": 1.44, "FBB": 1.11},  # https://adni.loni.usc.edu/adni-publications/Elman-2020-Amyloid-%CE%B2%20Positivity%20Predicts%20Cogni.pdf
        }

    def __init__(
            self, 
            container: DatasetContainer, 
            scans: Tuple[str],
            transforms: Optional[monai.transforms.Transform] = None,
            tracer: Optional[str] = None,
            clinical_data: Optional[List[str]] = None,
            classification: bool = False,
            amyloid_mean: Optional[float] = None,
            amyloid_std: Optional[float] = None,
            histogram_normalize: bool = False,
            brain_mask_centering: bool = False,
            return_patient_date: bool = False,
            top_percentile: Optional[float] = None,
            encode_dataset: bool = False,
            clip: bool = True,
            override_cutoff: Optional[float] = None,
            use_status: bool = False,
            regression_condition:bool=None,
            ) -> None:
        
        self.container = container
        self.classification = classification
        self.scans = scans
        self.tracer = tracer if isinstance(tracer, list) else [tracer]
        self.clinical_data = clinical_data
        self.encode_dataset = encode_dataset
        self.clip = clip
        self.override_cutoff = override_cutoff
        self.use_status = use_status

        self.transforms = transforms
        self.top_percentile = top_percentile
        
        self.patient_date_dict = self._make_dataset()
        self.indicies = list()
        self.return_patient_date = return_patient_date
        self.brain_mask_centering = brain_mask_centering

        self.histogram_normalize = monai.transforms.HistogramNormalize() if histogram_normalize else False

        self.amyloid_mean = amyloid_mean
        self.amyloid_std = amyloid_std
        self.regression_condition=regression_condition
        for patient, patient_dict in self.patient_date_dict.items():
            for date, date_list in patient_dict.items():
                for index in date_list:
                    self.indicies.append((index, patient, date))
    
    def _make_dataset(self) -> None:
        # only keep the top percentiles of the data. This is done to only look at the extreme cases (might be easier to predict)
        if self.top_percentile is not None:
            amy_dict = dict()
            for entry in self.container:
                dataset = entry.dataset
                amyloid = entry.amyloid
                tracer = entry.tracer

                entry_dict = entry.to_dict()
                if self.clinical_data is not None:
                    clinical_values = list()
                    for c in self.clinical_data:
                        if c in entry_dict.keys():
                            clinical_values.append(entry_dict[c] is None)
                    if any(clinical_values) or len(clinical_values) != len(self.clinical_data):
                        continue

                if amyloid is None or np.isnan(amyloid) or isinstance(amyloid, str):
                    continue
                
                if dataset not in amy_dict.keys():
                    amy_dict[dataset] = dict()
                if tracer not in amy_dict[dataset].keys():
                    amy_dict[dataset][tracer] = list()
                amy_dict[dataset][tracer].append(amyloid)
            # get the top percentile for each dataset
            percentile_dict = dict()

            for dataset, tracer_dict in amy_dict.items():
                percentile_dict[dataset] = dict()
                for tracer, amy_list in tracer_dict.items():
                    amy_list.sort()
                    lowest = amy_list[int(len(amy_list) * (1 - self.top_percentile))] + 1e-4
                    highest = amy_list[int(len(amy_list) * self.top_percentile)] - 1e-4
                    percentile_dict[dataset][tracer] = (lowest, highest)

        data_indicies = dict()
        for index, entry in enumerate(self.container):
            date = entry[0].date
            tracer = entry.tracer
            amyloid = entry.amyloid
            patient = entry.patient_id

            entry_dict = entry.to_dict()
            if self.clinical_data is not None:
                clinical_values = list()
                for c in self.clinical_data:
                    if c in entry_dict.keys():
                        clinical_values.append(entry_dict[c] is None or np.isnan(entry_dict[c]))

                if any(clinical_values) or len(clinical_values) != len(self.clinical_data):
                    continue
            
            sequences = [instance.sequence_type.lower() for instance in entry]
            
            if amyloid is None or np.isnan(amyloid) or isinstance(amyloid, str):
                continue
            if tracer is not None or self.tracer is not None:
                if tracer not in self.tracer and not any([t in tracer for t in self.tracer]):
                    continue

            if not all([scan.lower() in sequences for scan in self.scans]):
                continue

            if self.top_percentile is not None:
                dataset = entry.dataset
                cutoff = percentile_dict[dataset][tracer]
                # dont keep amyloid values that are in the middle (the difficult cases)
                if cutoff[0] < amyloid < cutoff[1]:
                    continue

            if patient not in data_indicies.keys():
                data_indicies[patient] = dict()
            if date not in data_indicies[patient].keys():
                data_indicies[patient][date] = list()

            data_indicies[patient][date].append(index)
        
        return data_indicies

    def __len__(self) -> int:
        # use self.indicies as the length accounting for the dates in each patient
        return len(self.indicies)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        new_index, patient, date = self.indicies[index]
        entry = self.container[new_index]
        
        imgs = []
        for scan in self.scans:
            seqs = [instance for instance in entry if instance.sequence_type == scan and "brain_mask" not in instance.image_path]

            if self.brain_mask_centering:
                mask = [instance for instance in entry if "brain_mask" in instance.image_path][0]
                mask = mask.open().get_fdata()[None, :, :, :]

            # choose randomly
            seq = np.random.choice(seqs)
            img = seq.open()
            affine = img.affine
            img = img.get_fdata()
            # quick fix in case of nan values
            img[np.isnan(img)] = 0
            if self.clip:
                img = np.clip(img, a_min=np.percentile(img, 0.5), a_max=np.percentile(img, 99.5))  # same clip as nnunet
            
            if self.histogram_normalize:
                norm_mask = img > 0
                img = self.histogram_normalize(img, norm_mask).numpy()
            imgs.append(img)

        imgs = np.stack(imgs, axis=0)
        meta_img = monai.data.meta_tensor.MetaTensor(imgs, affine)
        if self.brain_mask_centering:
            meta_mask = monai.data.meta_tensor.MetaTensor(mask, affine)
            meta_img = {"img": meta_img, "mask": meta_mask}

        imgs = self.transforms(meta_img)
        if self.brain_mask_centering:
            imgs = imgs["img"]

        x = torch.Tensor([entry.amyloid])
        if self.classification:
            amyloid = entry.amyloid
            tracer = entry.tracer
            dataset = entry.dataset
            status = entry.status
            
            if self.override_cutoff is not None:
                cutoff = self.override_cutoff
                status = None
            else:
                cutoff = self.CUTOFF[dataset][tracer]
            
            if status is not None and self.use_status:
                if status == "positive":
                    x = torch.Tensor([1]).float()
                elif status == "negative":
                    x = torch.Tensor([0]).float()
                else:
                    x = torch.Tensor([amyloid >= cutoff]).float()
            else:
                x = torch.Tensor([amyloid >= cutoff]).float()
            # x = torch.Tensor([amyloid >= cutoff]).float()
        
        # this is completely bull
        if self.encode_dataset:
            dataset = entry.dataset
            z = torch.zeros((1, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
            if dataset == "ADNI":
                z[:] = 1
            elif dataset == "OASIS":
                z[:] = 2
            elif dataset == "A4":
                z[:] = 3
            elif dataset == "HABS":
                z[:] = 4
            
            imgs = torch.cat([imgs, z], dim=0)


        if not self.classification and self.amyloid_mean is not None and self.amyloid_std is not None and self.regression_condition is None:
            x = x
            #x = x/100 
            #x = (x - self.amyloid_mean) / self.amyloid_std
        if not self.classification and self.amyloid_mean is not None and self.amyloid_std is not None and self.regression_condition is True:
            x = x
            #x = x/100 
            #x = (x - self.amyloid_mean) / self.amyloid_std
            return imgs, x, patient, date
        
        if self.return_patient_date:
            return imgs, x, patient, date, dataset
        
    
        return imgs, x