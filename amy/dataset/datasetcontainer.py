import os
import json
import h5py
import random
import contextlib
from copy import deepcopy

from glob import glob
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Union, Dict, List, Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from .utils import sort_sequence
from .base import BaseContainer
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

class DatasetContainer(BaseContainer):
    PET_MRI_THRESHOLD = 30

    def __init__(self, entries: List[DatasetEntry] = None) -> None:
        super().__init__(
            entries=entries if entries is not None else list(),
            )
    
    def split(self, splits: int = 5, split: int = 0, seed: int = 42):
        amyloids = np.array([entry.convert_to_centiloid()[0] for i, entry in enumerate(self.entries)])
        patients = np.array([entry.patient_id for i, entry in enumerate(self.entries)])

        pat_amy = dict()
        for i, patient in enumerate(patients):
            if patient not in pat_amy.keys():
                pat_amy[patient] = list()
            pat_amy[patient].append(amyloids[i])
        
        amyloids = np.array([np.mean(pat_amy[patient]) for patient in pat_amy.keys()])
        patients = np.array([patient for patient in pat_amy.keys()])
        # print(np.max(amyloids), np.min(amyloids))
        # amyloids = np.clip(amyloids, np.percentile(amyloids, 5), np.percentile(amyloids, 95))
        # place the amyloid values into 8 bins
        bins = np.array([0, 25, 50, 100])
        amyloids = np.digitize(amyloids, bins)
        print(np.unique(amyloids, return_counts=True))

        # sss = StratifiedShuffleSplit(n_splits=1, test_size=1-frac, random_state=seed)
        sss = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        out = sss.split(patients, amyloids)
        train, valid = list(out)[split]
        # train, valid = next(out)

        train_entries = list()
        valid_entries = list()

        for train_index in train:
            patient = patients[train_index]

            for entry in self.entries:
                if entry.patient_id == patient:
                    train_entries.append(deepcopy(entry))

        for valid_index in valid:
            patient = patients[valid_index]

            for entry in self.entries:
                if entry.patient_id == patient:
                    valid_entries.append(deepcopy(entry))
                    
        return DatasetContainer(entries=train_entries), DatasetContainer(entries=valid_entries)
    
    def split_old(self, frac: float = 0.8, seed: int = 42):
        patients = set([entry.patient_id for entry in self.entries])
        patients = list(patients)
        patients.sort()
        
        with temp_seed(seed):
            random.shuffle(patients)

        np.random.seed(seed)
        np.random.shuffle(patients)
        x, y = patients[:int(len(patients)*frac)], patients[int(len(patients)*frac):]

        train_entries = list()
        valid_entries = list()
        for entry in self.entries:
            if entry.patient_id in x:
                train_entries.append(deepcopy(entry))
            elif entry.patient_id in y:
                valid_entries.append(deepcopy(entry))
        return DatasetContainer(entries=train_entries), DatasetContainer(entries=valid_entries)

    def convert_to_centiloid(self):
        for entry in self.entries:
            amyloid, centiloid = entry.convert_to_centiloid()
            entry.amyloid = amyloid
            entry.centiloid = centiloid

    def centiloid_threshold(self):
        
        
        for entry in self.entries:
            
            amyloid = entry.centiloid_threshold()
            entry.amyloid = amyloid
            
            

    
    def from_A4(self, a4_dir: Union[str, Path], csv_file: Union[str, Path]):
        labels = self._load_labels_A4(csv_file)

        data = glob(str(Path(a4_dir) / Path("*/*/*/*/*.nii.gz"))) + glob(str(Path(a4_dir) / Path("*/*/*/*/*.nii")))
        data_dict = dict()

        for path in tqdm(data):
            path = Path(path)

            date_time = path.parent.parent.name
            scan = path.parent.parent.parent.name
            patient = path.parent.parent.parent.parent.name

            date = date_time.split("_")[0]
            time = date_time.split("_")[1]

            if patient not in data_dict.keys():
                data_dict[patient] = dict()
            if date not in data_dict[patient].keys():
                data_dict[patient][date] = list()
            data_dict[patient][date].append(path)            
        
        for patient in tqdm(data_dict.keys()):
            for date in data_dict[patient].keys():
                instances = list()
                for path in data_dict[patient][date]:
                    date_time = path.parent.parent.name
                    scan = path.parent.parent.parent.name
                    patient = path.parent.parent.parent.parent.name

                    date = date_time.split("_")[0]
                    time = "_".join(date_time.split("_")[1:])

                    if "T1" in scan and "brain_mask" not in str(path):
                        sequence = "MPRAGE"
                    elif "FLAIR" in scan:
                        sequence = "FLAIR"
                    elif "brain_mask" in str(path):
                        sequence = "MASK"
                    else:
                        sequence = scan
    
                    instance = DatasetInstance(
                        image_path=str(path),
                        sequence_type=sequence.upper(),
                        field_strength=None,
                        time=time,
                        date=date,
                        )
                    instances.append(instance)
                
                date = datetime.strptime(date, '%Y-%m-%d') if date is not None else None
                if patient not in labels.keys():
                    amyloid = None
                    tracer = None
                    centiloid = False
                    age=None
                    sex=None
                    apoe=None
                    edu=None
                    mmse=None
                    status=None
                else:
                    amyloid, tracer, centiloid, age, sex, apoe, edu, mmse, status = labels[patient]

                entry = DatasetEntry(
                    instances=instances,
                    date=None,
                    patient_id=patient,
                    amyloid=amyloid,
                    tracer=tracer,
                    centiloid=centiloid,
                    dataset="A4",
                    age=age,
                    sex=sex,
                    apoe=apoe,
                    edu=edu,
                    mmse=mmse,
                    status=status,
                    )
                
                self.entries.append(entry)
    
    def from_Stanford(self, stanford_dir: Union[str, Path], csv_file: Union[str, Path]):
        labels = self._load_labels_Stanford(csv_file)

        data = glob(str(Path(stanford_dir) / Path("*/*.nii.gz"))) + glob(str(Path(stanford_dir) / Path("*/*.nii")))
        data_dict = dict()

        for path in tqdm(data):
            path = Path(path)

            subject = path.parent.name

            if subject not in data_dict.keys():
                data_dict[subject] = list()
            data_dict[subject].append(path)

        for subject in tqdm(data_dict.keys()):
            instances = list()
            for path in data_dict[subject]:
                if "mask" in path.name:
                    sequence = "MASK"
                elif "ir-fspgr" in path.name or "irpspgr" in path.name:
                    sequence = "MPRAGE"
                elif "flair" in path.name:
                    sequence = "FLAIR"
                else:
                    sequence = "UNKNOWN"

                instance = DatasetInstance(
                    image_path=str(path),
                    sequence_type=sequence.upper(),
                    field_strength=None,
                    time=None,
                    date=None,
                    )
                instances.append(instance)
            
            if subject not in labels.keys():
                tracer, label = None, None
            else:
                tracer, label = labels[subject]

            entry = DatasetEntry(
                instances=instances,
                date=None,
                patient_id=subject,
                amyloid=label,
                tracer=tracer,
                centiloid=False,
                dataset="Stanford",
                )
            self.entries.append(entry)
    
    def from_Stanford(self, stanford_dir: Union[str, Path], csv_file: Union[str, Path]):
        labels = self._load_labels_Stanford(csv_file)

        data = glob(str(Path(stanford_dir) / Path("*/*.nii.gz"))) + glob(str(Path(stanford_dir) / Path("*/*.nii")))
        data_dict = dict()

        for path in tqdm(data):
            path = Path(path)

            subject = path.parent.name

            if subject not in data_dict.keys():
                data_dict[subject] = list()
            data_dict[subject].append(path)

        for subject in tqdm(data_dict.keys()):
            instances = list()
            for path in data_dict[subject]:
                if "mask" in path.name:
                    sequence = "MASK"
                elif "ir-fspgr" in path.name or "irpspgr" in path.name:
                    sequence = "MPRAGE"
                elif "flair" in path.name:
                    sequence = "FLAIR"
                else:
                    sequence = "UNKNOWN"

                instance = DatasetInstance(
                    image_path=str(path),
                    sequence_type=sequence.upper(),
                    field_strength=None,
                    time=None,
                    date=None,
                    )
                instances.append(instance)
            
            if subject not in labels.keys():
                tracer, label = None, None
            else:
                tracer, label = labels[subject]

            entry = DatasetEntry(
                instances=instances,
                date=None,
                patient_id=subject,
                amyloid=label,
                tracer=tracer,
                centiloid=False,
                dataset="Stanford",
                )
            self.entries.append(entry)

    def from_OASIS(self, oasis_dir: Union[str, Path], csv_file: Union[str, Path], rsf: bool = False):
        labels = self._load_labels_OASIS(csv_file)

        # first is subject, second is date, third is nothing, scan is in the name
        data = glob(str(Path(oasis_dir) / Path("*/*/anat/*.nii.gz"))) + glob(str(Path(oasis_dir) / Path("*/*/anat/*.nii")))
        data_dict = dict()

        age=None
        sex=None
        apoe=None
        edu=None
        mmse=None
        status=None
        
        
        for path in tqdm(data):
            path = Path(path)

            date = path.parent.parent.name
            subject = path.parent.parent.parent.name.split("-")[1]

            if subject not in data_dict.keys():
                data_dict[subject] = dict()
            if date not in data_dict[subject].keys():
                data_dict[subject][date] = list()
            data_dict[subject][date].append(path)
        
        for subject in tqdm(data_dict.keys()):
            for date in data_dict[subject].keys():
                instances = list()
                for path in data_dict[subject][date]:
                    if "T1w" in path.name:
                        sequence = "MPRAGE"
                    elif "FLAIR" in path.name:
                        sequence = "FLAIR"
                    elif "T2w" in path.name:
                        sequence = "T2"
                    else:
                        sequence = "UNKNOWN"

                    instance = DatasetInstance(
                        image_path=str(path),
                        sequence_type=sequence.upper(),
                        field_strength=None,
                        time=None,
                        date=date,  
                        )
                    instances.append(instance)

                if subject not in labels.keys():
                    tracer, label, label_rsf = None, None, None
                elif date not in labels[subject].keys():
                    tracer, label, label_rsf = None, None, None
                else:
                    label, tracer, label_rsf, age, sex, apoe, edu, mmse = labels[subject][date]
                    

                if rsf and tracer is not None:
                    tracer += "_RSF"

                entry = DatasetEntry(
                    instances=instances,
                    date=date,
                    patient_id=subject,
                    amyloid=label if not rsf else label_rsf,
                    tracer=tracer,
                    centiloid=True,
                    dataset="OASIS",
                    age=age,
                    sex=sex,
                    apoe=apoe,
                    edu=edu,
                    mmse=mmse,
                    status=status,
                    )
                self.entries.append(entry)
    
    def from_ADNI(self, adni_dir: Union[str, Path], csv_file: Union[str, Path], use_centiloid: bool = False):
        labels = self._load_labels_ADNI(csv_file)
        
        data = glob(str(Path(adni_dir) / Path("*/*/*/*/*.nii.gz"))) + glob(str(Path(adni_dir) / Path("*/*/*/*/*.nii")))
        data_dict = dict()

        for path in tqdm(data):
            path = Path(path)

            date_time = path.parent.parent.name
            scan = path.parent.parent.parent.name
            patient = path.parent.parent.parent.parent.name

            date = date_time.split("_")[0]
            time = date_time.split("_")[1]

            sequence = sort_sequence(scan)

            if patient not in data_dict.keys():
                data_dict[patient] = dict()
            if date not in data_dict[patient].keys():
                data_dict[patient][date] = list()
            data_dict[patient][date].append(path)            
        
        for patient in tqdm(data_dict.keys()):
            for date in data_dict[patient].keys():
                instances = list()
                for path in data_dict[patient][date]:
                    date_time = path.parent.parent.name
                    scan = path.parent.parent.parent.name
                    patient = path.parent.parent.parent.parent.name

                    date = date_time.split("_")[0]
                    time = "_".join(date_time.split("_")[1:])

                    sequence = sort_sequence(scan)
    
                    instance = DatasetInstance(
                        image_path=str(path),
                        sequence_type=sequence.upper(),
                        field_strength=None,
                        time=time,
                        date=date,
                        )
                    instances.append(instance)
                
                date = datetime.strptime(date, '%Y-%m-%d') if date is not None else None
                if patient not in labels.keys():
                    amyloid = None
                    tracer = None
                    age = None
                    sex = None
                    apoe = None
                    edu = None
                    mmse = None
                else:
                    pet_info = list()
                    for (label, tracer, amy_date, age, sex, apoe, edu, mmse, centiloid) in labels[patient]:
                        diff = abs((amy_date - date).days)
                        if diff <= self.PET_MRI_THRESHOLD:
                            pet_info.append((diff, label, tracer, amy_date, age, sex, apoe, edu, mmse, centiloid))
                    if len(pet_info) < 1:
                        amy_date = None
                        tracer, amyloid = None, None
                        age = None
                        sex = None
                        apoe = None
                        edu = None
                        mmse = None
                        centiloid = None
                    else:
                        # sort based on the difference between the PET and MRI scan
                        pet_info = sorted(pet_info, key=lambda x: x[0])
                        amyloid, tracer, amy_date, age, sex, apoe, edu, mmse, centiloid = pet_info[0][1], pet_info[0][2], pet_info[0][3],pet_info[0][4], pet_info[0][5], pet_info[0][6], pet_info[0][7], pet_info[0][8], pet_info[0][9]

                entry = DatasetEntry(
                    instances=instances,
                    date=str(amy_date),
                    patient_id=patient,
                    amyloid=centiloid if use_centiloid and centiloid is not None else amyloid,
                    tracer=tracer,
                    centiloid=True if use_centiloid and centiloid is not None else False,
                    dataset="ADNI",
                    age=age,
                    sex=sex,
                    apoe=apoe,
                    edu=edu,
                    mmse=mmse
                    )
                
                self.entries.append(entry)

    def from_HABS(self, habs_dir: Union[str, Path], csv_file: Union[str, Path]):
        labels = self._load_labels_HABS(csv_file)

        data = glob(str(Path(habs_dir) / Path("*/*/*.nii.gz"))) + glob(str(Path(habs_dir) / Path("*/*/*.nii")))
        data_dict = dict()
        for path in tqdm(data):
            path = Path(path)

            session = path.parent.name
            patient = path.parent.parent.name

            if patient not in data_dict.keys():
                data_dict[patient] = dict()
            if session not in data_dict[patient].keys():
                data_dict[patient][session] = list()
            data_dict[patient][session].append(path)
        
        for patient in tqdm(data_dict.keys()):
            for session in data_dict[patient].keys():
                instances = list()
                for path in data_dict[patient][session]:
                    session = path.parent.name
                    patient = path.parent.parent.name

                    if ("T1" in path.name or "MPRAGE" in path.name) and "brain_mask" not in str(path):
                        sequence = "MPRAGE"
                    elif "FLAIR" in path.name:
                        sequence = "FLAIR"
                    elif "brain_mask" in str(path):
                        sequence = "MASK"
                    elif "T2" in path.name and "star" not in path.name:
                        sequence = "T2"
                    elif "star" in path.name:
                        sequence = "T2_STAR"
                    else:
                        sequence = "UNKNOWN"

                    instance = DatasetInstance(
                        image_path=str(path),
                        sequence_type=sequence.upper(),
                        field_strength=None,
                        time=None,
                        date=session,
                        )
                    instances.append(instance)
                
                date = None
                if patient not in labels.keys():
                    tracer, label, centiloid, status = None, None, False, None
                else:
                    patient_list = labels[patient]
                    pet_info = list()
                    for label, tracer, centiloid, date, status in patient_list:
                        diff = abs((datetime.strptime(date, '%Y-%m-%d') - datetime.strptime(session, '%Y-%m-%d')).days)
                        if diff <= self.PET_MRI_THRESHOLD:
                            pet_info.append((diff, label, tracer, centiloid, date, status))
                    if len(pet_info) < 1:
                        tracer, label, centiloid, date, status = None, None, False, None, None
                    else:
                        # sort based on the difference between the PET and MRI scan
                        pet_info = sorted(pet_info, key=lambda x: x[0])
                        label, tracer, centiloid, date, status = pet_info[0][1], pet_info[0][2], pet_info[0][3], pet_info[0][4], pet_info[0][5]

                entry = DatasetEntry(
                    instances=instances,
                    date=date,
                    patient_id=patient,
                    amyloid=label,
                    tracer=tracer,
                    centiloid=centiloid,
                    dataset="HABS",
                    status=status,
                    )
                self.entries.append(entry)

    def _load_labels_ADNI(self, path):
        csv = pd.read_csv(path)
        # extract rows with a value in the label column
        labels = dict()

        for _, row in csv.iterrows():
            label = None
            tracer = None
            
            subject = row['PTID']
            session = row['EXAMDATE']

            try:
                centiloid = row['CENTILOIDS']
            except:
                centiloid = None

            try:
                age=row['AGE']
                sex=row['PTGENDER']
                sex = 1 if sex=="Male" else 2
                apoe=row['APOE4']
                edu=row['PTEDUCAT']
                mmse=row['MMSE']
            except:
                age=None
                sex=None
                apoe=None
                edu=None
                mmse=None

            date = datetime.strptime(session, '%Y-%m-%d') if session is not None else None

            if row["AV45"] is not None and np.isnan(row["AV45"]) == False:
                label = row['AV45']
                tracer = "AV45"
            elif row["PIB"] is not None and np.isnan(row["PIB"]) == False:
                label = row['PIB']
                tracer = "PIB"
            elif row["FBB"] is not None and np.isnan(row["FBB"]) == False:
                label = row['FBB']
                tracer = "FBB"

            if subject not in labels.keys():
                labels[subject] = list()
            
            labels[subject].append((label, tracer, date, age, sex, apoe, edu, mmse, centiloid))

        return labels
    
    def _load_labels_OASIS(self, path):
        csv = pd.read_csv(path)
        # extract rows with a value in the label column
        labels = dict()

        for _, row in csv.iterrows():
            label = None
            tracer = None
            age = None

            subject = row['subject_id']
            session = "ses-" + row['oasis_session_id'].split("_")[-1]
            tracer = row["tracer"]
            
            age = row['AGE']
            sex = row['SEX']
            apoe = row['APOE4']
            edu = row['EDU']
            mmse = row['MMSE']
            
            if not isinstance(tracer, str):
                tracer = None

            # non partial volume correction
            label = row["Centiloid_fSUVR_TOT_CORTMEAN"]
            label_rsf = row["Centiloid_fSUVR_rsf_TOT_CORTMEAN"]

            if subject not in labels.keys():
                labels[subject] = dict()

            labels[subject][session] = (label, tracer, label_rsf, age, sex, apoe, edu, mmse)
            #print(labels)
        
        return labels
    
    def _load_labels_Stanford(self, path):
        csv = pd.read_csv(path)

        labels = dict()
        tracer = "FBB"
        
        for _, row in csv.iterrows():
            subject = row['pet_scan_id']
            suvr = row['SUVrComposite']
            
            labels[subject] = (tracer, suvr)
        return labels

    def _load_labels_A4(self, path, centiloid: bool = True):
        csv = pd.read_csv(path)
        other_path = Path(path).parent / Path("STANFORD_A4_COMBINED.csv")
        csv_other = pd.read_csv(other_path)

        amy_status = dict()
        for _, row in csv_other.iterrows():
            subject = row['BID']
            amy_status[subject] = row['SCORE'].lower()

        labels = dict()
        for _, row in csv.iterrows():
            subject = row['BID']
            tracer = row['ligand']
            suvr = row['suvr_cer']
            if centiloid:
                suvr = row['centiloid']

            region = row['brain_region']
            if region != "Composite_Summary":
                continue
            
            if tracer == "Florbetapir":
                tracer = "AV45"
                
            try:
                age=row['AGE']
                sex=row['SEX']
                apoe=row['APOE4']
                edu=row['EDU']
                mmse=row['MMSE']
            except:
                age=None
                sex=None
                apoe=None
                edu=None
                mmse=None
            if subject not in amy_status.keys():
                status = None
            else:
                status = amy_status[subject]
            
            labels[subject] = (suvr, tracer, centiloid, age, sex, apoe, edu, mmse, status)
        return labels
    
    def _load_labels_HABS(self, path, centiloid: bool = False):
        csv = pd.read_csv(path)
        # (1.0366*X)-0.0265

        labels = dict()
        for _, row in csv.iterrows():
            subject = row['SubjIDshort']
            session = row['PIB_SessionDate']
            # study_num = row["StudyArc"]
            tracer = "PIB"
            #suvr = row['PIB_T1_SUVR_FLR'] #previous 
            suvr = row['PIB_FS_DVR_FLR']
            status = row["PIB_FS_DVR_Group"]
            # change from 40-60 to 50-70
            #suvr = 1.0366 * suvr - 0.0265
            suvr = 143.06 * suvr - 145.6
            centiloid = True

            status = "positive" if "+" in status else "negative"
            
            if subject not in labels.keys():
                labels[subject] = list()
            labels[subject].append((suvr, tracer, centiloid, session, status))
        return labels
    
    def to_bids(self, bids_dir: Union[str, Path]):
        # save data in the BIDS format
        to_bids = dict()
        
        for entry in self.entries:
            dataset = entry.dataset
            patient = entry.patient_id
            date = entry.date

            if dataset not in to_bids.keys():
                to_bids[dataset] = list()
            
            to_bids[dataset].append(patient)
        # remove duplicates
        for dataset in to_bids.keys():
            to_bids[dataset] = list(set(to_bids[dataset]))
        
        for dataset in to_bids.keys():
            for i, patient in tqdm(enumerate(to_bids[dataset], 1)):
                entries = [entry for entry in self.entries if entry.patient_id == patient and entry.dataset == dataset]
                # entries.sort(key=lambda x: x.date)

                date_counter = 1
                for entry in entries:
                    instance_dates = list(set([instance.date for instance in entry.instances]))
                    instance_dates.sort()
                    instance_date = instance_dates[0]

                    date = instance_date
                    if date is None:
                        date = date_counter
                        date_counter += 1

                    entry.to_bids(bids_dir, date=date, patient_num=i)
