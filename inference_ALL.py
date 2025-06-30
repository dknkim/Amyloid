import argparse
from pathlib import Path

import torch
import monai
import numpy as np

from amy.dataset.dataloader import Loader
from amy.models.efficientnet_model_b3 import EfficientNet
from amy.dataset.datasetcontainer import DatasetContainer

from amyloid.utils import valid_transforms

parser = argparse.ArgumentParser()
parser.add_argument("-testfold", type=int, required=True)


args = parser.parse_args()
testfold = int(args.testfold)

fold = "test"  

method = "classification" 

tracers = ["AV45", "FBB", "FBP"]

if method not in ["classification", "regression"]:
    raise ValueError("Method must be either classification or regression, not {}".format(method))

if fold == "test":
    a4 = DatasetContainer.from_json("dataset/a4/test.json")
    adni = DatasetContainer.from_json("dataset/adni/test.json")
    oasis = DatasetContainer.from_json("dataset/oasis/test.json")
else:
    a4 = DatasetContainer.from_json("dataset/a4/fold_{}/valid.json".format(fold))
    adni = DatasetContainer.from_json("dataset/adni/fold_{}/valid.json".format(fold))
    oasis = DatasetContainer.from_json("dataset/oasis/fold_{}/valid.json".format(fold))

dataset = a4 + adni + oasis


device = "cuda:7"

scans = [["MPRAGE", "FLAIR"],["MPRAGE"]]
cufoffs = Loader.CUTOFF

config = {
   
    "scans": scans,
    "top_percentile": None,
    "brain_extracted": True,
    "clinical_data": None,
    "encode_data": False,
    }
results = dict()
for scans in [["MPRAGE", "FLAIR"],["MPRAGE"]]:
    #del valid_loader, model, 
    valid_loader = Loader(
        container=dataset, 
        scans=["MPRAGE", "FLAIR"],
        tracer=tracers,
        transforms=valid_transforms,
        classification=True if method == "classification" else False,
        return_patient_date=True,
        )
    valid_loader.scans = scans
  


    model = EfficientNet(
        in_channels=len(scans),
        channels=320, 
        num_classes=1,
        dropout=0.2,
        )
  
    
    model_path = "weights/{}/fold_{}/2024/best_balanced_accuracy/checkpoint-best.pth".format("_".join(scans),testfold)

    model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])
    model.eval()

    model.to(device)

    scan = "_".join(scans)
    results[scan] = dict()
    with torch.no_grad():
        for j, (img, target, patient, date, dataset) in enumerate(valid_loader):
        #for img, target, patient, date in valid_loader:
            img = img.unsqueeze(0)
            img = img.to(device)

            out = model(img).cpu()

            
            out = torch.sigmoid(out).cpu()
            #out = out.item()[0]
            out = out[0].item()
            target = target[0].item()

            if dataset not in results[scan]:
                results[scan][dataset] = dict()
            if patient not in results[scan][dataset]:
                results[scan][dataset][patient] = dict()
            if date not in results[scan][dataset][patient]:
                results[scan][dataset][patient][date] = list()
            results[scan][dataset][patient][date].append((out, target))


    # save_to_json
    import json
    if fold == "test":
        name = "test_{}_MPRAGE_FLAIR_fold_{}.json".format(method,testfold)
    else:
        name = "valid_fold_{}_MPRAGE_FLAIR.json".format(fold)

    save = Path("eval/{}/".format(method))
    save.mkdir(exist_ok=True, parents=True)

    with open(save / name, "w") as f:
        json.dump(results, f, indent=4)

