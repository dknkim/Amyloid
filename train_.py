import os
import argparse
from pathlib import Path

import torch
import monai

from amy.dataset.datasetcontainer import DatasetContainer
from amy.dataset.dataloader import Loader
from amy.trainer.trainer import Trainer

from amy.models.efficientnet_model_b3 import EfficientNet
from utils import train_transforms, valid_transforms

parser = argparse.ArgumentParser()
parser.add_argument("-scans", nargs="+", type=str, required=True)
parser.add_argument("-fold", type=int, required=True)
parser.add_argument("-data", nargs="+", type=str, required=False, default=["adni", "oasis", "a4"])
parser.add_argument("-model", type=str, required=False, default="efficientnet")
parser.add_argument("-encode_data", type=int, required=False, default=0)
parser.add_argument("-hist_norm", type=int, required=False, default=0)
parser.add_argument("-gpu", type=int, required=True)

args = parser.parse_args()
gpu = int(args.gpu)
fold = int(args.fold)
scans = args.scans
data = list(args.data)
data = [d.lower() for d in data]
model_name = args.model
encode_data = bool(args.encode_data)
hist_norm = bool(args.hist_norm)

scans = [scan.upper() for scan in scans]


save_dir = "weights/{}/fold_{}".format("_".join(scans), fold) 

if encode_data:
    name = f"{model_name}_classification_{'_'.join(scans)}_fold_{fold}_encode"
else:
    name = f"{model_name}_classification_{'_'.join(scans)}_fold_{fold}_eff_b3_rot"

if hist_norm:
    name = f"{name}_hist_norm"
    save_dir = f"{save_dir}_hist_norm"

config = {
    "name": name,
    "epochs": 500,
    "iterative": False,
    "images_pr_iteration": 1,
    "val_images_pr_iteration": 1,
    "inputs_pr_iteration": 250,
    "batch_size": 8,
    "learning_rate": 5e-4,
    "optimizer": "AdamW",
    "lr_scheduler": "CosineAnnealingLR",
    "save_dir": save_dir,
    "save_period": 100,
    "weight_decay": 1e-5,
    "scans": scans,
    "histogram_normalize": hist_norm,
    "top_percentile": None,
    "brain_extracted": True,
    "clinical_data": None,
    "data": data,
    "model": model_name,
    "encode_data": encode_data,
    }



adni_train = DatasetContainer.from_json("dataset/adni/fold_{}/train.json".format(fold))
adni_valid = DatasetContainer.from_json("dataset/adni/fold_{}/valid.json".format(fold))

oasis_train = DatasetContainer.from_json("dataset/oasis/fold_{}/train.json".format(fold))
oasis_valid = DatasetContainer.from_json("dataset/oasis/fold_{}/valid.json".format(fold))

a4_train = DatasetContainer.from_json("dataset/a4/fold_{}/train.json".format(fold))
a4_valid = DatasetContainer.from_json("dataset/a4/fold_{}/valid.json".format(fold))

train = DatasetContainer()


if "adni" in data:
    train += adni_train
if "oasis" in data:
    train += oasis_train
if "a4" in data:
    train += a4_train
print("ADNI train length:", len(adni_train), "OASIS train length:", len(oasis_train), "A4 train length:", len(a4_train))
print("Train length:", len(train))
print("ADNI valid length:", len(adni_valid), "OASIS valid length:", len(oasis_valid), "A4 valid length:", len(a4_valid))



def change_to_mask_type(dataset):
    for entry in dataset:
        for instance in entry:
            if "brain_mask" in instance.image_path:
                instance.sequence_type = "MASK"

change_to_mask_type(train)

change_to_mask_type(adni_valid)
change_to_mask_type(oasis_valid)
change_to_mask_type(a4_valid)


tracers = ["AV45", "FBB", "FBP"]

train_loader = Loader(
    container=train,
    scans=["MPRAGE", "FLAIR"],
    tracer=tracers,
    transforms=train_transforms,
    classification=True,
    histogram_normalize=False,
    brain_mask_centering=False,
    top_percentile=config["top_percentile"],
    encode_dataset=config["encode_data"],
    clip=False,
    # override_cutoff=28.99,
    use_status=False,
    )

adni_valid_loader = Loader(
    container=adni_valid,
    scans=["MPRAGE", "FLAIR"],
    tracer=tracers,
    transforms=valid_transforms,
    classification=True,
    histogram_normalize=False,
    brain_mask_centering=False,
    top_percentile=config["top_percentile"],
    encode_dataset=config["encode_data"],
    clip=False,
    use_status=False,
    )
oasis_valid_loader = Loader(
    container=oasis_valid,
    scans=["MPRAGE", "FLAIR"],
    tracer=tracers,
    transforms=valid_transforms,
    classification=True,
    histogram_normalize=False,
    brain_mask_centering=False,
    top_percentile=config["top_percentile"],
    encode_dataset=config["encode_data"],
    clip=False,
    use_status=False,
    )
a4_valid_loader = Loader(
    container=a4_valid,
    scans=["MPRAGE", "FLAIR"],
    tracer=tracers,
    transforms=valid_transforms,
    classification=True,
    histogram_normalize=False,
    brain_mask_centering=False,
    top_percentile=config["top_percentile"],
    encode_dataset=config["encode_data"],
    clip=False,
    use_status=False,
    )


train_cases = len(train_loader)
valid_cases = len(adni_valid_loader) + len(oasis_valid_loader) + len(a4_valid_loader) 

config["train_cases"] = train_cases
config["valid_cases"] = valid_cases

if len(config["scans"]) == 1:
    train_loader.scans = config["scans"]

    adni_valid_loader.scans = config["scans"]
    oasis_valid_loader.scans = config["scans"]
    a4_valid_loader.scans = config["scans"]


print("Train Loader length:", len(train_loader))
print("ADNI valid Loader length:", len(adni_valid_loader), "OASIS valid Loader length:", len(oasis_valid_loader), "A4 valid Loader length:", len(a4_valid_loader))


model = EfficientNet(
    in_channels=len(scans),
    channels=320, 
    num_classes=1,
    dropout=0.2,
    )

# print model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
config["total_params"] = total_params
#12,184,930 total parameter

# BCE with logits loss
loss = torch.nn.BCEWithLogitsLoss()

train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                           num_workers=16,
                                           batch_size=config["batch_size"],
                                           shuffle=True,
                                           )

adni_valid_loader = torch.utils.data.DataLoader(dataset=adni_valid_loader,
                                           num_workers=16,
                                           batch_size=config["batch_size"],
                                           shuffle=False,
                                           )
oasis_valid_loader = torch.utils.data.DataLoader(dataset=oasis_valid_loader,
                                             num_workers=16,
                                             batch_size=config["batch_size"],
                                             shuffle=False,
                                             )
a4_valid_loader = torch.utils.data.DataLoader(dataset=a4_valid_loader,
                                            num_workers=16,
                                            batch_size=config["batch_size"],
                                            shuffle=False,
                                            )


print(len(train_loader), len(adni_valid_loader) + len(oasis_valid_loader) + len(a4_valid_loader) )

valid_loader = {"ADNI": adni_valid_loader, "OASIS": oasis_valid_loader, "A4": a4_valid_loader}


class LRPolicy(object):
    def __init__(self, initial, warmup_steps=10):
        self.warmup_steps = warmup_steps
        self.initial = initial

    def __call__(self, step):
        return self.initial + step/self.warmup_steps*(1 - self.initial)

warmup_steps = 20

optimizer = torch.optim.AdamW(params=model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer,  LRPolicy(initial=1e-2, warmup_steps=warmup_steps))
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(warmup_steps - config["epochs"]))

lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])


metrics = {
    "BCE": torch.nn.BCEWithLogitsLoss(),
    }

trainer = Trainer(
    model=model,
    loss_function=loss,
    metric_ftns=metrics,
    optimizer=optimizer,
    config=config,
    data_loader=train_loader,
    valid_data_loader=valid_loader,
    lr_scheduler=lr_scheduler,
    seed=None,
    # log_step=50,
    device="cuda:{}".format(gpu),
    mixed_precision=True,
    tags=["3D", "classification", "{}".format(model_name), "amyloid", ""],
    project="",
    classification=True,
    crop_foreground=False,
    entity=""
    )

trainer.train()