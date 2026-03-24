import os
from xml.parsers.expat import model
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import default_collate

from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchgeo.datasets import RESISC45


from datasets.liveability import LBMLoader
from datasets.sun397 import SUN397
from graph_inference import GSATViGReasoning


grid_approach_paths = {
        "vig_stem": "Isotropic_ViG_stem_",
        "fishnetDownsample": "fishnetDownsample_stem_",
        "bagnetTiny": "bagnetTiny_stem_"
    }

def get_resize_transform(resize_size):
    return v2.Resize((resize_size, resize_size), antialias=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NormalizeImage(torch.nn.Module):
    def forward(self, inpt):
        EPSILON = 1e-10
        min, max = inpt.min(), inpt.max()
        inpt_transformed = (inpt - min) / (EPSILON + max - min)
        return inpt_transformed

def get_transforms(dataset_name, resize_size, mode="train"):
    if dataset_name == "Liveability":
        if mode == "train":
            return v2.Compose([v2.ToImage(),
                               v2.ToDtype(torch.uint8, scale=True),
                               get_resize_transform(resize_size),
                               v2.RandAugment(),
                               v2.RandomErasing(p=0.25),
                               v2.ToDtype(torch.float32, scale=True),
                               NormalizeImage()])
        else:
            return v2.Compose([v2.ToImage(),
                               v2.ToDtype(torch.uint8, scale=True),
                               get_resize_transform(resize_size),
                               v2.ToDtype(torch.float32, scale=True),
                               NormalizeImage()])
    else:
        if mode == "train":
            transforms = v2.Compose([v2.ToDtype(torch.uint8),
                                     get_resize_transform(resize_size),
                                     v2.RandAugment(),
                                     v2.RandomErasing(p=0.25),
                                     v2.ToDtype(torch.float32, scale=True),
                                     NormalizeImage()])
        else:
            transforms = v2.Compose([v2.ToDtype(torch.uint8),
                                     get_resize_transform(resize_size),
                                     v2.ToDtype(torch.float32, scale=True),
                                     NormalizeImage()])
        def transforms_wrapper(x):
            if x["image"].shape[0] != 3:
                print("Error in image shape")
                print(x["ids"], x["image"].shape)
            image_transformed = transforms(x["image"])
            if image_transformed.shape[0] != 3:
                print("Error in transformed image shape")
                print(x["ids"],image_transformed.shape)
            return {"image": image_transformed, "label": x["label"]}
        return transforms_wrapper

def get_liveability_data_loaders(dataset_name, batch_size=32):
    num_classes = 1
    image_size = 256
    dataset_root_dir = "/home/datasets/{}/".format(dataset_name)
    # dataset_root_dir = "C:/Users/datasets/{}/".format(dataset)
    dataset_info_file = os.path.join(dataset_root_dir, "grid_geosplit_not_rescaled.geojson")

    lbm_data_module = LBMLoader(n_workers=8, batch_size=batch_size)
    lbm_data_module.setup_data_classes(dataset_info_file,
                                       dataset_root_dir,
                                       splits=['train', 'val', "test"],
                                       train_transforms=get_transforms(dataset_name, image_size, "train"),
                                       val_transforms=get_transforms(dataset_name, image_size, "val"),
                                       test_transforms=get_transforms(dataset_name, image_size, "test"))

    return num_classes, image_size, lbm_data_module.train_dataloader(), lbm_data_module.val_dataloader(), lbm_data_module.test_dataloader()

def get_resisc45_data_loaders(dataset_name, batch_size=32, num_workers=8, use_transforms=True):
    num_classes = 45
    image_size = 256
    #dataset_root_dir = "/home/datasets/{}".format(dataset_name)
    dataset_root_dir = "/home/datasets/"
    train_transforms = get_transforms(dataset_name, image_size, "train") if use_transforms else None
    val_transforms = get_transforms(dataset_name, image_size, "val") if use_transforms else None
    test_transforms = get_transforms(dataset_name, image_size, "test") if use_transforms else None

    train_dataset = RESISC45(root=dataset_root_dir, split="train", transforms=train_transforms, download=False)
    val_dataset = RESISC45(root=dataset_root_dir, split="val", transforms=val_transforms, download=False)
    test_dataset = RESISC45(root=dataset_root_dir, split="test", transforms=test_transforms, download=False)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return num_classes, image_size, train_data_loader, val_data_loader, test_data_loader

def getSUN397(batch_size=32, num_workers=8, use_transforms=True):
    num_classes = 397
    image_size = 256
    dataset_root_dir = "/home/datasets/SUN397/SUN397/"
    dataset_name = "sun397"
    train_transforms = get_transforms(dataset_name, image_size, "train") if use_transforms else None
    val_transforms = get_transforms(dataset_name, image_size, "val") if use_transforms else None
    test_transforms = get_transforms(dataset_name, image_size, "test") if use_transforms else None

    train_dataset = SUN397(root_dir=dataset_root_dir, split="Training_01.txt", transforms=train_transforms)
    val_dataset = SUN397(root_dir=dataset_root_dir, split="Testing_01.txt", transforms=val_transforms)
    test_dataset = SUN397(root_dir=dataset_root_dir, split="Testing_02.txt", transforms=test_transforms)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return num_classes, image_size, train_data_loader, val_data_loader, test_data_loader


def load_trained_model(model, model_root_dir, model_checkpoint_path):
   print("Loading model from: {}, {}".format(model_root_dir, model_checkpoint_path))
   model_path = os.path.join(model_root_dir, model_checkpoint_path)
   state_dict = torch.load(model_path, map_location=device)["state_dict"]
   new_state_dict = {}
   for k, v in state_dict.items():
        k = k.replace("model.", "")
        if k.startswith("gnn_bottleneck.gnn_"):
            k = k.replace("gnn_bottleneck.gnn_", "gnn_bottleneck.gnn_model.")
        new_state_dict[k] = v
   model.load_state_dict(new_state_dict)
   model.eval()
   if torch.cuda.is_available():
        model.cuda()

   return model
