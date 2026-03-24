import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from osgeo import gdal
from PIL import Image
import cv2

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


# dataset_root_dir = "C:/Users/datasets/Liveability/"
# dataset_info_file = os.path.join(dataset_root_dir, "grid_geosplit_not_rescaled.geojson")

#based on github.com/ahlevering/liveability-rs/blob/master/codebase/pt_funcs/dataloaders.py

class LBMDataContainer():
    def __init__(self, splits_file):
        self.labels = gpd.read_file(splits_file)
        self.labels = self.labels.set_crs(28992, allow_override=True)

        self.labels['geometry'] = self.labels.apply(self.fix_polygon, axis=1)
        # self.labels.set_geometry('geometry', inplace=True)

    def fix_polygon(self, row):
        centroid = row['geometry'].centroid
        grid_true_center_x = centroid.xy[0][0] - (centroid.xy[0][0] % 100) + 50
        grid_true_center_y = centroid.xy[1][0] - (centroid.xy[1][0] % 100) + 50
        return Point([grid_true_center_x, grid_true_center_y]).buffer(50, cap_style=3)


class LBMData(Dataset):
    def __init__(self, exp_data, ids, imgs_root, dim_scores, transforms):
        # Set params
        self.images_root = imgs_root
        self.transforms = transforms
        self.dim_scores = dim_scores
        self.exp_data = exp_data
        self.split_ids = ids

    def load_aerial_img(self, datapoint):
        patch_path = f"{self.images_root}{datapoint['region_name'].item()}/{datapoint['gridcode'].item()}.tiff"
        if not os.path.exists(patch_path):
            print("A file can't be found {}".format(patch_path))
        patch = np.array(gdal.Open(patch_path).ReadAsArray()).transpose([1,2,0])
        patch = Image.fromarray(patch)
        return patch

    def __getitem__(self, index):
        # Load labels
        point_id = self.split_ids[index]
        datapoint = self.exp_data.labels[self.exp_data.labels['gridcode'] == point_id]

        patch = self.load_aerial_img(datapoint)
        patch = self.transforms(patch)

        lbm_score = float(datapoint['rlbrmtr'].item())
        dim_scores = []
        if self.dim_scores is not None and len(self.dim_scores) > 0:
            for dim_score in self.dim_scores:
                dim_scores.append(float(datapoint[dim_score].item()))
        else:
            dim_scores = [0]

        lbm_pt_centroid = datapoint.centroid
        lat = lbm_pt_centroid.y.item()
        lon = lbm_pt_centroid.x.item()

        return {'ids': point_id,
                'lat': lat,
                'lon': lon,
                'image': patch,
                'dim_scores': dim_scores,
                'label': lbm_score
            }

    def __len__(self):
        return len(self.split_ids)

class LBMLoader(pl.LightningDataModule):
    def __init__(self, n_workers, batch_size, data_class=LBMData):
        super().__init__()
        self.batch_size = batch_size
        self.workers = n_workers
        self.data_class = data_class

        # self.dims = None
        self.splits = None
        self.exp_data = None

    def setup_data_classes(self, splits_file, imgs_root, dim_scores=None, splits=['train', 'val'], train_transforms=None, val_transforms=None, test_transforms=None):
        self.exp_data = LBMDataContainer(splits_file)
        print("Reading the dataset from {} and the labels from {}".format(imgs_root, splits_file))
        if 'all' in splits:
            all_ids = self.exp_data.labels['gridcode'].to_list()
            self.test_data = self.data_class(self.exp_data,
                                             all_ids,
                                             imgs_root,
                                             dim_scores,
                                             transforms=test_transforms)

        if 'test' in splits:
            test_split_labels = self.exp_data.labels[self.exp_data.labels['split'].isin(['test'])]
            print("Number of test examples: {}".format(len(test_split_labels)))

            # test_split_labels = self.exp_data.labels[self.exp_data.labels['region_name'].isin(['beesel_2016'])]
            test_ids = test_split_labels['gridcode'].to_list()
            self.test_data = self.data_class(self.exp_data,
                                             test_ids,
                                             imgs_root,
                                             dim_scores,
                                             transforms=test_transforms)

        if 'val' in splits:
            val_split_labels = self.exp_data.labels[self.exp_data.labels['split'].isin(['val'])]
            # val_split_labels = self.exp_data.labels[self.exp_data.labels['region_name'].isin(['weert_2016'])]
            val_ids = val_split_labels['gridcode'].to_list()
            print("Number of validation examples: {}".format(len(val_ids)))
            self.val_data = self.data_class(self.exp_data,
                                            val_ids,
                                            imgs_root,
                                            dim_scores,
                                            transforms=val_transforms)

        if 'train' in splits:
            train_split_labels = self.exp_data.labels[self.exp_data.labels['split'].isin(['train'])]
            # train_split_labels = self.exp_data.labels[self.exp_data.labels['region_name'].isin(['eindhoven_2016'])]
            train_ids = train_split_labels['gridcode'].to_list()
            print("Number of training examples: {}".format(len(train_ids)))
            self.train_data = self.data_class(self.exp_data,
                                              train_ids,
                                              imgs_root,
                                              dim_scores,
                                              transforms=train_transforms)
        self.splits = splits

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.workers, shuffle=False)
