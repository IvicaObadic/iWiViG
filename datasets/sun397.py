import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torchvision


class SUN397(Dataset):
    def __init__(self, root_dir, split, transforms):
        # Set params
        self.root_dir = root_dir
        with open(os.path.join(root_dir, split), 'r') as file:
            instances_names = file.readlines()
        
        class_ids = []
        with open(os.path.join(root_dir, "ClassName.txt"), 'r') as file:
            classes = file.readlines()
            for class_name in classes:
                class_ids.append(class_name.strip().split("/")[2])

        print(class_ids)
        self.dataset_instances = []
        self.class_idx = []
        for i, instance_name in enumerate(instances_names):
            instance_name = instance_name.strip()
            instance_class = instance_name.split("/")[2]
            self.dataset_instances.append(instance_name)
            self.class_idx.append(class_ids.index(instance_class))
        
        self.transforms = transforms


    def __getitem__(self, index):
        # Load labels
        image_id = self.dataset_instances[index]
        image_path = os.path.join(self.root_dir, image_id[1:])
        image = Image.open(image_path).convert('RGB') 
        image = torchvision.transforms.functional.pil_to_tensor(image)
        label = self.class_idx[index]

        item = {'ids': image_id,
                'image': image,
                'label': label}
        
        if self.transforms is not None:
            item = self.transforms(item)
        
        return item

    def __len__(self):
        return len(self.dataset_instances)
    
if __name__ == '__main__':
    dataset = SUN397(root_dir='/home/datasets/SUN397/SUN397', split='Training_01.txt', transforms=None)
    print(dataset[0])
    print(len(dataset))