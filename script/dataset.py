import os 
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFilter

class RoadSegmentationDataset(Dataset):
    def __init__(self, data_dir, image_folder='images', mask_folder='groundtruth', transform=None, transform_x=None):
        """
        Args:
            data_dir (str): Root directory of the dataset.
            image_folder (str): Name of the folder containing road images.
            mask_folder (str): Name of the folder containing segmentation masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.image_folder = os.path.join(data_dir, image_folder)
        self.mask_folder = os.path.join(data_dir, mask_folder)
        self.transform = transform
        self.transform_x = transform_x

        # Ensure the folders exist
        assert os.path.exists(self.image_folder), f"Image folder '{self.image_folder}' not found."
        assert os.path.exists(self.mask_folder), f"Mask folder '{self.mask_folder}' not found."

        # Get the list of file names in both folders
        self.image_list = sorted(os.listdir(self.image_folder))
        self.mask_list = sorted(os.listdir(self.mask_folder))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load image and mask

        img_name = self.image_list[idx]
        mask_name = self.image_list[idx]

        img_path = os.path.join(self.image_folder, img_name)
        mask_path = os.path.join(self.mask_folder, mask_name)

        image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)  # Adjust as needed
        mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32)    # Assuming grayscale mask, adjust as needed
        mask = mask/np.max(mask)

        if self.transform_x:
             image = self.transform_x(image)

        # Apply transformations if provided
        if self.transform:
            augmentations=self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        mask=mask.unsqueeze(0)
        return (image,mask)

def standardize(image):
    means = [np.mean(image.astype(np.float32)[:,:,i]) for i in range(3)]
    stds = [np.std(image.astype(np.float32)[:,:,i]) for i in range(3)]

    for i in range(len(means)):
        image[:,:,i] = (image[:,:,i]-means[i])/stds[i]
    return(image)