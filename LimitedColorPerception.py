import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class LimitedColorPerception(Dataset):
    def __init__(self, dataset, month_age, max_age=12.0, max_sigma=5.0):
        """
        Initializes the dataset with flexible visual acuity.

        Args:
            dataset (Dataset): The original dataset to wrap.
            month_age (float): The age in months of the subject.
            max_age (float, optional): The age in months when full visual acuity is reached. Defaults to 12.0.
            max_sigma (float, optional): The maximum sigma value for Gaussian blur. Defaults to 5.0.
        """
        self.dataset = dataset
        self.month_age = month_age
        self.max_age = max_age
        self.max_sigma = max_sigma

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index after applying the color perception transformation.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: (transformed image, label)
        """
        image, label = self.dataset[idx]

        # Ensure the image is in RGB format
        if isinstance(image, Image.Image):
            image = image.convert('RGB')

        # Apply the transformation
        image = self.transform(image)
        return image, label
    
