from VisualAcuity import VisualAcuityDataset
from LimitedColor import LimitedColorPerceptionDataset
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt

# Load the original dataset
original_dataset = CIFAR10(root='data', train=True, download=True)

# Create the VisualAcuityDataset with a specific month_age
month_ages = list(range(0, 13))  # Create a list of ages from 0 to 12
va_datasets = []
lc_datasets=[]
for month_age in month_ages:
    va_datasets.append(VisualAcuityDataset(original_dataset, month_age=month_age))
    lc_datasets.append(LimitedColorPerceptionDataset(original_dataset, month_age=month_age))

# Now you can use va_dataset with a DataLoader
from torch.utils.data import DataLoader
x=input("Property:")
data_loaders = []  # List to hold DataLoaders
# Create DataLoaders for each month_age
if x.lower().strip()=='va':
    for month_age in month_ages:
        data_loader=DataLoader(va_datasets[month_age], batch_size=32, shuffle=False)
        data_loaders.append(data_loader)
elif x.lower().strip()=='lc':
    for month_age in month_ages:
        data_loader=DataLoader(lc_datasets[month_age], batch_size=32, shuffle=False)
        data_loaders.append(data_loader)

def show_images_in_sequence(data_loaders):
    plt.figure(figsize=(25,25))
    for month_age,data_loader in zip(month_ages,data_loaders):
        for images,_ in data_loader:
            first_image=images[13].permute(1, 2, 0).numpy()*255
            first_image=first_image.astype(np.uint8)
            
            plt.subplot(1,len(data_loaders),month_age+1)
            plt.imshow(first_image)
            plt.axis('off')
            plt.title(f"{month_age} months")
            break
        
    plt.show()
print(data_loaders)
show_images_in_sequence(data_loaders)
