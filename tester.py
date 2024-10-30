from torchvision.datasets import CIFAR10

# Load the original dataset
original_dataset = CIFAR10(root='data', train=True, download=True)

# Create the VisualAcuityDataset with a specific month_age
month_age = 6  # For example, 6 months old
va_dataset = VisualAcuityDataset(original_dataset, month_age=month_age)

# Now you can use va_dataset with a DataLoader
from torch.utils.data import DataLoader

data_loader = DataLoader(va_dataset, batch_size=32, shuffle=True)

# Iterate over the data_loader
for images, labels in data_loader:
    # Your training code here
    pass