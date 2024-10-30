from VisualAcuityDataset import VisualAcuityDataset
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

# Load the original dataset
original_dataset = CIFAR10(root='data', train=True, download=True)

# Create the VisualAcuityDataset with a specific month_age
month_ages = list(range(0, 13))  # Create a list of ages from 0 to 12
va_datasets = []
for month_age in month_ages:
    va_datasets.append(VisualAcuityDataset(original_dataset, month_age=month_age))

# Now you can use va_dataset with a DataLoader
from torch.utils.data import DataLoader

# Create DataLoaders for each month_age
data_loaders = []  # List to hold DataLoaders
for month_age in month_ages:
    data_loader = DataLoader(va_datasets[month_age], batch_size=32, shuffle=False)
    data_loaders.append(data_loader)

def show_first_image_from_loader(data_loader):
    # Get the first batch from the data loader
    for images, _ in data_loader:
        # Get the first image from the batch
        first_image = images[0]
        # Convert the tensor to a numpy array and transpose it to (H, W, C) for displaying
        first_image_np = first_image.permute(1, 2, 0).numpy()
        # Display the image
        plt.imshow(first_image_np)
        plt.axis('off')  # Turn off axis
        plt.show()
        break  # Only show the first image

# Call the function to display the first image from data_loader[0]
show_first_image_from_loader(data_loaders[0])
show_first_image_from_loader(data_loaders[4])
show_first_image_from_loader(data_loaders[8])
show_first_image_from_loader(data_loaders[11])
print(len(data_loaders), data_loaders)
pass
