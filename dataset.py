import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, image_dir, transform=None, include_class=None, transform_class=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
            data_augmentation (bool, optional): Whether to apply data augmentation. Defaults to False.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

        if include_class:
            self.image_files = [f for f in self.image_files if int(f.split("_")[0]) in include_class]
        # Apply default transform if provided
        self.transform = transform
        self.transform_class = transform_class


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_name)
            image = Image.open(img_path).convert('RGB')

            # Extract class from filename (assumes format "class_id.jpg")
            class_id = int(img_name.split('_')[0])
            # Apply default transformations
            if self.transform:
                image = self.transform(image)
            if self.transform_class:
                class_id = self.transform_class[class_id]
            return image, class_id
        except Exception as e:
            print(f"Error loading image {img_name}: {e}. Skipping...")
            return self.__getitem__((idx + 1) % len(self.image_files))

# Example usage
if __name__ == "__main__":
    image_dir = "data/train"

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset with data augmentation enabled
    dataset = FaceDataset(image_dir=image_dir, transform=transform, data_augmentation=True)
    print(len(dataset))