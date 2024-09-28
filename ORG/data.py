import os
import json
from PIL import Image
from collections import defaultdict
import random
import torch
import torchvision
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, Food101, STL10
from torch.utils.data import Subset, DataLoader, Dataset
import torchvision.transforms as transforms 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
])

mnist_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# class ImageNetSubsetDataset(Dataset):
#     def __init__(self, root_dir, split='train', transform=None):
#         """
#         Args:
#             root_dir (str): Path to the root directory containing the dataset.
#             split (str): 'train' or 'val' to indicate the dataset split.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.root_dir = root_dir
#         self.split = split
#         self.transform = transform
        
#         # Load the label mappings from label.json
#         with open(os.path.join(root_dir, 'Labels.json'), 'r') as f:
#             self.label_map = json.load(f)
        
#         # Map class names to indices
#         self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.label_map.keys())}
#         # print(self.class_to_idx)
        
#         # Gather all image paths and their corresponding labels
#         self.image_paths = []
#         self.labels = []
        
#         if split == 'train':
#             subfolders = ['train.X1', 'train.X2', 'train.X3', 'train.X4']
#         elif split == 'val':
#             subfolders = ['val.X']
#         else:
#             raise ValueError("Split must be 'train' or 'val'")
        
#         for subfolder in subfolders:
#             folder_path = os.path.join(root_dir, subfolder)
#             for class_name in os.listdir(folder_path):
#                 class_folder = os.path.join(folder_path, class_name)
#                 if os.path.isdir(class_folder):
#                     for img_name in os.listdir(class_folder):
#                         img_path = os.path.join(class_folder, img_name)
#                         self.image_paths.append(img_path)
#                         self.labels.append(self.class_to_idx[class_name])
        
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         label = self.labels[idx]
#         image = Image.open(img_path).convert('RGB')
        
#         if self.transform:
#             image = self.transform(image)
            
#         image = image.type(torch.float32)
#         label = torch.tensor(label, dtype=torch.long)
        
#         return image, label

class ImageNetSubsetDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment_transform=None, train_size=1100):
        """
        Args:
            root_dir (str): Path to the root directory containing the dataset.
            transform (callable, optional): Transform to be applied to original images.
            augment_transform (callable, optional): Transform to be applied to augmented images.
            train_size (int): Number of original images per class for the training dataset.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.augment_transform = augment_transform
        self.train_size = train_size

        # Load the label mappings from Labels.json
        with open(os.path.join(root_dir, 'Labels.json'), 'r') as f:
            self.label_map = json.load(f)

        # Map class names to indices
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.label_map.keys())}
        
        # Combine images from both train and val datasets
        self.image_paths = defaultdict(list)
        self.labels = defaultdict(list)

        subfolders = ['train.X1', 'train.X2', 'train.X3', 'train.X4', 'val.X']

        for subfolder in subfolders:
            folder_path = os.path.join(root_dir, subfolder)
            for class_name in os.listdir(folder_path):
                class_folder = os.path.join(folder_path, class_name)
                if os.path.isdir(class_folder):
                    for img_name in os.listdir(class_folder):
                        img_path = os.path.join(class_folder, img_name)
                        self.image_paths[class_name].append(img_path)
                        self.labels[class_name].append(self.class_to_idx[class_name])

        # Shuffle images within each class
        for class_name in self.image_paths:
            combined = list(zip(self.image_paths[class_name], self.labels[class_name]))
            random.shuffle(combined)
            self.image_paths[class_name], self.labels[class_name] = zip(*combined)

        # Split into train and val datasets
        self.train_image_paths = []
        self.train_labels = []
        self.val_image_paths = []
        self.val_labels = []

        for class_name in self.image_paths:
            self.train_image_paths.extend(self.image_paths[class_name][:train_size])
            self.train_labels.extend(self.labels[class_name][:train_size])
            self.val_image_paths.extend(self.image_paths[class_name][train_size:])
            self.val_labels.extend(self.labels[class_name][train_size:])
        
        # Duplicate for augmented images
        self.train_image_paths = self.train_image_paths * 2
        self.train_labels = self.train_labels * 2
        self.val_image_paths = self.val_image_paths * 2
        self.val_labels = self.val_labels * 2

    def __len__(self):
        return len(self.train_image_paths) + len(self.val_image_paths)

    def __getitem__(self, idx):
        if idx < len(self.train_image_paths):
            img_path = self.train_image_paths[idx]
            label = self.train_labels[idx]
            is_augmented = idx >= (len(self.train_image_paths) // 2)
        else:
            val_idx = idx - len(self.train_image_paths)
            img_path = self.val_image_paths[val_idx]
            label = self.val_labels[val_idx]
            is_augmented = val_idx >= (len(self.val_image_paths) // 2)
        
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation if necessary
        if is_augmented and self.augment_transform:
            image = self.augment_transform(image)
        elif self.transform:
            image = self.transform(image)
        
        image = image.type(torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

original_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])

augment_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])    

# BATCH_SIZE = 64

def prepare_dataloader(args):
    if args['dataset_name'] == "cifar10":
        print("With CIFAR10")
        cifar10_train_dataset = CIFAR10(root = "./data/cifar10", train = True, transform = transform, download = True)
        cifar10_test_dataset = CIFAR10(root = "./data/cifar10", train = False, transform = transform, download = True)
        
        train_dataloader = DataLoader(cifar10_train_dataset, batch_size = args['batch_size'], shuffle = True, drop_last = True)
        test_dataloader = DataLoader(cifar10_test_dataset, batch_size = args['batch_size'], shuffle = False, drop_last = True)
        
    elif args['dataset_name'] == "mnist":
        print("With MNIST")
        mnist_train_dataset = MNIST(root = './data', train = True, transform = mnist_transform, download = True)
        mnist_test_dataset = MNIST(root = './data', train = False, transform = mnist_transform, download = True)
        
        train_dataloader = DataLoader(mnist_train_dataset, batch_size = args['batch_size'], shuffle = True, drop_last = True)
        test_dataloader = DataLoader(mnist_test_dataset, batch_size = args['batch_size'], shuffle = False, drop_last = True)
        
    elif args['dataset_name'] == "fashionmnist":
        print("with Fashion")
        fashion_train_dataset = FashionMNIST(root = './data', train = True, transform = mnist_transform, download = True)
        fashion_test_dataset = FashionMNIST(root = './data', train = False, transform = mnist_transform, download = True)
        
        train_dataloader = DataLoader(fashion_train_dataset, batch_size = args['batch_size'], shuffle = True, drop_last = True)
        test_dataloader = DataLoader(fashion_test_dataset, batch_size = args['batch_size'], shuffle = False, drop_last = True)
        
    elif args['dataset_name'] == "stl10":
        print("with STL10")
        stl10_train_dataset = STL10(root = './data', split = "train", transform = transform, download = True)
        stl10_test_dataset = STL10(root = './data', split = "test", transform = transform, download = True)
        
        train_dataloader = DataLoader(stl10_train_dataset, batch_size = args['batch_size'], shuffle = True, drop_last = True)
        test_dataloader = DataLoader(stl10_test_dataset, batch_size = args['batch_size'], shuffle = False, drop_last = True)
        
    elif args['dataset_name'] == "imagenet100":
        
        print("with ImageNet 100")
        # Create dataset instances
        data_dir = "./data/imagenet100"
        label_file = os.path.join(data_dir, 'Labels.json')
        # Create the dataset
        dataset = ImageNetSubsetDataset(
            root_dir=data_dir,
            transform=original_transform,
            augment_transform=augment_transform
        )
        
        combined_loader = DataLoader(
            dataset,
            batch_size=args['batch_size'],
            shuffle=True,  # Shuffle data
            # num_workers=num_workers,  # Parallel loading
            drop_last=True  # Drop last batch if it's smaller than batch_size
        )
        
        train_size = len(dataset.train_image_paths)
        val_size = len(dataset.val_image_paths)

        # Create separate DataLoaders for train and val
        train_dataloader = DataLoader(
            torch.utils.data.Subset(dataset, list(range(train_size))),
            batch_size=args['batch_size'],
            shuffle=True,
            # num_workers=num_workers,
            drop_last=True
        )

        test_dataloader = DataLoader(
            torch.utils.data.Subset(dataset, list(range(train_size, train_size + val_size))),
            batch_size=args['batch_size'],
            shuffle=False,  # No need to shuffle validation data
            # num_workers=num_workers,
            drop_last=False
        )

        # imagenet100_train_dataset = ImageNetSubsetDataset(data_dir, 'train', transform=transform)
        # imagenet100_val_dataset = ImageNetSubsetDataset(data_dir, 'val', transform=transform)

        # # Create data loaders
        # train_dataloader = DataLoader(imagenet100_train_dataset, batch_size=args['batch_size'], shuffle=True, drop_last= True)
        # test_dataloader = DataLoader(imagenet100_val_dataset, batch_size=args['batch_size'], shuffle=False, drop_last = True)
        
    else:
        print("check proper db name it must be from mnist, cifar10, stl10, fashionmnist")
        return None, None
        
    return train_dataloader, test_dataloader

# print("With Food101")
# train_dataset = Food101(root = "/workspace/data/food101", split = "train", transform = transform, download = True)
# test_dataset = Food101(root = "/workspace/data/food101", split = "test", transform = transform, download = True)

# num_samples_train = int(0.20 * len(train_dataset))
# indices_train = torch.randperm(len(train_dataset))[:num_samples_train]

# num_samples_test = int(0.20 * len(test_dataset))
# indices_test = torch.randperm(len(test_dataset))[:num_samples_test]

# train_subset_dataset = Subset(train_dataset, indices_train)
# test_subset_dataset = Subset(test_dataset, indices_test)

# train_dataset = FashionMNIST(root = "./data", train= True, transform = transform, download = True)
# test_dataset = FashionMNIST(root = "./data", train= False, transform = transform, download = True)

# train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True, drop_last = True)
# test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False, drop_last = True)

# train_dataloader = DataLoader(stl10_train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)
# test_dataloader = DataLoader(stl10_test_dataset, batch_size = BATCH_SIZE, shuffle = False, drop_last = True)
