import torch
import torchvision
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, Food101, STL10
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms 

transform = transforms.Compose([
    transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
])

mnist_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
])

BATCH_SIZE = 64

print("With CIFAR10")
cifar10_train_dataset = CIFAR10(root = "./data/cifar10", train = True, transform = transform, download = True)
cifar10_test_dataset = CIFAR10(root = "./data/cifar10", train = False, transform = transform, download = True)

# print("With MNIST")
# mnist_train_dataset = MNIST(root = './data', train = True, transform = mnist_transform, download = True)
# mnist_test_dataset = MNIST(root = './data', train = False, transform = mnist_transform, download = True)

# print("with Fashion")
# fashion_train_dataset = FashionMNIST(root = './data', train = True, transform = mnist_transform, download = True)
# fashion_test_dataset = FashionMNIST(root = './data', train = False, transform = mnist_transform, download = True)

# print("with STL10")
# stl10_train_dataset = STL10(root = './data', split = "train", transform = transform, download = True)
# stl10_test_dataset = STL10(root = './data', split = "test", transform = transform, download = True)

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

train_dataloader = DataLoader(cifar10_train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)
test_dataloader = DataLoader(cifar10_test_dataset, batch_size = BATCH_SIZE, shuffle = False, drop_last = True)
