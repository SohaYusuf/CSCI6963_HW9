# Soha Yusuf (RIN: 662011092)
# Shiuli Subhra Ghosh (RIN: )
# Jainik Meheta (RIN: 662096080)

# python train_buildings.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision
from model import *

from dataset import BuildingDataset

import numpy as np

import os, sys

IMAGE_HEIGHT = 189
IMAGE_WIDTH = 252

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_train_val(org_train_set, valid_ratio=0.1):

    num_train = len(org_train_set)

    split = int(np.floor(valid_ratio * num_train))        

    indices = list(range(num_train))

    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    new_train_set = Subset(org_train_set, train_idx)
    val_set = Subset(org_train_set, val_idx)

    assert num_train - split == len(new_train_set)
    assert split == len(val_set)

    return new_train_set, val_set

def test(net, loader, device):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in loader:

            data, target = data.to(device), target.to(device)
            
            # output = net(data)
            output = F.log_softmax(net(data), dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())
            total_samples += target.size(0)  # Count number of samples
            
            total = total + 1

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss / total_samples, correct, len(loader.dataset),
        (100. * correct / len(loader.dataset))), flush=True)
    
    return 100.0 * correct / len(loader.dataset), test_loss / total_samples

def train(net, loader, optimizer, epoch, device, log_interval=1):
    # prepare model for training (only important for dropout, batch norm, etc.)
    net.train()

    correct = 0
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(loader):

        data, target = data.to(device), target.to(device)
        
        # clear up gradients for backprop
        optimizer.zero_grad()
        output = F.log_softmax(net(data), dim=1)

        # use NLL loss
        loss = F.nll_loss(output, target)

        # compute gradients and make updates
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += (pred.eq(target.data.view_as(pred)).sum().item())

        total_loss += loss.item() * data.size(0)  # accumulate loss
        total_samples += data.size(0)              # accumulate number of samples

        if batch_idx % log_interval == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader), loss.item()), flush=True)

    mean_loss = total_loss / total_samples  # calculate mean loss
    accuracy = 100.0 * correct / len(loader.dataset)
    print('\tAccuracy: {:.2f}%'.format(accuracy), flush=True)  
    return accuracy, mean_loss


if __name__ == '__main__':

    save_path = "results_buildings/"
    os.makedirs(save_path, exist_ok=True)

    # image parameters
    resize_factor = 1
    new_h = int(IMAGE_HEIGHT / resize_factor)
    new_w = int(IMAGE_WIDTH / resize_factor)

    # feel free to change the hardcoded numbers -- they are from the CIFAR10 code
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    resize = torchvision.transforms.Resize(size = (new_h, new_w))
    convert = torchvision.transforms.ConvertImageDtype(torch.float)
    
    # Data augmentation for training
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),           # Randomly flip images horizontally
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(degrees=30),
        torchvision.transforms.RandomCrop(size=(new_h, new_w), padding=4),  # Random crop with padding
        resize,
        convert,
        normalize
    ])
    test_transforms = torchvision.transforms.Compose([resize, convert, normalize])

    data_dir = 'data/'
    train_labels_dir = os.path.join(data_dir, 'train_labels.csv')
    val_labels_dir = os.path.join(data_dir, 'val_labels.csv')

    train_dataset = BuildingDataset(train_labels_dir, data_dir, transform=train_transforms)
    test_dataset = BuildingDataset(val_labels_dir, data_dir, transform=test_transforms)

    plot_data_images(train_dataset, name=f'{save_path}/buildings_train_images.png')
    plot_data_images(test_dataset, name=f'{save_path}/buildings_test_images.png')

    # set training hyperparameters
    train_batch_size = 64
    test_batch_size = 64
    n_epochs = 40
    learning_rate = 1e-3
    seed = 100
    input_dim = (3, new_h, new_w)
    out_dim = 11
    momentum = 0.9

    print(f'input_dim: {input_dim}, out_dim: {out_dim}')

    # put data into loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    network = CNN(in_dim=input_dim, out_dim=out_dim, n_layers=5)
    # Compute the total number of parameters
    total_params = count_parameters(network)
    print(f"Total number of parameters: {total_params}")
    network = network.to(device)

    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    PATH = model_path + 'cnn_buildings.pth'
    
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        network.load_state_dict(torch.load(PATH))

    # sanity check -- output should be close to 1/11
    print('Initial accuracy', flush=True)
    test(network, test_loader, device)

    train_accuracy_list = []
    val_accuracy_list = []

    train_loss_list = []
    val_loss_list = []

    # training loop
    for epoch in range(1, n_epochs + 1):
        train_acc, train_loss = train(network, train_loader, optimizer, epoch, device)
        val_acc, val_loss = test(network, test_loader, device)

        train_accuracy_list.append(train_acc)
        val_accuracy_list.append(val_acc)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)


    test_acc, test_loss = test(network, test_loader, device)
    print(f'Final test accuracy: {test_acc}, Final test loss: {test_loss}')
    torch.save(network.state_dict(), PATH)

    # Save the accuracies to separate text files
    np.savetxt(f'{save_path}/RPI_train_accuracies.txt', train_accuracy_list, header='Train Accuracy', delimiter=',', fmt='%f')
    np.savetxt(f'{save_path}/RPI_val_accuracies.txt', val_accuracy_list, header='Val Accuracy', delimiter=',', fmt='%f')

    plot_results(train_accuracy_list, val_accuracy_list, name=f'{save_path}/RPI_accuracy_plot.png', plot_accuracy=True)
    plot_results(train_loss_list, val_loss_list, name=f'{save_path}/RPI_loss_plot.png', plot_loss=True)