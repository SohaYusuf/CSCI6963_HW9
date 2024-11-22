# Soha Yusuf (RIN: 662011092)
# Shiuli Subhra Ghosh (RIN: )
# Jainik Meheta (RIN: )

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


if __name__ == '__main__':

    # image parameters
    resize_factor = 1
    new_h = int(IMAGE_HEIGHT / resize_factor)
    new_w = int(IMAGE_WIDTH / resize_factor)

    # feel free to change the hardcoded numbers -- they are from the CIFAR10 code
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    resize = torchvision.transforms.Resize(size = (new_h, new_w))
    convert = torchvision.transforms.ConvertImageDtype(torch.float)
    
    test_transforms = torchvision.transforms.Compose([resize, convert, normalize])

    data_dir = 'data/'
    val_labels_dir = os.path.join(data_dir, 'val_labels.csv')
    val_dataset = BuildingDataset(train_labels_dir, data_dir, transform=test_transforms)

    plot_data_images(val_dataset, name='buildings_images.png')

    # set training hyperparameters
    test_batch_size = 100
    n_epochs = 20
    learning_rate = 1e-3
    seed = 100
    input_dim = (3, new_h, new_w)
    out_dim = 11
    momentum = 0.9

    print(f'input_dim: {input_dim}, out_dim: {out_dim}')

    # put data into loaders
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)

    network = CNN(in_dim=input_dim, out_dim=out_dim, n_layers=5)
    network = network.to(device)

    # Compute the total number of parameters
    total_params = count_parameters(network)
    print(f"Total number of parameters: {total_params}")

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
   
    test_accuracy_list = []
    test_loss_list = []

    # training loop
    for epoch in range(1, n_epochs + 1):
        test_acc, test_loss = test(network, test_loader, device)

        test_accuracy_list.append(test_acc)
        test_loss_list.append(test_loss)

    torch.save(network.state_dict(), PATH)

    # Save the accuracies to separate text files
    np.savetxt('RPI_test_accuracies.txt', test_accuracy_list, header='Test Accuracy', delimiter=',', fmt='%f')