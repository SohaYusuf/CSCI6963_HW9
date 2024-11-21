import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision
from model import CNN, count_parameters

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

    with torch.no_grad():
        for data, target in loader:

            data, target = data.to(device), target.to(device)
            
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())
            
            total = total + 1

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(loader.dataset),
        (100. * correct / len(loader.dataset))), flush=True)
    
    return 100.0 * correct / len(loader.dataset)

def train(net, loader, optimizer, epoch, device, log_interval=1):
    # prepare model for training (only important for dropout, batch norm, etc.)
    net.train()

    correct = 0
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

        if batch_idx % log_interval == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader), loss.item()), flush=True)

    accuracy = 100.0 * correct / len(loader.dataset)
    print('\tAccuracy: {:.2f}%'.format(accuracy), flush=True)  
    return accuracy


if __name__ == '__main__':

    # image parameters
    resize_factor = 1
    new_h = int(IMAGE_HEIGHT / resize_factor)
    new_w = int(IMAGE_WIDTH / resize_factor)

    # feel free to change the hardcoded numbers -- they are from the CIFAR10 code
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    resize = torchvision.transforms.Resize(size = (new_h, new_w))
    convert = torchvision.transforms.ConvertImageDtype(torch.float)
    
    train_transforms = torchvision.transforms.Compose([resize, convert, normalize])
    test_transforms = torchvision.transforms.Compose([resize, convert, normalize])

    data_dir = 'data/'
    train_labels_dir = os.path.join(data_dir, 'train_labels.csv')
    val_labels_dir = os.path.join(data_dir, 'val_labels.csv')

    train_dataset = BuildingDataset(train_labels_dir, data_dir, transform=train_transforms)
    val_dataset = BuildingDataset(train_labels_dir, data_dir, transform=test_transforms)



    # Plotting (leaving this here in case you'd like to take a look)
    # image = train_dataset[10][0]
    # image = image.permute(1,2,0)

    # plt.figure()
    # plt.imshow(image)
    # #plt.imshow(torch.reshape(image, (new_h, new_w)), cmap='gray_r')
    # plt.show()

    # set training hyperparameters
    train_batch_size = 100
    test_batch_size = 100
    n_epochs = 30
    learning_rate = 1e-3
    seed = 100
    input_dim = (3, new_h, new_w)
    out_dim = 11
    momentum = 0.9

    print(f'input_dim: {input_dim}, out_dim: {out_dim}')

    # put data into loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
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

    train_accuracy_list = []
    test_accuracy_list = []

    # training loop
    for epoch in range(1, n_epochs + 1):
        train_acc = train(network, train_loader, optimizer, epoch, device)
        test_acc = test(network, test_loader, device)

        train_accuracy_list.append(train_acc)
        test_accuracy_list.append(test_acc)

    torch.save(network.state_dict(), PATH)
    
    plot_accuracies(train_accuracy_list, test_accuracy_list)