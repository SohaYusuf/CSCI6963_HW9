# Soha Yusuf (RIN: 662011092)
# Shiuli Subhra Ghosh (RIN: )
# Jainik Meheta (RIN: )

# python cnn_classification_cifar10.py

import torchvision
from model import *
import numpy as np
import torch
from torch.utils.data import Subset
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_image(image):
    image = np.reshape(image, (3, 32, 32))
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()


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
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())
            total_samples += target.size(0)  # Count number of samples

            total = total + 1

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss / total_samples, correct, len(loader.dataset),
        (100. * correct / len(loader.dataset))), flush=True)

    return 100.0 * correct / len(loader.dataset), test_loss / total_samples


def train(net, loader, optimizer, epoch, device, log_interval=100):
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

    print('\tAccuracy: {:.2f}%'.format(100.0 * correct / len(loader.dataset)), flush=True)
    return 100.0 * correct / len(loader.dataset), total_loss / total_samples


if __name__ == '__main__':

    save_path = "results_cifar10/"
    os.makedirs(save_path, exist_ok=True)

    # set hyper-parameters
    train_batch_size = 100
    test_batch_size = 100
    n_epochs = 30
    learning_rate = 1e-2
    seed = 100
    input_dim = (3,32,32)
    out_dim = 10
    momentum = 0.9

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    train_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])

    train_dataset = torchvision.datasets.CIFAR10('./datasets/', train=True, download=True, transform=train_transforms)
    test_dataset = torchvision.datasets.CIFAR10('./datasets/', train=False, download=True, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    plot_data_images(train_dataset, name=f'{save_path}/CIFAR10.png')

    # create neural network object
    network = CNN_small(in_dim=input_dim, out_dim=out_dim)
    network = network.to(device)

    # Compute the total number of parameters
    total_params = count_parameters(network)
    print(f"Total number of parameters: {total_params}")

    # set up optimizer
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    train_accuracy_list = []
    test_accuracy_list = []

    train_loss_list = []
    test_loss_list = []

    # training loop
    for epoch in range(1, n_epochs + 1):
        train_acc, train_loss = train(network, train_loader, optimizer, epoch, device)
        test_acc, test_loss = test(network, test_loader, device)

        train_accuracy_list.append(train_acc)
        test_accuracy_list.append(test_acc)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    # Save the accuracies to separate text files
    np.savetxt(f'{save_path}/CIFAR10_train_accuracies.txt', train_accuracy_list, header='Train Accuracy', delimiter=',', fmt='%f')
    np.savetxt(f'{save_path}/CIFAR10_test_accuracies.txt', test_accuracy_list, header='Test Accuracy', delimiter=',', fmt='%f')
    
    plot_results(train_accuracy_list, test_accuracy_list, name=f'{save_path}/CIFAR10_accuracy_plot.png', plot_accuracy=True)
    plot_results(train_loss_list, test_loss_list, name=f'{save_path}/CIFAR10_loss_plot.png', plot_loss=True)