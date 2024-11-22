import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

class FC(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_size):
        super().__init__()

        self.num_layers = num_hidden_layers * 2 + 3 # *2 accounts for ReLU layers, +3 is input layer, input relu layer, output layer

        self.in_dim = in_dim
        self.out_dim = out_dim        

        self.layer_size = layer_size

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_size))
        self.num_hidden_layers = num_hidden_layers

        for i in range(1,self.num_hidden_layers):
            self.layer_list.append(nn.Linear(self.layer_size, self.layer_size))
            

        self.layer_list.append(nn.Linear(self.layer_size, self.out_dim))
        
    def forward(self, x):

        x = x.view(-1, self.in_dim)

        for i in range(self.num_hidden_layers):
            x = F.relu(self.layer_list[i](x))

        return self.layer_list[self.num_hidden_layers](x)


class CNN(nn.Module):
    """
    A Convolutional Neural Network for image classification.

    Attributes:
        in_dim (int): Dimension of input images (height x width x channels).
        out_dim (int): Number of output classes for classification.
        n_layers (int): Number of convolutional layers in the network.

    Inputs:
        x (torch.Tensor): A batch of input images with shape (batch_size, channels, height, width).

    Outputs:
        torch.Tensor: Class scores for each input image with shape (batch_size, out_dim).
    """
    
    def __init__(self, in_dim, out_dim, n_layers):
        super(CNN, self).__init__()

        # Store dimensions and layer count
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        # Define the number of filters for each layer
        self.filters = [16, 32, 64, 64, 32]

        # Convolution parameters
        self.kernel_size = (3, 3)
        self.stride = 1
        self.padding = 1

        # Initialize convolutional layers
        self.convs = self._make_conv_layers()

        # Placeholder for the fully connected layer
        self.fc = None

    def _make_conv_layers(self):
        """Construct convolutional layers and return as a ModuleList."""
        layers = []
        for i in range(self.n_layers):
            in_channels = 3 if i == 0 else self.filters[min(i - 1, len(self.filters) - 1)]
            out_channels = self.filters[min(i, len(self.filters) - 1)]

            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, self.kernel_size, stride=self.stride, padding=self.padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
        return nn.ModuleList(layers)

    def forward(self, x):
        """Forward pass through the network."""
        for conv in self.convs:
            x = conv(x)  # Apply convolutional layers
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Initialize the fully connected layer if needed
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), self.out_dim).to(x.device)
        
        # Pass through the fully connected layer
        x = self.fc(x)
        return x 
 


class CNN_small(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim        

        self.layer1_filters = 16  
        self.layer2_filters = 32 
        self.layer3_filters = 32  

        self.kernel_size = (3, 3)
        self.stride = 1
        self.padding = 1

        # First Convolutional Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.layer1_filters, self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(self.layer1_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Max pooling to reduce spatial dimensions
        )

        # Second Convolutional Layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.layer1_filters, self.layer2_filters, self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(self.layer2_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Max pooling to further reduce spatial dimensions
        )

        # Third Convolutional Layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.layer2_filters, self.layer3_filters, self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(self.layer3_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Max pooling to reduce spatial dimensions
        )

        # Flatten the output from conv3 and calculate the input size for the fully connected layer
        self.fc_inputs = self.layer3_filters * 4 * 4  # Output size after 3x2 pooling layers, reducing height and width to 4x4

        # Single fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.fc_inputs, self.out_dim)  # Only one fully connected layer with output size equal to number of classes (out_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc(x)
        return x



# Function to calculate the total number of parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def plot_results(train_list, test_list, name, plot_accuracy=True, plot_loss=False):

    epochs = range(1, len(train_list) + 1)  # Create a range for epochs

    # Set up the plot
    fontsize = 25
    plt.figure(figsize=(8, 6))
    plt.gca().set_facecolor('#F5F5F5')

    if plot_accuracy:
        label1 = 'Train Accuracy'
        label2 = 'Test Accuracy'
        xlabel = 'Epochs'
        ylabel = 'Accuracy'

    if plot_loss:
        label1 = 'Train loss'
        label2 = 'Test loss'
        xlabel = 'Epochs'
        ylabel = 'Loss'

    # Plotting both training and test accuracy
    plt.plot(train_list, label=label1, color='blue', marker='o')
    plt.plot(test_list, label=label2, color='orange', marker='o')

    # Configure x-axis and y-axis
    plt.xscale('linear')  # Change to 'log' if you want logarithmic scaling
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    # Add a legend and grid
    plt.legend(fontsize=fontsize)
    plt.grid(color='#D3D3D3', linestyle='-', linewidth=0.5)  # Light gray fine grid
    plt.grid(True)

    # Configure tick labels
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Save the figure
    plt.savefig(name, bbox_inches="tight", dpi=300)
    plt.show()  # Display the plot


def plot_data_images(train_dataset, name):

    # Number of images to display
    num_images = 20

    # Get random indices from the dataset
    random_indices = random.sample(range(len(train_dataset)), num_images)

    # Create a figure to display the images
    plt.figure(figsize=(12,6))

    for i, idx in enumerate(random_indices):
        # Get the image and label
        image, label = train_dataset[idx]
        
        # Permute the image dimensions for plotting (C, H, W) to (H, W, C)
        image = image.permute(1, 2, 0)
        
        # Create a subplot for each image
        plt.subplot(4, 5, i + 1)  # 4 rows and 5 columns
        plt.imshow(image)
        plt.title(f'Label: {label}')  # Show label if needed
        plt.axis('off')  # Hide axes

    plt.tight_layout()
    plt.savefig(name, dpi=300)
    # plt.show()