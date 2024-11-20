import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim        

    def forward(self, x):
        pass


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