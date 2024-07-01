import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from sklearn.metrics import accuracy_score, f1_score
import argparse
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR


# Inbuilt Batch Normalization ResNet
class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class BasicResNet(nn.Module):
    def __init__(self, n, r):
        super(BasicResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, n)
        self.layer2 = self.make_layer(32, n, stride=2)
        self.layer3 = self.make_layer(64, n, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, r)

    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.layer1(res)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.avg_pool(res)
        res = res.view(res.size(0), -1)
        res = self.fc(res)
        return res
    
# Custom Batch Normalization Resnet
class CustomBatchNormalization(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(CustomBatchNormalization, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


class CustomBatchResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(CustomBatchResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = CustomBatchNormalization(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = CustomBatchNormalization(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                CustomBatchNormalization(out_channels)
            )
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class CustomBatchResNet(nn.Module):
    def __init__(self, n, r):
        super(CustomBatchResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = CustomBatchNormalization(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, n)
        self.layer2 = self.make_layer(32, n, stride=2)
        self.layer3 = self.make_layer(64, n, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, r)

    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(CustomBatchResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(CustomBatchResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.layer1(res)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.avg_pool(res)
        res = res.view(res.size(0), -1)
        res = self.fc(res)
        return res
    

# Instance Normalization Implemented ResNet
class InstanceNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(InstanceNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # Initialize parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Calculate instance statistics
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)

        # Apply instance normalization
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        out =  x_normalized * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out

# Define the Residual Block
class InstanceResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(InstanceResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.in1 = InstanceNormalization(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = InstanceNormalization(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                InstanceNormalization(out_channels)
            )
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class InstanceResNet(nn.Module):
    def __init__(self, n, r):
        super(InstanceResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = InstanceNormalization(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, n)
        self.layer2 = self.make_layer(32, n, stride=2)
        self.layer3 = self.make_layer(64, n, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, r)

    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(InstanceResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(InstanceResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        res = self.conv1(x)
        res = self.in1(res)
        res = self.relu(res)
        res = self.layer1(res)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.avg_pool(res)
        res = res.view(res.size(0), -1)
        res = self.fc(res)
        return res
    

# Batch Instance Normalization Implemented ResNet
class BatchInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # Batch normalization parameters
        self.batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        # Instance normalization parameters
        self.instance_norm = nn.InstanceNorm2d(num_features, eps=eps)

    def forward(self, x):
        # Batch normalization
        x_batch_norm = self.batch_norm(x)
        # Instance normalization
        x_instance_norm = self.instance_norm(x)
        # Combine batch and instance normalization results
        return x_batch_norm + x_instance_norm
    
class BatchInstanceNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchInstanceNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Batch normalization parameters
        self.batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

        # Instance normalization parameters
        self.instance_norm = nn.InstanceNorm2d(num_features, eps=eps)

        # Scaling parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Apply batch normalization
        batch_normalized = self.batch_norm(x)

        # Apply instance normalization
        instance_normalized = self.instance_norm(x)

        # Compute Batch-Instance Normalization
        out = self.gamma.view(1,-1,1,1) * batch_normalized + self.beta.view(1,-1,1,1) + instance_normalized - self.gamma.view(1,-1,1,1) * instance_normalized.mean(dim=(2, 3), keepdim=True)

        return out

# Define the Residual Block
class BatchInstanceResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(BatchInstanceResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchInstanceNormalization(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchInstanceNormalization(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BatchInstanceNormalization(out_channels)
            )
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class BatchInstanceResNet(nn.Module):
    def __init__(self, n, r):
        super(BatchInstanceResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchInstanceNormalization(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, n)
        self.layer2 = self.make_layer(32, n, stride=2)
        self.layer3 = self.make_layer(64, n, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, r)

    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(BatchInstanceResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BatchInstanceResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.layer1(res)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.avg_pool(res)
        res = res.view(res.size(0), -1)
        res = self.fc(res)
        return res
    


# Layer Normalization Implemented ResNet
class LayerNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LayerNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x):
        # Compute mean and variance across the feature dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        # Normalize input tensor
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        # Scale and shift normalized input using learnable parameters
        return x_normalized * self.gamma.view(1,-1,1,1) + self.beta.view(1,-1,1,1)

# Define the Residual Block
class LayerResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(LayerResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ln1 = LayerNormalization(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln2 = LayerNormalization(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                LayerNormalization(out_channels)
            )
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ln2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class LayerResNet(nn.Module):
    def __init__(self, n, r):
        super(LayerResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln1 = LayerNormalization(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, n)
        self.layer2 = self.make_layer(32, n, stride=2)
        self.layer3 = self.make_layer(64, n, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, r)

    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(LayerResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(LayerResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        res = self.conv1(x)
        res = self.ln1(res)
        res = self.relu(res)
        res = self.layer1(res)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.avg_pool(res)
        res = res.view(res.size(0), -1)
        res = self.fc(res)
        return res
    




# Group Normalization Implemented ResNet
class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.num_groups, -1)
        mean = torch.mean(x,dim=2, keepdim=True)
        var = torch.var(x,dim=2, unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        x_normalized = x_normalized.view(N, C, H, W)
        return x_normalized * self.gamma + self.beta
    
# Define the Residual Block
class GroupResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(GroupResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = GroupNorm(4,out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = GroupNorm(4,out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                GroupNorm(4, out_channels)
            )
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class GroupResNet(nn.Module):
    def __init__(self, n, r):
        super(GroupResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = GroupNorm(4, self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, n)
        self.layer2 = self.make_layer(32, n, stride=2)
        self.layer3 = self.make_layer(64, n, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, r)

    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(GroupResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(GroupResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        res = self.conv1(x)
        res = self.gn1(res)
        res = self.relu(res)
        res = self.layer1(res)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.avg_pool(res)
        res = res.view(res.size(0), -1)
        res = self.fc(res)
        return res
    
    

# ---------------------------------------------------No Norm -------------------------------------------------------    
    
class NoNormResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(NoNormResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class NoNormResNet(nn.Module):
    def __init__(self, n, r):
        super(NoNormResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, n)
        self.layer2 = self.make_layer(32, n, stride=2)
        self.layer3 = self.make_layer(64, n, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, r)

    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(NoNormResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(NoNormResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.layer1(res)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.avg_pool(res)
        res = res.view(res.size(0), -1)
        res = self.fc(res)
        return res



def evaluate(model, data_loader, device,file):
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            file.write(str(preds[0].item())+'\n')

if __name__ == "__main__":
    device = torch.device("cpu")
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--model_file", type=str, help="Path to the model file")
    parser.add_argument("--normalization", type=str, help="Type of normalization (e.g., 'bn')")
    parser.add_argument("--n", type=int, help="Number 'n'")
    parser.add_argument("--test_data_file", type=str, help="Path to the test data file")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    args = parser.parse_args()
    model_path = args.model_file
    normalization = args.normalization
    n = args.n
    test_case_path = args.test_data_file
    output_file = args.output_file
    model = None
    if normalization=="bn":
        model = CustomBatchResNet(2,25)
    elif normalization=="in":
        model = InstanceResNet(2,25)
    elif normalization=="bin":
        model = BatchInstanceResNet(2,25)
    elif normalization=="ln":
        model = LayerResNet(2,25)
    elif normalization=="gn":
        model = GroupResNet(2,25)
    elif normalization=="nn":
        model = NoNormResNet(2,25)
    elif normalization=="inbuild":
        model = BasicResNet(2,25)
    else:
        raise Exception("Invalid Normalization")
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    try:
        test_dataset = datasets.ImageFolder(root=test_case_path, transform=transform)
    except:
        raise("Invalid test file path")
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    with open('output.txt', 'w') as f:
        evaluate(model, test_loader, device,f)
        f.close()