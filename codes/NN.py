import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score


# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
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

class ResNet(nn.Module):
    def __init__(self, n, r):
        super(ResNet, self).__init__()
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
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
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
    
# Hyperparameters
batch_size = 32
num_epochs = 50
learning_rate = 0.001
n = 2
r = 25

# Transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Create a dataset instance using ImageFolder
train_dir = 'Birds_25/train'
test_dir = 'Birds_25/test'
val_dir = 'Birds_25/val'
trainset = datasets.ImageFolder(train_dir, transform=transform)
testset = datasets.ImageFolder(test_dir, transform=transform)
valset = datasets.ImageFolder(val_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(n=2, r=25).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Define functions for computing metrics
def compute_accuracy_and_f1(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return accuracy, micro_f1, macro_f1

def plot_metrics(train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies,
                 train_f1_micro, val_f1_micro, test_f1_micro, train_f1_macro, val_f1_macro, test_f1_macro):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_f1_micro, label='Train F1 Micro')
    plt.plot(val_f1_micro, label='Val F1 Micro')
    plt.plot(test_f1_micro, label='Test F1 Micro')
    plt.plot(train_f1_macro, label='Train F1 Macro')
    plt.plot(val_f1_macro, label='Val F1 Macro')
    plt.plot(test_f1_macro, label='Test F1 Macro')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Training and validation loop with metrics tracking
train_losses = []
val_losses = []
test_losses = []
train_accuracies = []
val_accuracies = []
test_accuracies = []
train_f1_micro_array = []
val_f1_micro_array = []
test_f1_micro_array = []
train_f1_macro_array = []
val_f1_macro_array = []
test_f1_macro_array = []

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        print(i)
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i%100 == 99:
            print(correct)
            print(total)
            print(f'Epoch {epoch + 1}, Train Loss: {running_loss / 100}, Train Acc: {correct / total}')
            print(running_loss)
    train_losses.append(running_loss / len(trainloader))
    train_accuracies.append(correct / total)
    train_accuracy, train_f1_micro, train_f1_macro = compute_accuracy_and_f1(model, trainloader, device)
    val_accuracy, val_f1_micro, val_f1_macro = compute_accuracy_and_f1(model, valloader, device)
    test_accuracy, test_f1_micro, test_f1_macro = compute_accuracy_and_f1(model, testloader, device)
    print("OKOKOK")
    val_accuracies.append(val_accuracy)
    test_accuracies.append(test_accuracy)
    train_f1_macro_array.append(train_f1_macro)
    train_f1_micro_array.append(train_f1_micro)
    val_f1_micro_array.append(val_f1_micro)
    test_f1_micro_array.append(test_f1_micro)
    val_f1_macro_array.append(val_f1_macro)
    test_f1_macro_array.append(test_f1_macro)
    print(train_f1_micro_array)
    print(val_f1_micro_array)
    print(test_f1_micro_array)
    print(train_f1_macro_array)
    print(val_f1_macro_array)
    print(test_f1_macro_array)
    # scheduler.step()
    # print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]}, Train Acc: {train_accuracies[-1]}, Val Acc: {val_accuracy}, Test Acc: {test_accuracy}')

        

print('Training finished.')
torch.save(model.state_dict(), 'resnet_model.pkl')
plot_metrics(train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies,train_f1_micro_array, val_f1_micro_array, test_f1_micro_array, train_f1_macro_array, val_f1_macro_array, test_f1_macro_array)