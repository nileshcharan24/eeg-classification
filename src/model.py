import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class EEGClassifier(nn.Module):
    def __init__(self, input_size=128, num_classes=3):
        """
        Deep 1D-CNN with Residual connections for mental state classification.
        Takes flat PSD features (e.g., size 128) and reshapes to use Conv1D.
        """
        super(EEGClassifier, self).__init__()
        
        # Initial Convolution
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual Blocks
        self.layer1 = ResidualBlock1D(64, 64, stride=1, dropout=0.3)
        self.layer2 = ResidualBlock1D(64, 128, stride=2, dropout=0.4)
        self.layer3 = ResidualBlock1D(128, 256, stride=2, dropout=0.4)
        self.layer4 = ResidualBlock1D(256, 512, stride=2, dropout=0.5)
        
        # Global Average Pooling adapts to any input size
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape initially: (Batch, Features) e.g., (Batch, 128)
        if x.dim() == 2:
            x = x.unsqueeze(1) # Reshape to (Batch, 1, Features) for 1D-CNN
            
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        
        return x

def get_device():
    """Returns the appropriate device and explicitly warns if GPU is not available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("\n" + "="*60)
        print("⚠️  WARNING: CUDA is returning False. GPU is NOT available.")
        print("Your Python environment is missing the GPU-enabled version of PyTorch.")
        print("The model will train on the CPU, which will be significantly slower.")
        print("="*60 + "\n")
        return torch.device("cpu")
