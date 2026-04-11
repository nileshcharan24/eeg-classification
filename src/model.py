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

class EEGNetClassifier(nn.Module):
    def __init__(self, input_size=256, num_classes=3, num_channels=32):
        """
        EEGNet architecture for 2D inputs: (batch, 1, channels, time_samples/features)
        """
        super(EEGNetClassifier, self).__init__()
        
        self.num_channels = num_channels
        # If input_size is the flat size (e.g., 256), features_per_channel is input_size // 32 = 8
        # If input_size is already the features_per_channel (e.g., 8), then we use it directly
        self.features_per_channel = input_size // num_channels if input_size >= num_channels else input_size
        
        F1 = 8
        D = 2
        F2 = 16
        
        # 1. Temporal/Feature Conv
        kernel_time = min(4, self.features_per_channel) if self.features_per_channel >= 1 else 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_time), padding=(0, kernel_time//2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # 2. Depthwise Spatial Conv
        self.depthwise = nn.Conv2d(F1, F1 * D, (self.num_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        
        pool_time1 = min(2, self.features_per_channel)
        self.avgpool1 = nn.AvgPool2d((1, pool_time1))
        self.dropout1 = nn.Dropout(0.25)
        
        # 3. Separable Conv (Depthwise + Pointwise)
        kernel_sep = min(8, self.features_per_channel)
        self.separable_depthwise = nn.Conv2d(F1 * D, F1 * D, (1, kernel_sep),
                                             padding=(0, kernel_sep//2), groups=F1 * D, bias=False)
        self.separable_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        
        pool_time2 = min(2, max(1, self.features_per_channel // pool_time1))
        self.avgpool2 = nn.AvgPool2d((1, pool_time2))
        self.dropout2 = nn.Dropout(0.25)
        
        # Dynamically calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.num_channels, self.features_per_channel)
            x = self.conv1(dummy_input)
            x = self.batchnorm1(x)
            x = self.depthwise(x)
            x = self.batchnorm2(x)
            x = self.elu(x)
            x = self.avgpool1(x)
            x = self.dropout1(x)
            
            x = self.separable_depthwise(x)
            x = self.separable_pointwise(x)
            x = self.batchnorm3(x)
            x = self.elu(x)
            x = self.avgpool2(x)
            x = self.dropout2(x)
            
            flattened_size = x.view(1, -1).size(1)
            
        self.fc = nn.Linear(flattened_size, num_classes)
        
    def forward(self, x):
        # Ensure 4D input: (Batch, 1, Channels, Features)
        if x.dim() == 2:
            x = x.view(-1, 1, self.num_channels, self.features_per_channel)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.view(-1, 1, self.num_channels, self.features_per_channel)
            
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        
        return x

class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_size=128, num_classes=3):
        super(CNNLSTMClassifier, self).__init__()
        
        # Layer 1 (Temporal/Frequency Conv)
        # Assuming input is (Batch, 1, Features) where Features is treated as the sequence for 1D Conv
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.elu1 = nn.ELU()
        self.drop1 = nn.Dropout(0.3)
        
        # Layer 2 (Spatial Conv)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.elu2 = nn.ELU()
        self.drop2 = nn.Dropout(0.3)
        
        # Layer 3 (LSTM)
        # CNN output is (Batch, 128, Features). We permute to (Batch, Features, 128) for LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        
        # Layer 4 (Dense)
        self.fc1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        
        # Output Layer
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input Reshape: Ensure inputs are (batch_size, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.drop2(x)
        
        # Prepare for LSTM: (Batch, Channels, Length) -> (Batch, Length, Channels)
        x = x.permute(0, 2, 1)
        
        out, (hn, cn) = self.lstm(x)
        
        # Take the output of the final time step
        x = out[:, -1, :]
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop3(x)
        
        x = self.fc2(x)
        
        return x

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
        try:
            # Test allocation to catch hardware/capability mismatches (e.g. on Kaggle)
            _ = torch.zeros(1).cuda()
            return torch.device("cuda")
        except (RuntimeError, AssertionError) as e:
            print("\n" + "="*60)
            print("⚠️  WARNING: CUDA is available, but a hardware/capability mismatch occurred:")
            print(f"   {e}")
            print("Falling back to CPU safely. The model will train on the CPU.")
            print("="*60 + "\n")
            return torch.device("cpu")
    else:
        print("\n" + "="*60)
        print("⚠️  WARNING: CUDA is returning False. GPU is NOT available.")
        print("Your Python environment is missing the GPU-enabled version of PyTorch.")
        print("The model will train on the CPU, which will be significantly slower.")
        print("="*60 + "\n")
        return torch.device("cpu")
