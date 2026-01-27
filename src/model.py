import torch
import torch.nn as nn


class DynamicCNN(nn.Module):
    def __init__(self, patch_size, kernel_size, start_channels, num_layers, regressor_hidden_size, dropout):
        """
        CNN model with dynamically determined number of layers.
        Used for biofilm surface area prediction.

        Args:
            patch_size (int): The height/width of the square patch (64, 80, 128, 160, 224, 240).
            kernel_size (int): The size of the convolutional kernel (3 or 5).
            start_channels (int): The number of channels in the first layer (16, 32, 64).
            num_layers (int): The number of convolutional layers to stack (3, 4, 5).
            regressor_hidden_size (int): The number of neurons in the hidden layer of the regressor (128 or 256).
            dropout (float): The dropout rate (0.2, 0.4, 0.5).
        
        Returns:
            nn.Module: DynamicCNN model.
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_channels = 1 # Input is grayscale
        
        # Dynamically build the 'num_layers'
        for i in range(num_layers):
            # Ensure 'same' padding (3x3 = 1 padding, 5x5 = 2 padding)
            padding = (kernel_size - 1) // 2
            
            # Decide output channels (double every layer, e.g., 32, 64, 128)
            out_channels = start_channels * (2 ** i)
            
            block = nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels), # Stabilizer
                nn.ReLU(),
                nn.MaxPool2d(2, 2) # Halves dimension
            )
            
            self.layers.append(block)
            current_channels = out_channels

        # Calculate Flatten Size automatically
        # We simulate a pass to see how big the feature map is at the end
        with torch.no_grad():
            dummy = torch.zeros(1, 1, patch_size, patch_size)
            for layer in self.layers:
                dummy = layer(dummy)
            # e.g., if output is (1, 128, 4, 4), flat_size is 2048
            self.flat_size = dummy.view(1, -1).shape[1]

        # 3. Regression Head (Outputs 1 value)
        self.regressor = nn.Sequential(
            nn.Linear(self.flat_size, regressor_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout), # Helps prevent overfitting
            nn.Linear(regressor_hidden_size, 1) # Output size 1 for Surface Area
        )

    def forward(self, x):
        # Pass through dynamic list of layers
        for layer in self.layers:
            x = layer(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Predict
        x = self.regressor(x)
        return x



class FixedCNN(nn.Module):
    """
    Fixed CNN model for biofilm surface area prediction.
    Used for comparison with DynamicCNN.
    
    Args:
        None
        
    Returns:
        nn.Module: SimpleCNN model.
    """
    def __init__(self):
        super().__init__()
        
        # --- LAYER 1 ---
        # Input: (1, 128, 128)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        # Output of Layer 1: (32, 64, 64) 

        # --- LAYER 2 ---
        # Input: (32, 64, 64)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output of Layer 2: (64, 32, 32)

        # --- LAYER 3 ---
        # Input: (64, 32, 32)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output of Layer 3: (128, 16, 16)

        # --- REGRESSION HEAD ---
        # 128 channels * 16 height * 16 width = 32,768
        self.regressor = nn.Sequential(
            nn.Linear(32768, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1) # Output 1 single value
        )

    def forward(self, x):
        x = self.block1(x) # Becomes 64x64
        x = self.block2(x) # Becomes 32x32
        x = self.block3(x) # Becomes 16x16
        
        # Flatten: (Batch, 128, 16, 16) -> (Batch, 32768)
        x = x.view(x.size(0), -1) 
        
        x = self.regressor(x)
        return x
