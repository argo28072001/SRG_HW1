import torch
import torch.nn as nn
import pytorch_lightning as pl
from thop import profile
from melbanks import LogMelFilterBanks

class SpeechClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # Mel spectrogram layer
        self.mel_spec = LogMelFilterBanks(
            n_fft=400,
            samplerate=16000,
            hop_length=160,
            n_mels=80
        )
        
        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Metrics
        self.train_acc = pl.metrics.Accuracy(task='binary')
        self.val_acc = pl.metrics.Accuracy(task='binary')
        self.test_acc = pl.metrics.Accuracy(task='binary')
        
    def forward(self, x):
        # Convert to mel spectrograms
        x = self.mel_spec(x)
        # Add channel dimension
        x = x.unsqueeze(1)
        # Pass through CNN
        x = self.conv_layers(x)
        # Pass through FC layers
        x = self.fc_layers(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float().unsqueeze(1))
        self.train_acc(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float().unsqueeze(1))
        self.val_acc(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.test_acc(y_hat, y)
        self.log('test_acc', self.test_acc)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def calculate_flops(self, input_size=(1, 1, 16000)):
        input_tensor = torch.randn(input_size)
        flops, _ = profile(self, inputs=(input_tensor,))
        return flops 