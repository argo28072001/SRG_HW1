import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
import os

class BinarySpeechCommandsDataset(Dataset):
    def __init__(self, root_dir, subset='training'):
        self.dataset = SPEECHCOMMANDS(root=root_dir, download=True)
        self.subset = subset
        
        # Filter for only "yes" and "no" samples
        self.data = []
        for i in range(len(self.dataset)):
            waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[i]
            if label in ['yes', 'no']:
                # Check split based on validation_list.txt and testing_list.txt
                file_path = f'{speaker_id}/{utterance_number}_{label}.wav'
                if self._check_split(file_path, subset):
                    self.data.append(i)
    
    def _check_split(self, file_path, subset):
        validation_list = os.path.join(self.dataset._path, 'validation_list.txt')
        testing_list = os.path.join(self.dataset._path, 'testing_list.txt')
        
        with open(validation_list, 'r') as f:
            val_files = set(f.read().splitlines())
        with open(testing_list, 'r') as f:
            test_files = set(f.read().splitlines())
            
        if subset == 'training':
            return file_path not in val_files and file_path not in test_files
        elif subset == 'validation':
            return file_path in val_files
        else:  # testing
            return file_path in test_files
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dataset_idx = self.data[idx]
        waveform, sample_rate, label, _, _ = self.dataset[dataset_idx]
        # Convert label to binary (0 for "no", 1 for "yes")
        label = 1 if label == "yes" else 0
        return waveform, label 