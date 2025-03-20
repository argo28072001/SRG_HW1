import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import time

from dataset import BinarySpeechCommandsDataset
from model import SpeechClassifier

# Data setup
train_dataset = BinarySpeechCommandsDataset(root_dir='./data', subset='training')
val_dataset = BinarySpeechCommandsDataset(root_dir='./data', subset='validation')
test_dataset = BinarySpeechCommandsDataset(root_dir='./data', subset='testing')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

# Model setup
model = SpeechClassifier()

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath='checkpoints',
    filename='speech-classifier-{epoch:02d}-{val_acc:.2f}',
    save_top_k=3,
    mode='max',
)

# Logger
logger = TensorBoardLogger('logs', name='speech_classifier')

# Custom callback for epoch time tracking
class TimeCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        trainer.logger.experiment.add_scalar('epoch_time', epoch_time, trainer.current_epoch)

# Training
trainer = pl.Trainer(
    max_epochs=30,
    callbacks=[checkpoint_callback, TimeCallback()],
    logger=logger,
    accelerator='auto',
    devices=1
)

# Print model statistics
print(f"Number of parameters: {model.count_parameters():,}")
print(f"Estimated FLOPs: {model.calculate_flops():,}")

# Train the model
trainer.fit(model, train_loader, val_loader)

# Test the model
trainer.test(model, test_loader) 