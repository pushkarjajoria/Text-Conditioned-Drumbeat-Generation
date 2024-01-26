import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import wandb
from unsupervised_pretraining.create_unsupervised_dataset import MidiDataset, get_filenames_and_tags
from unsupervised_pretraining.model import CLAMP


# Other necessary imports: torchvision, numpy, etc.

with open('unsupervised_pretraining/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define hyperparameters from the config
EPOCHS = config['Training']['epochs']
BATCH_SIZE = config['Training']['batch_size']
LEARNING_RATE = config['Training']['learning_rate']
WEIGHT_DECAY = config['Training']['weight_decay']
CHECKPOINT_INTERVAL = config['Training']['checkpoint_interval']


def save_checkpoint(model, run_name, epoch, wandb, save_type="checkpoint"):
    path = os.path.join(save_type, run_name)
    os.makedirs(path, exist_ok=True)
    filename = f'model_epoch_{epoch}.pth' if epoch \
        else 'model_final.pth'
    full_path = os.path.join(path, filename)
    # Save locally
    torch.save(model.state_dict(), full_path)
    # Save on wandb - make sure the file is in the current directory or subdirectory.
    wandb.save(full_path)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


if __name__ == "__main__":
    project_name = 'Unsupervised Pretraining' if torch.cuda.is_available() else '[Dev][Mac] Unsupervised Pretraining'
    run_name = datetime.now().strftime("%m%d_%H%M")
    wandb.init(project=project_name, name=run_name)
    # Initialize Dataloader
    file_name_and_tags = get_filenames_and_tags(dataset_dir="../datasets/Groove_Monkee_Mega_Pack_GM", filter_common_tags=True)
    train_dataset = MidiDataset(file_name_and_tags)  # Placeholder paths
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, optimizer, and learning rate scheduler
    model = CLAMP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.3)

    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    # Training loop
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for midi_data, text_data in tqdm(train_loader):
            text_embeddings, midi_embeddings = model(midi_data, text_data)
            loss = model.contrastive_loss(text_embeddings, midi_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")

        # Step the learning rate scheduler
        scheduler.step()

        # Log metrics to WandB
        wandb.log({"epoch": epoch, "loss": epoch_loss})

        # Save model checkpoint periodically
        if (epoch+1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, run_name, epoch, wandb)

        # Early stopping check
        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    # Save final model
    save_checkpoint(model, run_name, None, wandb, save_type="trained_models")

    # Close WandB run
    wandb.finish()