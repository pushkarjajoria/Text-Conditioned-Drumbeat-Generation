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


def save_checkpoint(model, run_name, epoch, wandb, save_type="checkpoint", dir="unsupervised_pretraining"):
    path = os.path.join(dir, save_type, run_name)
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
    wandb.config.update({
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS
    })

    # Log system and training configuration
    print(f"Running on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}, Weight decay: {WEIGHT_DECAY}, Epochs: {EPOCHS}")

    # Initialize Dataloader
    file_name_and_tags = get_filenames_and_tags(dataset_dir="datasets/Groove_Monkee_Mega_Pack_GM", filter_common_tags=True)
    print(f"Length of dataset: {len(file_name_and_tags)}")
    train_dataset = MidiDataset(file_name_and_tags)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"DataLoader initialized with {len(train_loader)} batches per epoch")

    # Initialize model, optimizer, and learning rate scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    model = CLAMP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.3)
    print("Model, optimizer, and scheduler initialized")

    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    print("Early stopping setup with patience: 10 and min_delta: 0.001")

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
            batch_loss = loss.item()
            wandb.log({"batch_loss": batch_loss, "epoch": epoch})

        epoch_loss /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}, Learning rate: {scheduler.get_last_lr()}")

        # Step the learning rate scheduler
        scheduler.step()

        # Log metrics to WandB
        wandb.log({"epoch": epoch, "loss": epoch_loss, "learning_rate": scheduler.get_last_lr()})

        # Save model checkpoint periodically
        if (epoch+1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, run_name, epoch, wandb)
            print(f"Checkpoint saved for Epoch {epoch}")

        # Early stopping check
        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    # Save final model
    save_checkpoint(model, run_name, None, wandb, save_type="trained_models")
    print("Final model saved")

    # Close WandB run
    wandb.finish()
    print("Training complete, WandB run finished")