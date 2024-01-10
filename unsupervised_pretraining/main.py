import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
import wandb

from unsupervised_pretraining.create_unsupervised_dataset import MidiDataset, get_filenames_and_tags
from unsupervised_pretraining.model import CLAMP


# Other necessary imports: torchvision, numpy, etc.

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define hyperparameters from the config
EPOCHS = config['Training']['epochs']
BATCH_SIZE = config['Training']['batch_size']
LEARNING_RATE = config['Training']['learning_rate']
WEIGHT_DECAY = config['Training']['weight_decay']
CHECKPOINT_INTERVAL = config['Training']['checkpoint_interval']


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
    wandb.init(project="your_project_name", entity="your_wandb_username")
    # Initialize Dataloader
    file_name_and_tags = get_filenames_and_tags(dataset_dir="../datasets/Groove_Monkee_Mega_Pack_GM", filter_common_tags=True)
    train_dataset = MidiDataset(file_name_and_tags)  # Placeholder paths
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, optimizer, and learning rate scheduler
    model = CLAMP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # Example scheduler

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
            torch.save(model.state_dict(), f"model_checkpoint_epoch_{epoch}.pth")

        # Early stopping check
        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Save final model
    torch.save(model.state_dict(), "final_model.pth")

    # Close WandB run
    wandb.finish()