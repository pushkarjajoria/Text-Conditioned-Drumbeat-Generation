import os
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from DDPM.main import load_or_process_dataset
from text_supervised_pretraining.model import CLAMP

# Other necessary imports: torchvision, numpy, etc.

with open('text_supervised_pretraining/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define hyperparameters from the config
EPOCHS = config['Training']['epochs']
BATCH_SIZE = config['Training']['batch_size']
LEARNING_RATE = config['Training']['learning_rate']
WEIGHT_DECAY = config['Training']['weight_decay']
CHECKPOINT_INTERVAL = config['Training']['checkpoint_interval']


if __name__ == "__main__":
    project_name = 'Unsupervised Pretraining' if torch.cuda.is_available() else '[Dev][Mac] Unsupervised Pretraining'
    run_name = "CUDA_test_mac"

    train_dataset = load_or_process_dataset(dataset_dir="datasets/Groove_Monkee_Mega_Pack_GM")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, optimizer, and learning rate scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    model = CLAMP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.3)

    # Set the seed for PyTorch
    torch.manual_seed(42)
    # Set the seed for Python's random module
    random.seed(42)
    # Set the seed for NumPy
    np.random.seed(42)
    # If you're using GPU, you also need to set the seed for CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Training loop
    for midi_data, text_data in tqdm(train_loader):
        text_embeddings, midi_embeddings = model(midi_data, text_data)
        loss = model.contrastive_loss(text_embeddings, midi_embeddings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Step the learning rate scheduler
        scheduler.step()
        # Save the model after 1 forward pass
        path = os.path.join(dir, save_type, run_name)
        os.makedirs(path, exist_ok=True)
        filename = f'model_epoch_{epoch}.pth' if epoch \
            else 'model_final.pth'
        full_path = os.path.join(path, filename)
        # Save locally
        torch.save(model.state_dict(), full_path)

    print("Test finished")
