import logging
import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ddpm import Diffusion
from model import UnconditionalEncDecMHA
from unsupervised_pretraining.model import CLAMP
from utils.utils import setup_logging, get_data, save_midi
import torch.nn as nn
import argparse


def train(args):
    prev_epoch = 0
    setup_logging(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = get_data(args)
    # model = EncDecWithMHA(args.time_embedding_dimension).to(device)
    model = UnconditionalEncDecMHA(args.time_embedding_dimension).to(device)
    clamp_model = CLAMP()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion()
    logger = SummaryWriter(os.path.join("../runs", args.run_name))
    l = len(dataloader)     # Number of batches
    if args.warm_start:
        state_dict = torch.load(args.checkpoint_path)
        model.load_state_dict(state_dict)
        prev_epoch = args.prev_epoch

    # Change the dataloader same as unconditional as we also need the tags for each midi file to generate the
    #  text embeddings.

    for epoch in range(prev_epoch, prev_epoch + args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, drum_beats in enumerate(pbar):
            drum_beats = drum_beats.to(device)  # batch x channel x img_w x img_h
            t = diffusion.sample_timesteps(drum_beats.shape[0]).to(device)  # batch_size x 1
            x_t, noise = diffusion.noise_drum_beats(drum_beats, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch*l + i)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join("../checkpoint", args.run_name, f"checkpoint.pt"))
            sampled_beats = diffusion.sample(model, n=5).numpy().squeeze()
            save_midi(sampled_beats, os.path.join("../results", args.run_name), epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = f"Unconditional_MHA"
    args.epochs = 20
    args.batch_size = 128
    args.dataset_path = "../datasets/Groove_Monkee_Mega_Pack_GM.npy"
    args.lr = 3e-4
    args.time_embedding_dimension = 128
    args.warm_start = False
    args.checkpoint_path = "../checkpoint/Unconditional_MHA/checkpoint.pt"
    args.prev_epoch = 20
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(args)
