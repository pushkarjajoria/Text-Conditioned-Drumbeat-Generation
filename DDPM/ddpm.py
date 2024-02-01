import logging
import os
from enum import Enum

import torch
from tqdm import tqdm

from DDPM.model import EncoderDecoderBN
from utils.utils import save_midi

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class BetaSchedule(Enum):
    LINEAR = 1
    QUADRATIC = 2
    SIGMOID = 3
    COSINE = 4


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, num_drum_instruments=9, num_time_slices=64,
                 channels=1, schedule: BetaSchedule = BetaSchedule.LINEAR):
        self.channels = channels
        self.width = num_drum_instruments
        self.height = num_time_slices
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.beta = self.prepare_noise_schedule(schedule).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self, beta_schedule):
        if beta_schedule == BetaSchedule.LINEAR:
            "Linear Schedule"
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif beta_schedule == BetaSchedule.QUADRATIC:
            "Quadratic Schedule"
            return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.noise_steps) ** 2
        else:
            raise NotImplementedError("")

    def noise_drum_beats(self, x, t):
        """
        :param x: drum beats of shape (batch x num_instrument=9, num_time_slices=64 sixteenth notes.)
        :param t: number of noising time steps of shape (batch_size, )
        :return: Noised drum beats and the sampled noise both with the shape the same as x
        """
        # Nones are changing tensor of shape (batch,) -> (batch, 1, 1)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_ht = torch.sqrt(1. - self.alpha_hat[t])[:, None, None]
        epsilon = torch.randn_like(x)  # ε
        """
        Refer to the paper to understand this equation better but basically,
        q(x_t|x_o) = N(x_t;√(alpha_hat_t_), √(1 - alpha_hat_t)I)
        Use Reparameterization Trick to get the below equation where ε is sampled from a unit normal distribution,
            q(x_t|x_0, t) = √(α_bar_t) + √(1 - α_bar_t)ε
        """
        q_xt = sqrt_alpha_hat * x + sqrt_one_minus_alpha_ht * epsilon
        return q_xt, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new drum beats")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.channels, self.width, self.height)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # x_t-1|(x_t, t)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 127).type(torch.uint8)
        return x


if __name__ == "__main__":
    diff = Diffusion()
    model = EncoderDecoderBN()
    samples = diff.sample(model, 2).numpy()
    save_midi(samples.squeeze(), os.path.join("../results", "DDPM_Unconditional_groove_monkee"), 0)
    print(samples)