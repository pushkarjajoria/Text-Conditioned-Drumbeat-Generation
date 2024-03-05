import logging
import os
import torch
from tqdm import tqdm

from DDPM.ddpm import BetaSchedule
from DDPM.model import EncoderDecoderBN, ConditionalLatentEncDecMHA
from utils.utils import save_midi


class LatentDiffusion:
    """
    This class, just like the Diffusion class, handles the diffusion but only in the latent space.
    The input z is expected to be a vector and not a matrix/2D-Tensor (As is the case with images).
    """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, latent_dimension=128, schedule: BetaSchedule = BetaSchedule.QUADRATIC):
        self.latent_dimension = latent_dimension
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
            return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.noise_steps) ** 2
        elif beta_schedule == BetaSchedule.SIGMOID:
            "Sigmoid Schedule"
            # Creating a sigmoid schedule that smoothly transitions from beta_start to beta_end
            t = torch.linspace(-6, 6, self.noise_steps)  # Sigmoid input range
            return self.beta_start + (self.beta_end - self.beta_start) * (1 / (1 + torch.exp(-t)))
        elif beta_schedule == BetaSchedule.COSINE:
            "Cosine Schedule"
            # Creating a cosine schedule that starts and ends smoothly
            t = torch.linspace(0, torch.pi, self.noise_steps)  # Cosine input range
            return self.beta_start + (self.beta_end - self.beta_start) * (1 - torch.cos(t)) / 2
        else:
            raise NotImplementedError("")

    def noise_z(self, z, t):
        """
        :param x: A latent code of a midi pianoroll (batch x size of latent dimension)
        :param t: number of noising time steps of shape (batch_size, )
        :return: Noised Z and the sampled noise both with the shape the same as z (the latent code of a midi pianoroll)
        """
        # Nones are changing tensor of shape (batch,) -> (batch, 1, 1)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_ht = torch.sqrt(1. - self.alpha_hat[t])[:, None]
        epsilon = torch.randn_like(z)  # ε
        """
        Refer to the paper to understand this equation better but in short,
        q(x_t|x_o) = N(x_t;√(alpha_hat_t_), √(1 - alpha_hat_t)I)
        Use Reparameterization Trick to get the below equation where ε is sampled from a unit normal distribution,
            q(x_t|x_0, t) = √(α_bar_t) + √(1 - α_bar_t)ε
        """
        q_xt = sqrt_alpha_hat * z + sqrt_one_minus_alpha_ht * epsilon
        return q_xt, epsilon

    def denoise_z(self, model, z, noised_steps, text_embeddings):
        batch_len = z.shape[0]
        model.eval()
        with torch.no_grad():
            for i in tqdm(reversed(range(1, noised_steps)), position=0):
                t = (torch.ones(batch_len) * i).long().to(self.device)
                predicted_noise = model(z, t, text_embeddings).permute(0, 2, 1)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(z)
                else:
                    noise = torch.zeros_like(z)
                # x_t-1|(x_t, t)
                z = 1 / torch.sqrt(alpha) * (
                        z - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        return z

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_conditional(self, denoising_model, n, text_embeddings, midi_decoder):
        logging.info(f"Sampling {n} new drum beats")
        denoising_model.eval()
        with torch.no_grad():
            z = torch.randn((n, self.latent_dimension)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = denoising_model(z, t, text_embeddings)
                alpha = self.alpha[t][:, None]
                alpha_hat = self.alpha_hat[t][:, None]
                beta = self.beta[t][:, None]
                if i > 1:
                    noise = torch.randn_like(z)
                else:
                    noise = torch.zeros_like(z)
                # x_t-1|(x_t, t)
                z = 1 / torch.sqrt(alpha) * (
                        z - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        denoising_model.train()
        midi_decoder.eval()
        decoded_midi = midi_decoder.decode_midi(z)
        decoded_midi = decoded_midi.clamp(0, 1)
        decoded_midi = (decoded_midi * 127).type(torch.uint8)
        midi_decoder.train()
        return decoded_midi

    def _denoising_test(self, t, noise, noised_z, z):
        alpha_t = self.alpha[t][:, None]
        alpha_hat_t = self.alpha_hat[t][:, None]

        # Theoretical denoising step to reverse the noise addition based on known parameters
        # In practice, you would use the model's predicted noise instead of the actual noise
        denoised_z = (noised_z - torch.sqrt(1 - alpha_hat_t) * noise) / torch.sqrt(alpha_t)

        # Assert that denoised_z is close to the original z
        # Using torch.allclose with a tolerance for floating point arithmetic
        tolerance = 1e-3  # Tolerance level might need adjustment based on your specific requirements
        assert torch.allclose(denoised_z, z, atol=tolerance), \
            f"Denoised z is not close to the original z. Mean diff: {torch.mean(torch.abs(denoised_z - z))}"

        print("Assertion passed: Denoised z is close to the original z")


if __name__ == "__main__":
    diff = LatentDiffusion()
    batch_size = 1
    z = torch.stack([torch.rand(128) for _ in range(batch_size)])
    t = torch.randint(1, 1000, (batch_size,))
    noised_z, noise = diff.noise_z(z, t)
    diff._denoising_test(t, noise, noised_z, z)
