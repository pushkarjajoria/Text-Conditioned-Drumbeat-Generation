import os

import numpy as np
import torch
from ddpm import Diffusion
from midi_processing.mid2numpy import write_image
from model import EncoderDecoderBN
from utils import save_midi
from scipy.stats import kstest


def ks_test(input_array=np.random.normal(size=(9, 64))):
    # Flatten the input array
    flat_input = input_array.flatten()

    # Compute the KS test against a unit normal distribution
    ks_statistic, p_value = kstest(flat_input, 'norm')

    # Output the test results
    print("Kolmogorov-Smirnov test results:")
    print("KS statistic: {0:.4f}".format(ks_statistic))
    print("p-value: {0:.4f}".format(p_value))
    if p_value < 0.05:
        print("The null hypothesis can be rejected, the sample may not come from a unit normal distribution.")
    else:
        print("The null hypothesis cannot be rejected, the sample may come from a unit normal distribution.")


def test_model():
    state_dict = torch.load("models/DDPM_Unconditional_groove_monkee_20230422002713/checkpoint.pt")
    model = EncoderDecoderBN()
    model.load_state_dict(state_dict)
    model.eval()
    diff = Diffusion()
    sampled_beats = diff.sample(model, n=5).numpy().squeeze()
    save_midi(sampled_beats, os.path.join("../results", "DDPM_Unconditional_groove_monkee_20230422002713"), "Test_GT_80", 80)


def test_noising():
    def _save(noised_beats, steps):
        noised_beats = (noised_beats.clamp(-1, 1) + 1) / 2
        noised_beats = (noised_beats * 127).type(torch.uint8)
        noised_beats = noised_beats.squeeze().numpy()
        save_midi(noised_beats, "test_noise", steps, 0)
        [write_image(n, os.path.join("test_noise", f"Epoch{steps}", f"sample{i}")) for i, n in enumerate(noised_beats)]

    data = np.load("./datasets/Groove_Monkee_Mega_Pack_GM.npy")
    data = data.astype(np.float32)
    data = data.reshape((data.shape[0], 1, data.shape[1], data.shape[2]))
    # Scaling values from -1 to 1
    data = ((data/127) * 2) - 1
    diffusion = Diffusion()
    drum_beats = torch.Tensor(data[0:2])
    _save(drum_beats, "Original")
    t = torch.Tensor(np.array([999, 999], dtype=np.int32))
    t = t.long()
    torch.manual_seed(42)
    for steps in [0, 5, 10, 50, 100, 999]:
        steps = torch.Tensor(np.array([steps, steps], dtype=np.int32))
        steps = steps.long()
        noised_beats, e = diffusion.noise_drum_beats(drum_beats, steps)
        _save(noised_beats, steps)
    ks_test(noised_beats.numpy())
    # noised_beats = (noised_beats.clamp(-1, 1) + 1) / 2
    # noised_beats = (noised_beats * 127).type(torch.uint8)
    # noised_beats = noised_beats.numpy().squeeze()
    # path = "../testing output"
    # save_midi(noised_beats, path, epoch="_1000TS", ghost_threshold=0)
    print("done")


if __name__ == "__main__":
    test_noising()
