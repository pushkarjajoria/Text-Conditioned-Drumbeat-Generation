# Text Conditioned Drumbeat Generation
This study introduces a text-conditioned approach
to generating drumbeats with Latent Diffusion Models
(LDMs). It uses informative conditioning text extracted
from training data filenames. By pretraining a text and
drumbeat encoder through contrastive learning within a
multimodal network, aligned following CLIP, we align the
modalities of text and music closely. Additionally, we ex-
amine an alternative text encoder based on multihot text
encodings. Inspired by music’s multi-resolution nature,
we propose a novel LSTM variant, MultiResolutionLSTM,
designed to operate at various resolutions independently.
In common with recent LDMs in the image space, it speeds
up the generation process by running diffusion in a latent
space provided by a pretrained unconditional autoencoder.
We demonstrate the originality and variety of the gener-
ated drumbeats by measuring distance (both over binary pi-
anorolls and in the latent space) versus the training dataset
and among the generated drumbeats. We also assess the
generated drumbeats through a listening test focused on
questions of quality, aptness for the prompt text, and nov-
elty. We show that the generated drumbeats are novel and
apt to the prompt text, and comparable in quality to those
created by human musicians.
