import unittest
import torch
import torch.nn.functional as F

from unsupervised_pretraining.main import contrastive_loss


class TestContrastiveLoss(unittest.TestCase):

    def test_output_type(self):
        """Test if the output of the function is a torch.Tensor"""
        batch_size, embedding_dim, temperature = 10, 64, 0.07
        text_embeddings = torch.randn(batch_size, embedding_dim)
        midi_embeddings = torch.randn(batch_size, embedding_dim)

        loss = contrastive_loss(text_embeddings, midi_embeddings, temperature)
        self.assertIsInstance(loss, torch.Tensor, "Output should be a torch.Tensor")

    def test_symmetry(self):
        """Test if the loss is symmetric with respect to text and midi embeddings"""
        batch_size, embedding_dim, temperature = 10, 64, 0.07
        text_embeddings = torch.randn(batch_size, embedding_dim)
        midi_embeddings = torch.randn(batch_size, embedding_dim)

        loss1 = contrastive_loss(text_embeddings, midi_embeddings, temperature)
        loss2 = contrastive_loss(midi_embeddings, text_embeddings, temperature)
        self.assertAlmostEqual(loss1.item(), loss2.item(), msg="Loss should be symmetric")

    def test_known_values(self):
        """Test the function with known input and output values"""
        text_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        midi_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        temperature = 1.0

        expected_loss = torch.tensor(0.3466)  # This value should be pre-calculated for this specific case
        loss = contrastive_loss(text_embeddings, midi_embeddings, temperature)
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4,
                               msg="Loss does not match expected value for known input")

    # Add more tests as needed, such as for different batch sizes, embedding dimensions, etc.


if __name__ == '__main__':
    unittest.main()
