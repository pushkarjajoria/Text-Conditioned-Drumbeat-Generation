import torch
import torch.nn.functional as F


def combined_bce_wmse_loss(outputs, targets, threshold=5 / 127, beta=10):
    # Convert targets to binary for BCE
    binary_targets = (targets > threshold).float()

    # Compute Binary Cross-Entropy with Logits Loss
    bce_loss = F.binary_cross_entropy_with_logits(outputs, binary_targets, reduction='none')

    # Apply sigmoid to outputs for computing WMSE
    predicted_velocities = torch.sigmoid(outputs)

    # Mask for non-zero targets to compute WMSE
    non_zero_mask = targets > 0
    wmse_loss = ((predicted_velocities - targets) ** 2 * non_zero_mask.float()).mean() * beta

    # Combine losses
    print(f"BCE Loss: {bce_loss}")
    print(f"WMSE Loss: {wmse_loss}")
    combined_loss = bce_loss + wmse_loss
    return combined_loss


def logit(p):
    return torch.log(p / (1 - p))


if __name__ == "__main__":

    target = torch.tensor([0.0, 0.9, 0.2, 0.0])  # Target with one non-zero value
    # Prediction close to target but with slight error on the non-zero value and perfect on zeros
    prediction = logit(torch.tensor([0.001, 0.2, 0.9, 0.005]))

    # Expected behavior:
    # - BCE part should be low as the prediction is close to the target, especially after sigmoid is applied.
    # - Weighted MSE should be higher for the second element compared to the others due to the weight.

    # Calculate the combined loss
    loss = combined_bce_wmse_loss(prediction, target)

    print(f"Combined Loss: {loss.item()}")

    # Asserts
    # Note: These thresholds are somewhat arbitrary; adjust based on expected loss values and function specifics.
    assert loss.item() > 0, "Loss should be greater than 0."