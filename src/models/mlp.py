import torch
from torch import nn
from src.utils import init_weights_xavier


class MLP(nn.Module):
    """
    The simple multi-layer perceptron architecture from the original code with a configurable input size.

    Args:
        input_size (int): The size of the input tensor.

    """

    def __init__(self, input_size: int = None):
        super().__init__()

        # Define the fully connected layers in a sequential block
        self.fc_block = nn.Sequential(
            nn.Linear(input_size * 2, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        ).apply(init_weights_xavier)

    def forward_g(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the generator network.

        Args:
            x (torch.Tensor): The input tensor.
            input_mask (torch.Tensor): A binary mask tensor indicating missing values.

        Returns:
            torch.Tensor: The output tensor of the generator network.

        """
        noise_matrix = torch.distributions.uniform.Uniform(0, 0.01).sample(x.shape).to(x.device)

        # Concatenate the input tensor with the noise matrix
        x = input_mask * x + (1 - input_mask) * noise_matrix

        input_tensor = torch.cat([x, input_mask], dim=1)

        imputation = self.fc_block(input_tensor)

        # Concatenate the original data with the imputed data
        res = input_mask * x + (1 - input_mask) * imputation

        return res, imputation

    def forward_d(self, x: torch.Tensor, hint_matrix: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the discriminator network.

        Args:
            x (torch.Tensor): The input tensor.
            hint_matrix (torch.Tensor): A binary matrix indicating which elements of the input tensor are missing.

        Returns:
            torch.Tensor: The output tensor of the discriminator network.

        """
        x = torch.cat([x, hint_matrix], dim=1)
        return self.fc_block(x)
