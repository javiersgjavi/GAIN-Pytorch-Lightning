import torch
from typing import Dict, Tuple, List, Any
import pytorch_lightning as pl
from models.mlp import MLP
from utils import loss_c_d, loss_c_g


class HintGenerator:
    """
    Class that generates a hint matrix with the new definition of the hint matrix that can be found in the
    original repository.
    """

    def __init__(self, prop_hint: float):
        self.prop_hint = prop_hint

    def generate(self, input_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate a hint matrix with the same shape as the input mask tensor.

        Args:
            input_mask (torch.Tensor): Tensor of binary values indicating which values in the input are missing.

        Returns:
            hint_matrix (torch.Tensor): Tensor of binary values with the same shape as input_mask indicating with the
            hints to be used for the discriminator. The values are 1 if it is a known value and 0 if it is a value
            to be determined by the discriminator.
        """
        hint_mask = torch.rand(size=input_mask.size())
        hint_matrix = 1 * (hint_mask < self.prop_hint)
        hint_matrix = input_mask * hint_matrix.to(input_mask.device)
        return hint_matrix


class GAIN(pl.LightningModule):
    def __init__(self, input_size: int, alpha: float, hint_rate: float):
        """
        A PyTorch Lightning module implementing the GAIN (Generative Adversarial Imputation Network) algorithm.

        Args:
            input_size (int): The number of features in the input data.
            alpha (float): A hyperparameter controlling the weight of the reconstruction loss.
            hint_rate (float): The rate of known values in the hint matrix.

        Attributes:
            generator (MLP): The generator model.
            discriminator (MLP): The discriminator model.
            hint_generator (HintGenerator): The hint generator.
            loss_d (function): The discriminator loss function.
            loss_g (function): The generator loss function.
            loss_mse (torch.nn.MSELoss): The mean squared error loss function.
            alpha (int): A hyperparameter controlling the weight of the reconstruction loss.
        """
        super().__init__()
        super().save_hyperparameters()

        self.generator = MLP(input_size=input_size)
        self.discriminator = MLP(input_size=input_size)
        self.hint_generator = HintGenerator(prop_hint=hint_rate)

        self.loss_d = loss_c_d
        self.loss_g = loss_c_g
        self.loss_mse = torch.nn.MSELoss()

        self.alpha = alpha

    # -------------------- Methods from PyTorch Lightning --------------------

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Any]]:
        """
            Configure the optimizers for the GAN model.
        """
        opt_d = torch.optim.Adam(self.discriminator.parameters())
        opt_g = torch.optim.Adam(self.generator.parameters())
        return [opt_d, opt_g], []

    def training_step(self, batch: Tuple, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        """
        Runs a single training step on a batch of data.

        Args:
            batch (Tuple): Tuple of input data, `x_real`, `x`, and `input_mask`.
            batch_idx (int): Index of the current batch.
            optimizer_idx (int): Index of the optimizer to use for this step.

        Returns:
            Any: The computed loss for the current step.
        """

        # Generate GAN outputs for the given batch
        outputs = self.return_gan_outputs(batch)

        # Compute the discriminator and generator loss based on the generated outputs
        d_loss, g_loss = self.loss(outputs)

        # Calculate the root mean squared error (RMSE) between the real and imputed data
        self.calculate_rmse(outputs)

        # Select the appropriate loss based on the optimizer index
        loss = d_loss if optimizer_idx == 0 else g_loss

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> None:
        """Runs a single validation step on a batch of data.

        Args:
            batch (Tuple): Tuple of input data, `x_real`, `x`, and `input_mask`.
            batch_idx (int): Index of the current batch.
        """

        # Generate GAN outputs for the given batch
        outputs = self.return_gan_outputs(batch)

        # Calculate the root mean squared error (RMSE) between the real and imputed data
        self.calculate_rmse(outputs, type_step='val')

    def test_step(self, batch: Tuple, batch_idx: int) -> None:
        """Runs a single test step on a batch of data.

        Args:
            batch (Tuple): Tuple of input data, `x_real`, `x`, and `input_mask`.
            batch_idx (int): Index of the current batch.
        """

        # Generate GAN outputs for the given batch
        outputs = self.return_gan_outputs(batch)

        # Calculate the root mean squared error (RMSE) between the real and imputed data
        self.calculate_rmse(outputs, type_step='test')

    # -------------------- Custom methods --------------------

    def calculate_rmse(self, outputs: Dict[str, torch.Tensor], type_step: str = 'train') -> None:
        """
            Calculates the root mean squared error (RMSE) between the real input and the imputed output of a batch.

            Args:
                outputs: A dictionary containing the output tensors for a batch.
                type_step: A string indicating whether the batch is for training or validation (default is 'train').
            """
        x_real = outputs['x_real']
        imputation = outputs['imputation']
        input_mask = outputs['input_mask']
        mse = self.loss_mse(imputation[~input_mask], x_real[~input_mask])
        rmse = torch.sqrt(mse)
        self.logger.experiment.add_scalars('rmse', {type_step: rmse}, self.global_step)
        self.log('rmse_train', rmse, prog_bar=True)

    def loss(self, outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Calculates the loss of the generator and discriminator. The losses are calculated with the
            new method described in the original repository. With the new hint matrix definition, the
            losses are calculated with all the known values and the values to be determined by the discriminator.

            Args:
                outputs: A dictionary containing the output tensors for a batch.

            Returns:
                A tuple containing the discriminator loss and generator loss, respectively.
            """
        x_real = outputs['x_real']
        imputation = outputs['imputation']
        d_pred = outputs['d_pred']
        input_mask = outputs['input_mask']

        d_loss = self.loss_d(d_pred, input_mask)

        g_loss_fake = self.loss_g(d_pred, input_mask)
        g_loss_real = self.loss_mse(imputation[input_mask], x_real[input_mask])
        g_loss = g_loss_fake + self.alpha * g_loss_real

        log_dict = {'Generator': g_loss_fake, 'Discriminator': d_loss}
        self.logger.experiment.add_scalars(f'G VS D (fake)', log_dict, self.global_step)
        self.log('G_loss_real', g_loss_real)

        return d_loss, g_loss

    def return_gan_outputs(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Returns the output tensors of the generator and discriminator for a given batch.

        Args:
            batch: A tuple containing the real input tensor, the input tensor with missing values, and the input mask
            tensor.

        Returns:
            A dictionary containing the output tensors of the generator and discriminator for the batch, as well as the
            real input and the input mask.
        """
        x_real, x, input_mask = batch

        # Forward Generator
        x_fake, imputation = self.generator.forward_g(x=x, input_mask=input_mask)

        # Generate Hint Matrix
        hint_matrix = self.hint_generator.generate(input_mask)

        # Forward Discriminator
        d_pred = self.discriminator.forward_d(x=x_fake, hint_matrix=hint_matrix)

        res = {
            'x_real': x_real,
            'x_fake': x_fake,
            'imputation': imputation,
            'd_pred': d_pred,
            'input_mask': input_mask,
        }
        return res
