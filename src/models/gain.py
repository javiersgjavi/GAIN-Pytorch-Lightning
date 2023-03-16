import torch
import pytorch_lightning as pl
from typing import Dict, Tuple
from src.models.mlp import MLP
from src.utils import loss_d, loss_g


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

        # Three main components of the GAIN model
        self.generator = MLP(input_size=input_size)
        self.discriminator = MLP(input_size=input_size)
        self.hint_generator = HintGenerator(prop_hint=hint_rate)

        self.loss_mse = torch.nn.MSELoss()

        self.alpha = alpha

    # -------------------- Custom methods --------------------

    def calculate_error_imputation(self, outputs: Dict[str, torch.Tensor], type_step: str = 'train') -> None:
        """
            Calculates the mean squared error (MSE) and the root mean squared error (RMSE) between the real input
            and the imputed output of a batch.

            Args:
                outputs: A dictionary containing the output tensors for a batch.
                type_step: A string indicating whether the batch is for training or validation (default is 'train').
            """
        x_real = outputs['x_real']
        x_fake = outputs['x_fake']
        input_mask = outputs['input_mask_bool']

        mse = self.loss_mse(x_fake[~input_mask], x_real[~input_mask])

        self.log('mse', mse, prog_bar=True)
        self.log('rmse', torch.sqrt(mse), prog_bar=True)
        self.logger.experiment.add_scalars('mse_graph', {type_step: mse}, self.global_step)

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
        d_pred = outputs['d_pred']
        x_real = outputs['x_real']
        imputation = outputs['imputation']
        input_mask_int = outputs['input_mask_int']
        input_mask_bool = outputs['input_mask_bool']

        # --------------------- Discriminator loss ---------------------
        d_loss = loss_d(d_pred, input_mask_int)

        # --------------------- Generator loss -------------------------
        g_loss_adversarial = loss_g(d_pred, input_mask_int)
        g_loss_reconstruction = self.loss_mse(imputation[input_mask_bool], x_real[input_mask_bool])

        g_loss = g_loss_adversarial + self.alpha * g_loss_reconstruction
        # ---------------------------------------------------------------

        log_dict = {'Generator': g_loss_adversarial, 'Discriminator': d_loss}
        self.logger.experiment.add_scalars(f'G VS D (fake)', log_dict, self.global_step)
        self.log('G_loss_reconstruction', g_loss_reconstruction)

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
        x, x_real, input_mask_bool, input_mask_int = batch

        # Forward Generator
        x_fake, imputation = self.generator.forward_g(x=x, input_mask=input_mask_int)

        # Generate Hint Matrix
        hint_matrix = self.hint_generator.generate(input_mask_int)

        # Forward Discriminator
        d_pred = self.discriminator.forward_d(x=x_fake, hint_matrix=hint_matrix)

        res = {
            'x_real': x_real,
            'x_fake': x_fake,
            'd_pred': d_pred,
            'imputation': imputation,
            'input_mask_int': input_mask_int,
            'input_mask_bool': input_mask_bool,
        }
        return res

    # -------------------- Methods from PyTorch Lightning --------------------

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer]:
        """
            Configure the optimizers for the GAN model.
        """
        opt_d = torch.optim.Adam(self.discriminator.parameters())
        opt_g = torch.optim.Adam(self.generator.parameters())
        return opt_d, opt_g

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

        # Calculate the mean squared error (MSE) between the real and imputed data
        self.calculate_error_imputation(outputs)

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

        # Calculate the mean squared error (MSE) between the real and imputed data
        self.calculate_error_imputation(outputs, type_step='val')

    def test_step(self, batch: Tuple, batch_idx: int) -> None:
        """Runs a single test step on a batch of data.

        Args:
            batch (Tuple): Tuple of input data, `x_real`, `x`, and `input_mask`.
            batch_idx (int): Index of the current batch.
        """

        # Generate GAN outputs for the given batch
        outputs = self.return_gan_outputs(batch)

        # Calculate the mean squared error (MSE) between the real and imputed data
        self.calculate_error_imputation(outputs, type_step='test')
