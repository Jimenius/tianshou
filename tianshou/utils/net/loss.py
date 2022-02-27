import torch
from torch import nn


class GaussianMLELoss(nn.Module):
    """Loss function from Maximum Likelihood Estimate of Gaussian distribution.

    :param float coeff: Coefficient of optional variable normalization.
    """

    def __init__(
        self,
        opt_coeff: float = 0.01,
    ) -> None:
        super().__init__()
        self._coeff = opt_coeff

    def forward(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        max_logvar: torch.Tensor,
        min_logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Execute loss calculation.

        :param torch.Tensor mean: tensor of mean from the network output.
        :param torch.Tensor logvar: tensor of logarithm of variance
            from the network output.
        :param torch.Tensor max_logvar: tensor of maximum logarithm of variance.
        :param torch.Tensor min_logvar: tensor of minimum logarithm of variance.
        :param torch.Tensor y: tensor of target.
        """
        inv_var = torch.exp(-logvar)
        mse = torch.mean(torch.square(mean - target) * inv_var)
        var_loss = torch.mean(logvar)
        opt_loss = self._coeff * (torch.sum(max_logvar) - torch.sum(min_logvar))
        loss = mse + var_loss + opt_loss

        return loss
