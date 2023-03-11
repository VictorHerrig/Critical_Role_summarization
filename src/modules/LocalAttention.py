from torch import nn
from torch import Tensor


class LocalSelfAttention(nn.Module):
    def __init__(self, window_size=1024):
        """

        """
        super().__init__()
        self._window_size = window_size

    def forward(self, val: Tensor) -> Tensor:
        """

        Parameters
        ----------
        val: torch.Tensor
            Tensor of shape (batch x token x dim)

        Returns
        -------
        Tensor
            ...
        """


    @property
    def window_size(self):
        return self._window_size
