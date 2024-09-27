import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """
    The goal of this class is
    to implement the residual
    connection to avoid vanishing
    gradient problem by adding the
    original input of a layer to the
    layer output

    Arguments:
        -in_dimensions: int: The dimensions
        of the input Tensor
        -out_dimensions: int: The dimensions
        of the output Tensor
    Returns
        -None
    """

    def __init__(self, in_dimensions: int, out_dimensions: int) -> None:
        super().__init__()
        self.in_dimensions = in_dimensions
        self.out_dimensions = out_dimensions

        if self.in_dimensions != self.out_dimensions:
            self.downsample_layer = nn.Linear(
                self.in_dimensions, self.out_dimensions
            )
        else:
            self.downsample_layer = None

    def forward(self, X: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        The goal of this method is
        to add the input and output
        Tensors together as the skip
        connexion

        Arguments:
            -X: torch.Tensor: The tensor
            to be added
            -output: torch.Tensor: The
            output of the layer
        Returns:
            -residual_tensor: torch.Tensor:
            The new Tensor after addition
        """

        if self.downsample_layer:
            X = self.downsample_layer(X)

        residual_tensor = output + X

        return residual_tensor
