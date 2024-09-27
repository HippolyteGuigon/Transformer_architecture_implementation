import torch
from torch import Tensor

from typing import Optional


def relu(x: Tensor, axis: Optional[int] = None) -> Tensor:
    """
    The goal of this function is to
    implement the relu function
    for neuron activation

    Arguments:
        -x: torch.tensor: The neuron
        to be activated
        -axis: Optional[int]: The axis
        along which the max operation
        should be applied
    Returns:
        -relu_result: torch.tensor:
        The neuron output once activated
    """

    relu_result = torch.clamp(x, min=0)

    return relu_result


def softmax(x: Tensor, axis: Optional[int] = None) -> Tensor:
    """
    The goal of this function is to
    implement the softmax function
    for neuron activation

    Arguments:
        -x: torch.tensor: The neuron
        to be activated
        -axis: Optional[int]: The axis
        along which the max operation
        should be applied
    Returns:
        -softmax_result: torch.tensor:
        The neuron output once activated
    """

    if axis is None:
        x = x.view(-1)
        axis = 1

    shift_x = x - torch.max(x, dim=axis, keepdims=True)[0]
    numerator = torch.exp(shift_x)
    denominator = torch.sum(torch.exp(shift_x), dim=axis, keepdims=True)

    softmax_result = numerator / denominator

    return softmax_result
