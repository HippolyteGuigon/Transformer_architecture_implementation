import torch
from torch import Tensor

from typing import Optional


def sigmoid(x: Tensor) -> Tensor:
    """
    The goal of this function is to
    implement the sigmoid function
    for neuron activation

    Arguments:
        -x: Tensor: The neuron
        to be activated
    Returns:
        -sigmoid_result: Tensor: The neuron
        output once activated
    """

    sigmoid_result = 1 / (1 + torch.exp(-x))

    return sigmoid_result


def relu(x: Tensor) -> Tensor:
    """
    The goal of this function is to
    implement the relu function
    for neuron activation

    Arguments:
        -x: Tensor: The neuron
        to be activated
    Returns:
        -relu_result: Tensor: The neuron
        output once activated
    """

    relu_result = torch.clamp(x, min=0)

    return relu_result


def softmax(x: Tensor, axis: Optional[int] = None) -> Tensor:
    """
    The goal of this function is to
    implement the softmax function
    for neuron activation

    Arguments:
        -x: Tensor: The neuron
        to be activated
        -axis: Optional[int]: The axis
        along which the max operation
        should be applied
    Returns:
        -softmax_result: Tensor:
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


def tanh(x: Tensor) -> Tensor:
    """
    The goal of this function is to
    implement the hyperbolic tangent
    function for neuron activation

    Arguments:
        -x: Tensor: The neuron
        to be activated
    Returns:
        -tanh_result: Tensor:
        The neuron output once activated
    """

    numerator = torch.exp(x) - torch.exp(-x)
    denominator = torch.exp(x) + torch.exp(-x)

    tanh_result = numerator / denominator

    return tanh_result
