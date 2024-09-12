import torch


def softmax(x: torch.tensor) -> torch.tensor:
    """
    The goal of this function is to
    implement the softmax function
    for neuron activation

    Arguments:
        -x: torch.tensor: The neuron
        to be activated
    Returns:
        -softmax_result: torch.tensor:
        The neuron output once activated
    """

    shift_x = x - torch.max(x)
    numerator = torch.exp(shift_x)
    denominator = torch.sum(torch.exp(shift_x))

    softmax_result = numerator / denominator

    return softmax_result
