"""
network.py  —  fully-connected network used by the PINN
"""

import torch
import torch.nn as nn


class DNN(nn.Module):
    """Fully-connected network with optional layer-wise learnable activation scaling."""

    def __init__(self, layers, w_init="xavier", b_init="zeros",
                 activation="tanh", laaf=False, dtype=torch.float64):
        super().__init__()
        self.laaf = laaf
        self.dtype = dtype
        activations = {"tanh": torch.tanh, "silu": torch.nn.functional.silu,
                       "gelu": torch.nn.functional.gelu}
        try:
            self.activation = activations[activation.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported activation {activation!r}; choose {tuple(activations)}") from exc
        if len(layers) < 2 or any(int(n) <= 0 for n in layers):
            raise ValueError("layers must contain at least two positive dimensions")

        self.weights = nn.ParameterList()
        self.biases  = nn.ParameterList()
        self.alphas  = nn.ParameterList() if laaf else []

        for in_dim, out_dim in zip(layers[:-1], layers[1:]):
            w = torch.empty(in_dim, out_dim, dtype=dtype)
            if w_init.lower() == "xavier":
                nn.init.xavier_uniform_(w)
            else:
                nn.init.uniform_(w, -1.0, 1.0)
            self.weights.append(nn.Parameter(w))

            b = (torch.zeros(1, out_dim, dtype=dtype) if b_init.lower() == "zeros"
                 else nn.init.uniform_(torch.empty(1, out_dim, dtype=dtype), -1.0, 1.0))
            self.biases.append(nn.Parameter(b))

            if laaf:
                self.alphas.append(nn.Parameter(10.0 * torch.ones(out_dim, dtype=dtype)))

    def forward(self, x):
        h = x
        for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            h = torch.mm(h, W) + b
            if self.laaf:
                h = self.activation(h * self.alphas[i])
            else:
                h = self.activation(h)
        return torch.mm(h, self.weights[-1]) + self.biases[-1]
