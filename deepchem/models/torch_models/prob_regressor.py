"""PyTorch implementation of fully connected networks.
"""
import logging
import numpy as np
import torch
import torch.nn.functional as F
from collections.abc import Sequence as SequenceCollection

import deepchem as dc
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import _make_pytorch_shapes_consistent
from deepchem.metrics import to_one_hot
from deepchem.models.losses import SmoothKLLoss2D, SmoothCELoss2D
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
from deepchem.utils.typing import ActivationFn, LossFn, OneOrMany
from deepchem.utils.pytorch_utils import get_activation

logger = logging.getLogger(__name__)




class ProbabilityRegressor(TorchModel):
    """A fully connected network for multitask regression.

    This class provides lots of options for customizing aspects of the model: the
    number and widths of layers, the activation functions, regularization methods,
    etc.

    It optionally can compose the model from pre-activation residual blocks, as
    described in https://arxiv.org/abs/1603.05027, rather than a simple stack of
    dense layers.  This often leads to easier training, especially when using a
    large number of layers.  Note that residual blocks can only be used when
    successive layers have the same width.  Wherever the layer width changes, a
    simple dense layer will be used even if residual=True.
    """

    def __init__(self,
                 n_tasks: int,
                 n_features: int,
                 output_bins: int = 36,
                 layer_sizes: Sequence[int] = [1000],
                 weight_init_stddevs: OneOrMany[float] = 0.02,
                 bias_init_consts: OneOrMany[float] = 1.0,
                 weight_decay_penalty: float = 0.0,
                 weight_decay_penalty_type: str = 'l2',
                 dropouts: OneOrMany[float] = 0.5,
                 activation_fns: OneOrMany[ActivationFn] = 'relu',
                 uncertainty: bool = False,
                 residual: bool = False,
                 **kwargs) -> None:
        """Create a MultitaskRegressor.

        In addition to the following arguments, this class also accepts all the keywork arguments
        from TensorGraph.

        Parameters
        ----------
        n_tasks: int
            number of tasks
        n_features: int
            number of features
        layer_sizes: list
            the size of each dense layer in the network.  The length of this list determines the number of layers.
        weight_init_stddevs: list or float
            the standard deviation of the distribution to use for weight initialization of each layer.  The length
            of this list should equal len(layer_sizes)+1.  The final element corresponds to the output layer.
            Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
        bias_init_consts: list or float
            the value to initialize the biases in each layer to.  The length of this list should equal len(layer_sizes)+1.
            The final element corresponds to the output layer.  Alternatively this may be a single value instead of a list,
            in which case the same value is used for every layer.
        weight_decay_penalty: float
            the magnitude of the weight decay penalty to use
        weight_decay_penalty_type: str
            the type of penalty to use for weight decay, either 'l1' or 'l2'
        dropouts: list or float
            the dropout probablity to use for each layer.  The length of this list should equal len(layer_sizes).
            Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
        activation_fns: list or object
            the PyTorch activation function to apply to each layer.  The length of this list should equal
            len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
            same value is used for every layer.  Standard activation functions from torch.nn.functional can be specified by name.
        uncertainty: bool
            if True, include extra outputs and loss terms to enable the uncertainty
            in outputs to be predicted
        residual: bool
            if True, the model will be composed of pre-activation residual blocks instead
            of a simple stack of dense layers.
        """
        self.n_tasks = n_tasks
        self.n_features = n_features
        self.output_bins = output_bins
        n_layers = len(layer_sizes)
        if not isinstance(weight_init_stddevs, SequenceCollection):
            weight_init_stddevs = [weight_init_stddevs] * (n_layers + 1)
        if not isinstance(bias_init_consts, SequenceCollection):
            bias_init_consts = [bias_init_consts] * (n_layers + 1)
        if not isinstance(dropouts, SequenceCollection):
            dropouts = [dropouts] * n_layers
        if isinstance(
                activation_fns,
                str) or not isinstance(activation_fns, SequenceCollection):
            activation_fns = [activation_fns] * n_layers
        activation_fns = [get_activation(f) for f in activation_fns]
        if uncertainty:
            if any(d == 0.0 for d in dropouts):
                raise ValueError(
                    'Dropout must be included in every layer to predict uncertainty'
                )

        # Define the PyTorch Module that implements the model.

        class PytorchImpl(torch.nn.Module):

            def __init__(self):
                super(PytorchImpl, self).__init__()
                self.layers = torch.nn.ModuleList()
                prev_size = n_features
                for size, weight_stddev, bias_const in zip(
                        layer_sizes, weight_init_stddevs, bias_init_consts):
                    layer = torch.nn.Linear(prev_size, size)
                    torch.nn.init.normal_(layer.weight, 0, weight_stddev)
                    torch.nn.init.constant_(layer.bias, bias_const)
                    self.layers.append(layer)
                    prev_size = size
                self.output_layer = torch.nn.Linear(prev_size, n_tasks)
                torch.nn.init.normal_(self.output_layer.weight, 0,
                                      weight_init_stddevs[-1])
                torch.nn.init.constant_(self.output_layer.bias,
                                        bias_init_consts[-1])
                self.uncertainty_layer = torch.nn.Linear(prev_size, n_tasks)
                torch.nn.init.normal_(self.output_layer.weight, 0,
                                      weight_init_stddevs[-1])
                torch.nn.init.constant_(self.output_layer.bias, 0)

            def forward(self, inputs):
                x, dropout_switch = inputs
                prev_size = n_features
                next_activation = None
                for size, layer, dropout, activation_fn, in zip(
                        layer_sizes, self.layers, dropouts, activation_fns):
                    y = x
                    if next_activation is not None:
                        y = next_activation(x)
                    y = layer(y)
                    if dropout > 0.0 and dropout_switch:
                        y = F.dropout(y, dropout)
                    if residual and prev_size == size:
                        y = x + y
                    x = y
                    prev_size = size
                    next_activation = activation_fn
                if next_activation is not None:
                    y = next_activation(y)
                logits =self.output_layer(y)

                logits = logits.view(-1, output_bins, output_bins)  # Reshape into 36x36

                logits = F.softmax(logits.view(logits.size(0), -1), dim=1)  # 2D Softmax
                prob_density = logits.view(-1, output_bins, output_bins)
                prob_density = prob_density / prob_density.sum(dim=[1, 2], keepdim=True)  # Normalize across 2D binscd 

                prob_density = prob_density.view(-1, output_bins*output_bins)
                return prob_density, logits



        model = PytorchImpl()
        regularization_loss: Optional[Callable]
        if weight_decay_penalty != 0:
            weights = [layer.weight for layer in model.layers]
            if weight_decay_penalty_type == 'l1':
                regularization_loss = lambda: weight_decay_penalty * torch.sum(  # noqa: E731
                    torch.stack([torch.abs(w).sum() for w in weights]))
            else:
                regularization_loss = lambda: weight_decay_penalty * torch.sum(  # noqa: E731
                    torch.stack([torch.square(w).sum() for w in weights]))
        else:
            regularization_loss = None
        loss: Union[dc.models.losses.Loss, LossFn]

        output_types = ['prediction', 'loss']
        
        # have to implement loss first
        loss: Loss = SmoothCELoss2D()
        super(ProbabilityRegressor,
              self).__init__(model,
                             loss,
                             output_types=output_types,
                             regularization_loss=regularization_loss,
                             **kwargs)

    def default_generator(
            self,
            dataset: dc.data.Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                if mode == 'predict':
                    dropout = np.array(0.0)
                else:
                    dropout = np.array(1.0)
                yield ([X_b, dropout], [y_b], [w_b])
