import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model import WeightedSumAndMax, GCN
from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel
from typing import Optional
import numpy as np


import torch.nn as nn
import torch.nn.functional as F

from dgllife.model import WeightedSumAndMax, GCN

class CustomGCNWithReadout(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_feats,
                 gnn_norm=None,
                 activation=None,
                 residual=True,
                 batchnorm=False,
                 dropout=0.0,
                 nfeat_name='x',
                 mode='regression',
                 n_tasks=1,
                 n_classes=2):
        """
        Custom GCN with graph readout layer, including validation for mode and DGL dependencies.

        Parameters
        ----------
        in_feats: int
            Number of input features for each node.
        hidden_feats: list of int
            List of hidden layer sizes for the GCN.
        gnn_norm: str, optional
            Normalization type for the GCN layers.
        activation: callable, optional
            Activation function applied to GCN layers.
        residual: bool
            Whether to add residual connections in GCN layers.
        batchnorm: bool
            Whether to apply batch normalization.
        dropout: float
            Dropout rate for GCN layers.
        nfeat_name: str
            Key for the node features in the DGLGraph.
        mode: str
            Either 'regression' or 'classification'. Default is 'regression'.
        n_tasks: int
            Number of prediction tasks.
        n_classes: int
            Number of classes for classification (only used if mode='classification').
        """
        try:
            import dgl  # noqa: F401
        except ImportError:
            raise ImportError('This class requires dgl.')
        try:
            import dgllife  # noqa: F401
        except ImportError:
            raise ImportError('This class requires dgllife.')

        if mode not in ['classification', 'regression']:
            raise ValueError("mode must be either 'classification' or 'regression'.")

        super(CustomGCNWithReadout, self).__init__()

        self.n_tasks = n_tasks
        self.mode = mode
        self.n_classes = n_classes
        self.nfeat_name = nfeat_name

        # Determine the output size
        if mode == 'classification':
            out_size = n_tasks * n_classes
        else:
            out_size = n_tasks

        # Ensure graph_conv_layers is defined
        if hidden_feats is None:
            hidden_feats = [64, 64]
        num_gnn_layers = len(hidden_feats)


        # GCN layers
        self.gnn = GCN(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            gnn_norm=gnn_norm,
            activation=[activation] * num_gnn_layers,
            residual=[residual] * num_gnn_layers,
            batchnorm=[batchnorm] * num_gnn_layers,
            dropout=[dropout] * num_gnn_layers
        )

        # Readout for graph-level features
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
    
    def forward(self, g, feats):
        """Predict graph labels

        Parameters
        ----------
        g: DGLGraph
            A DGLGraph for a batch of graphs. It stores the node features in
            ``dgl_graph.ndata[self.nfeat_name]``.

        Returns
        -------
        torch.Tensor
            The model output.

        * When self.mode = 'regression',
            its shape will be ``(dgl_graph.batch_size, self.n_tasks)``.
        * When self.mode = 'classification', the output consists of probabilities
            for classes. Its shape will be ``(dgl_graph.batch_size, self.n_tasks, self.n_classes)``
            if self.n_tasks > 1; its shape will be ``(dgl_graph.batch_size, self.n_classes)`` if
            self.n_tasks is 1.
        torch.Tensor, optional
            This is only returned when self.mode = 'classification', the output consists of the
            logits for classes before softmax.
        """
        feats =  self.gnn (g, feats)
        graph_feats = self.readout(g, feats)
        
        return graph_feats




class CombinedGCNMLPModel(nn.Module):
    def __init__(self,
                 in_feats,
                 gcn_hidden_feats,
                 rdkit_input_size,
                 rdkit_hidden_feats,
                 combined_hidden_feats,
                 n_tasks=1,
                 activation=F.relu,
                 mode='regression',
                 n_classes=2,
                nfeat_name: str = 'x'):
        """
        Combined model integrating GCN graph features and RDKit descriptors for property prediction.

        Parameters
        ----------
        in_feats: int
            Number of input features for each node.
        gcn_hidden_feats: list of int
            List of hidden layer sizes for the GCN.
        rdkit_input_size: int
            The dimensionality of RDKit descriptors.
        rdkit_hidden_feats: list of int
            List of hidden layer sizes for the RDKit MLP.
        combined_hidden_feats: list of int
            List of hidden layer sizes for the combined MLP.
        n_tasks: int
            Number of prediction tasks.
        activation: callable
            Activation function applied to hidden layers.
        mode: str
            Either 'regression' or 'classification'. Default is 'regression'.
        n_classes: int
            Number of classes for classification (only used if mode='classification').
        """
        try:
            import dgl  # noqa: F401
        except ImportError:
            raise ImportError('This class requires dgl.')
        try:
            import dgllife  # noqa: F401
        except ImportError:
            raise ImportError('This class requires dgllife.')

        if mode not in ['classification', 'regression']:
            raise ValueError("mode must be either 'classification' or 'regression'.")

        super(CombinedGCNMLPModel, self).__init__()

        self.n_tasks = n_tasks
        self.mode = mode
        self.n_classes = n_classes
        self.nfeat_name = nfeat_name

        # Determine the output size for GCN
        if mode == 'classification':
            self.gcn_out_size = n_tasks * n_classes
        else:
            self.gcn_out_size = n_tasks

        # Graph-based feature extractor (GCN)
        self.graph_extractor = CustomGCNWithReadout(
            in_feats=in_feats,
            hidden_feats=gcn_hidden_feats,
            gnn_norm=None,
            activation=activation,
            residual=True,
            batchnorm=True,
            dropout=0.1,
            mode=mode,
            n_tasks=n_tasks,
            n_classes=n_classes
        )

        # RDKit descriptor feature extractor (MLP)
        rdkit_encoder_layers = []
        for i in range(len(rdkit_hidden_feats)):
            if i == 0:
                rdkit_encoder_layers.append(nn.Linear(rdkit_input_size, rdkit_hidden_feats[i]))
            else:
                rdkit_encoder_layers.append(nn.Linear(rdkit_hidden_feats[i-1], rdkit_hidden_feats[i]))
            rdkit_encoder_layers.append(nn.ReLU())  # Add ReLU after each linear layer
        self.rdkit_encoder = nn.Sequential(*rdkit_encoder_layers)

        # Combined prediction network (MLP)
        combined_input_size = rdkit_hidden_feats[-1] + gcn_hidden_feats[-1] * 2  # Input size depends on last hidden size
        combined_mlp_layers = []

        for i in range(len(combined_hidden_feats)):
            if i == 0:
                combined_mlp_layers.append(nn.Linear(combined_input_size, combined_hidden_feats[i]))
            else:
                combined_mlp_layers.append(nn.Linear(combined_hidden_feats[i-1], combined_hidden_feats[i]))
            combined_mlp_layers.append(nn.ReLU())

        combined_mlp_layers.append(nn.Linear(combined_hidden_feats[-1], self.gcn_out_size)) # Output layer

        self.final_mlp = nn.Sequential(*combined_mlp_layers)

     

    def forward(self, inputs):
        """
        Forward pass of the combined model.

        Parameters
        ----------
        graph: DGLGraph
            Batched DGLGraph.
        node_feats: torch.Tensor
            Node features for the graph.
        rdkit_feats: torch.Tensor
            RDKit descriptors for the graphs in the batch.

        Returns
        -------
        torch.Tensor
            The model's predictions.
        """
        # Graph-based features from GCN
        graph, rdkit_feats = inputs
        node_feats = graph.ndata[self.nfeat_name]
        graph_output = self.graph_extractor(graph, node_feats)

        # Descriptor-based features from RDKit MLP
        rdkit_embedding = self.rdkit_encoder(rdkit_feats)

        # Combine graph features and descriptor features
        combined_features = torch.cat([graph_output, rdkit_embedding], dim=1)

        # Final predictions
        predictions = self.final_mlp(combined_features)
        
        if self.mode == 'classification':
            if self.n_tasks == 1:
                logits = predictions.view(-1, self.n_classes)
                softmax_dim = 1
            else:
                logits = predictions.view(-1, self.n_tasks, self.n_classes)
                softmax_dim = 2
            proba = F.softmax(logits, dim=softmax_dim)
            return proba, logits
        else:
            return predictions

        return predictions


class GCNRdkitModel(TorchModel):
    """
    DeepChem-compatible model combining GCN graph features with RDKit descriptors.
    """
    def __init__(self,
                 n_tasks: int,
                 graph_conv_layers: Optional[list] = None,
                 activation=None,
                 rdkit_input_size: int = 210,
                 rdkit_hidden_feats: list = [64, 64, 64],
                 combined_hidden_feats: list = [128, 64],
                 mode: str = 'regression',
                 number_atom_features: int = 30,
                 n_classes: int = 2,
                 self_loop: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        Inherits from CombinedGCNMLPModel and TorchModel.
        """
        if graph_conv_layers is None:
            graph_conv_layers = [64, 64]
        if activation is None:
            activation = F.relu

        # Initialize the base model
        model = CombinedGCNMLPModel(in_feats=number_atom_features,
            gcn_hidden_feats=graph_conv_layers,
            rdkit_input_size=rdkit_input_size,
            rdkit_hidden_feats=rdkit_hidden_feats,
            combined_hidden_feats=combined_hidden_feats,
            n_tasks=n_tasks,
            activation=activation,
            mode=mode,
            n_classes=n_classes
        )

        # Define loss and output types
        if mode == 'regression':
            loss: Loss = L2Loss()
            output_types = ['prediction']
        elif mode == 'classification':
            loss = SparseSoftmaxCrossEntropy()
            output_types = ['prediction', 'loss']
        else:
            raise ValueError("Mode must be either 'regression' or 'classification'.")

        # Initialize the TorchModel
        super(GCNRdkitModel, self).__init__(model,
                                                     loss=loss,
                                                     output_types=output_types,
                                                     **kwargs)

        self._self_loop = self_loop

    def _prepare_batch(self, batch):
        """
        Prepare batch data for GCN and RDKit descriptors.
        """
        try:
            import dgl
        except ImportError:
            raise ImportError('This class requires dgl.')

        inputs, labels, weights = batch

        # Convert graphs to DGLGraphs with self-loops
        dgl_graphs = [
            graph[0].to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0] ]
        dgl_batch = dgl.batch(dgl_graphs).to(self.device)

        # Extract RDKit descriptors

        rdkit_features = torch.tensor(np.array([rdkit[1] for rdkit in inputs[0]]), dtype=torch.float32, device=self.device)

        # Prepare labels and weights
        _, labels, weights = super(GCNRdkitModel, self)._prepare_batch(
            ([], labels, weights))

        return (dgl_batch, rdkit_features), labels, weights

