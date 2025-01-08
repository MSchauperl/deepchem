from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.graph_data import GraphData
from typing import Tuple
import numpy as np
from deepchem.feat.molecule_featurizers import RDKitDescriptors, MolGraphConvFeaturizer


class MolGraphConvRdkitFeaturizer(MolecularFeaturizer):
    """
    A combined featurizer that generates both graph features (GraphData)
    and RDKit descriptors (numpy.ndarray) for a molecule.
    """

    def __init__(self,
                 use_edges: bool = False,
                 use_chirality: bool = False,
                 use_partial_charge: bool = False,
                 descriptors: list = [],
                 is_normalized: bool = False):
        """
        Parameters
        ----------
        use_edges: bool, default False
            Whether to use edge features for the graph featurizer.
        use_chirality: bool, default False
            Whether to include chirality information in the graph features.
        use_partial_charge: bool, default False
            Whether to use partial charges in the graph features.
        descriptors: list of str, optional
            A list of RDKit descriptors to compute. If empty, all available descriptors will be computed.
        is_normalized: bool, default False
            Whether to normalize RDKit descriptor features.
        """
        self.graph_featurizer = MolGraphConvFeaturizer(
            use_edges=use_edges,
            use_chirality=use_chirality,
            use_partial_charge=use_partial_charge
        )
        self.rdkit_featurizer = RDKitDescriptors(
            descriptors=descriptors,
            is_normalized=is_normalized
        )

    def _featurize(self, datapoint) -> Tuple[GraphData, np.ndarray]:
        """
        Featurize a single molecule into a tuple of GraphData and RDKit descriptors.

        Parameters
        ----------
        datapoint: RDKitMol
            RDKit molecule object to featurize.

        Returns
        -------
        Tuple[GraphData, np.ndarray]
            A tuple containing:
            - GraphData: Graph representation of the molecule.
            - np.ndarray: RDKit descriptors.
        """
        # Featurize graph features
        graph_data = self.graph_featurizer._featurize(datapoint)

        # Featurize RDKit descriptors
        rdkit_data = self.rdkit_featurizer._featurize(datapoint)

        return graph_data, rdkit_data
