from deepchem.feat.molecule_featurizers import RDKitDescriptors
from deepchem.feat.peptide_featurizers import PRDKitDescriptors
import deepchem as dc
import pandas as pd
import numpy as np
from rdkit.Chem import PandasTools

loader = dc.data.CSVLoader(
        tasks=[ 'm_0', 'm_1',],
        feature_field='3Smiles',
        id_field='ID',
        featurizer=PRDKitDescriptors()
    )


train_csv ='/mnt/c/Users/schau/Documents/PerpetualMedicine/RamachandranPlots/RamachandranPlots/deepchem_test_dataset.csv'
train_dataset_dir = '/mnt/c/Users/schau/Documents/PerpetualMedicine/RamachandranPlots/RamachandranPlots/deepchem_test101'
train_dataset = loader.create_dataset(train_csv, data_dir=train_dataset_dir)