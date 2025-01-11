import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import seaborn as sns
import deepchem as dc


# loader = dc.data.CSVLoader(
#                 tasks=['NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
#        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'], feature_field ="smiles",id_field='ID', featurizer = dc.feat.CircularFingerprint())
# dataset = loader.create_dataset('../data/tox21/tox_train_dataset.csv', data_dir = '../datasets/tox21_cicfin_train/')

dataset = dc.data.DiskDataset('/home/jupyter/programs/ML_algorithms/tests/perm_test_data/group_split_1/test_dataset_molgraphconvrdkitfeaturizer/')

n_tasks = 3
model  = dc.models.GCNRdkitModel(
            n_tasks=n_tasks,mode = 'classification')

model.fit(dataset)