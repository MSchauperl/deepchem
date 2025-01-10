import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import seaborn as sns
import deepchem as dc
from sklearn.ensemble import RandomForestClassifier


# loader = dc.data.CSVLoader(
#                 tasks=['NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
#        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'], feature_field ="smiles",id_field='ID', featurizer = dc.feat.CircularFingerprint())
# dataset = loader.create_dataset('../data/tox21/tox_train_dataset.csv', data_dir = '../datasets/tox21_cicfin_train/')

dataset = dc.data.DiskDataset('/home/jupyter/programs/ML_algorithms/datasets/tox21_train_group/cluster_chem_split_1/train_dataset_circularfingerprint')
sklearn_model = RandomForestClassifier(n_estimators=100, random_state=42)
dc_model = dc.models.SklearnModel(sklearn_model, mode='classification')
dc_model.fit(dataset)
test = dc_model.predict(dataset)
print(test)