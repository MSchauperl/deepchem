import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import seaborn as sns
import deepchem as dc
from sklearn.ensemble import RandomForestClassifier


from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
train_dataset = dc.data.DiskDataset('/mnt/ml_general/perm/datasets/cluster_chem_split_1/train_dataset_circularfingerprint')
model = dc.models.SklearnModel(RandomForestRegressor(),mode = 'regression')
model.fit(train_dataset)
model.predict(train_dataset)