import torch
import dgl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import deepchem as dc
import os

os.chdir('/mnt/c/Users/schau/Documents/PerpetualMedicine/DeepChemModels')

dataset = dc.data.DiskDataset( 'dc_datasets/allperm_dataset_gcn_rdkit/')
splitter = dc.splits.RandomSplitter()
# Splitting dataset into train and test datasets
train_dataset, test_dataset = splitter.train_test_split(dataset)
# Define the pipeline

dc_model = dc.models.GCNRdkitModel(
mode='regression', n_tasks=1,graph_conv_layers = [64,64],
             batch_size=128, learning_rate=0.0001,
)

dc_model.fit(train_dataset, nb_epoch = 3,)