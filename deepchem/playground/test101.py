import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
#import mis_scripts
import deepchem as dc
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns


train_dataset_disk = dc.data.DiskDataset('dc_datasets/train_rdkit_cicfin_pampa/') 
test_dataset_disk = dc.data.DiskDataset('dc_datasets/test_rdkit_cicfin_pampa/')  

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,2, figsize=(10, 10))

# Perform 10 different train-test splits
for i in range(2):
    splitter = dc.splits.RandomSplitter()
    # Splitting dataset into train and test datasets

    train_dataset_disk 
    train_dataset = train_dataset_disk
    test_dataset = test_dataset_disk
    # Define the pipeline
    
    # Define the 5-fold cross-validation
    # not possible in DC out of the box. have to implement myself
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform Randomized Search CV on the training set
    # Define a model
    dc_model = dc.models.MultitaskRegressor(
    n_tasks=1,
    n_features=210,
    layer_sizes=[100,100,50,20],  # Two layers as requested
    dropouts=0.2,
    learning_rate=0.0001,
    batch_size=20,
    model_dir=f"rdkit_model3_{i}"
)
    for _ in range(10):
        dc_model.fit(train_dataset, nb_epoch = 50)
        # Evaluate the model on the testing set
        y_pred = dc_model.predict(test_dataset)
        #y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]
        y_pred_train = dc_model.predict(train_dataset)
    
        metrics = [
            dc.metrics.Metric(dc.metrics.r2_score),
            dc.metrics.Metric(dc.metrics.mean_absolute_error),
            dc.metrics.Metric(dc.metrics.mean_squared_error)
        ]
        
        # Evaluate on training and test sets
        train_scores = dc_model.evaluate(train_dataset, metrics)
        test_scores = dc_model.evaluate(test_dataset, metrics)
        
        # # Print the results
        # print("Train Scores:")
        # print("R²:", train_scores['r2_score'])
        # print("MAE:", train_scores['mean_absolute_error'])
        # print("MSE:", train_scores['mean_squared_error'])
        
        print("\nTest Scores:")
        print("R²:", test_scores['r2_score'])
        print("MAE:", test_scores['mean_absolute_error'])
        print("MSE:", test_scores['mean_squared_error'])

    #all_metrics.append(metrics)
    # Create a scatter plot with a regression line for each split
    sns.scatterplot(x=train_dataset.y.flatten(), y=y_pred_train.flatten(), ax=axs[i,0])
    sns.regplot(x=train_dataset.y.flatten(), y=y_pred_train.flatten(), ax=axs[i,0], scatter=False, color='r')
    axs[i,0].set_title(f"Split {i+1} Predicted vs Actual Training Set")
    axs[i,0].set_xlabel('Actual Values')
    axs[i,0].set_ylabel('Predicted Values')
    # Create a scatter plot with a regression line for each split
    sns.scatterplot(x=test_dataset.y.flatten(), y=y_pred.flatten(), ax=axs[i,1])
    sns.regplot(x=test_dataset.y.flatten(), y=y_pred.flatten(), ax=axs[i,1], scatter=False, color='b')
    axs[i,1].set_title(f"Split {i+1} Predicted vs Actual Test Set")
    axs[i,1].set_xlabel('Actual Values')
    axs[i,1].set_ylabel('Predicted Values')

# Adjust layout
plt.tight_layout()
plt.show()
