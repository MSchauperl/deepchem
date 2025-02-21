from deepchem.models import GCNModelAngle
import deepchem as dc
import numpy as np

test_dataset = dc.data.DiskDataset(data_dir = '/mnt/c/Users/schau/Documents/PerpetualMedicine/RamachandranResults/ram_data/random_split_1/test_dataset_molgraphconvfeaturizer')

# 3. Define Model
model = GCNModelAngle(
    n_tasks=test_dataset.y.shape[1],  # Number of bins
    mode='regression', 
    graph_conv_layers =[64,64,128],
    predictor_hidden_feats = 256,
    predictor_dropout = 0.2
    # Change to 'classification' for probabilities
)
avg_kld = dc.metrics.Metric(dc.metrics.kl_divergence, np.sum)

# 4. Train Model
model.fit(test_dataset, nb_epoch=1)
model.evaluate(test_dataset, metrics = [avg_kld],)
# # 5. Evaluate Model
predicted = model.predict(test_dataset)
print("Predicted angle distributions:", predicted)