import deepchem as dc
import numpy as np
import pickle

train_dataset = dc.data.DiskDataset('/mnt/c/Users/schau/Documents/PerpetualMedicine/RamachandranPlots/RamachandranPlots/ram_data_group/group_split_1/train_dataset_pcircularfingerprint/')

# Define DeepChem's MultitaskRegressor
model = dc.models.ProbabilityRegressor(
    n_tasks=36*36,  # Number of output properties
    n_features=train_dataset.X.shape[1],  # Input feature dimension
    layer_sizes=[512, 256, 128],  # Hidden layer sizes
    dropouts=0.3,
    learning_rate=0.00001,
    batch_size=32,
    mode="regression"
    
)
sum_kld = dc.metrics.Metric(dc.metrics.kl_divergence, np.sum)

# Train the model
for i in range(50):
    print(model.fit(train_dataset, nb_epoch=1))
    print(model.evaluate(train_dataset, metrics = [sum_kld],))
    predicted = model.predict(train_dataset)

    with open(f"/mnt/c/Users/schau/Documents/PerpetualMedicine/RamachandranPlots/RamachandranPlots/tmp/prediction_{i}.pkl", "wb") as f:
        pickle.dump(predicted, f)

# # 5. Evaluate Model

print("Predicted angle distributions:", predicted)