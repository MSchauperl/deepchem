


import deepchem as dc
# Step 2: Create datasets using different featurizers

csv_file = '/mnt/c/Users/schau/Documents/PerpetualMedicine/RamachandranPlots/RamachandranPlots/ram_data_with_groups_split2.csv'
split_data_dir ='/mnt/c/Users/schau/Documents/PerpetualMedicine/RamachandranPlots/RamachandranPlots/tmp'

featurizer = dc.feat.PRDKitDescriptors()


# Create and save DeepChem datasets
loader = dc.data.CSVLoader(
    tasks=[f'm_{i}' for i in range(36*36)] ,
    feature_field='3Smiles',
    featurizer=featurizer
)
dataset = loader.create_dataset(csv_file, data_dir=split_data_dir)

        