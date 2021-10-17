from datasets.pre_processing_datasets import preprocess_datasets as ppc
from create_new_dataset_versions import create_new_version_dataset as cnvd
import numpy as np
from visualize_transformed_datasets import visualize_ver_transf, visualize_series_transf

dataset = 'prison'
prison = ppc('prison').apply_preprocess()
n_s = 10 # number of samples per version
n = prison['train']['n'] # number of points in each series
s = prison['train']['s'] # number of series in dataset

transformations = ['jitter', 'scaling', 'magnitude_warp', 'time_warp']

y = np.tile(np.expand_dims(prison['train']['data'], axis=0), (10, 1, 1))
y_new = np.zeros((n_s, n, s))
transf = []
with open(f'{dataset}_original.npy', 'wb') as f:
    np.save(f, y)
for i in range(1, 7):
    # Create 6 different versions of a dataset, adding cumulatively new transformations
    transfs = np.tile(np.random.choice(transformations, size=32), (10, 1))
    transf.append(transfs)
    for j in range(1, 11):
        # Create 10 samples of each version
        y_new[j-1] = cnvd(y[j-1],
                                 transfs[j-1],
                                 version = i,
                                 sample = j).apply_transformations()
    with open(f'{dataset}_version_{i}_{j}samples.npy', 'wb') as f:
        np.save(f, y_new)
    y = y_new
transf = np.array(transf) # (n_versions, n_samples, n_series) - n_samples are just repeating transformations
print('SUCCESS: Stored 6 transformed versions (each with 10 samples) of the original dataset')

visualize_ver_transf('prison', 1, 6, transf[0, 0, :])

visualize_series_transf('prison', 6, 6, transf[:, 0, :])
