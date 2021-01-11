import numpy as np
import matplotlib.pyplot as plt

# use torchvision to load MNIST dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from scipy.linalg import eigh # Solve eigen values

from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio

from utils import *

MNISTDataset = datasets.MNIST( root="./dataset/", train=True, transform=transforms.ToTensor(), download=True)

data, targets = MNISTDataset.data.numpy(), MNISTDataset.targets.numpy()

p = np.random.permutation(len(data)) # shuffle the same way using permutation
data, targets = data[p], targets[p]

print(f"data{data.shape} dtype: {data.dtype}")
print(f"targets{targets.shape} dtype: {targets.dtype}")

# displayRowComparaison(data[np.newaxis, :8], labels = targets[:8].astype('U'))

data = data.reshape(data.shape[0], -1) # reshape as vectors

# Data-preprocessing: Standardizing the data
# It is mandatory before applying PCA to convert mean = 0 and standard deviation = 1 in order to explain well our data
standardizedData, mean, std = standardizing(data)

print(f"standardizedData{standardizedData.shape} dtype: {standardizedData.dtype}")

# Compute covariance matrix
cm = (standardizedData.T @ standardizedData) / (standardizedData.shape[0]-1)
# cm = np.cov(standardizedData, rowvar=False)

values, vectors = eigh(cm) # finding eigen-values and corresponding eigen-vectors 

cumulativeExplainedVariance = np.cumsum(values[::-1] / np.sum(values))

p = 0.99
cutIndex = np.argwhere(cumulativeExplainedVariance > p).min()

fig, ax = plt.subplots()
ax.plot(cumulativeExplainedVariance)
ax.hlines(p, 0, cutIndex, linestyles ="dotted", colors ="b")
ax.vlines(cutIndex, 0, p, linestyles ="dotted", colors ="b")
ax.set_ylabel('Cumulative explained variance')
ax.set_xlabel('Number of components')
ax.set_title(f'The first {cutIndex} main axes explain {p*100}% of the data variance.')
plt.show()

# subsample
samplesNum = 20
subStandardizeData = standardizedData[:samplesNum]
subTargets = targets[:samplesNum]
subData = np.clip(subStandardizeData * std + mean, 0, 255).astype(np.uint8).reshape(-1, 28, 28)

annotationsByP = []
backProjectedByP = []
dataDescriptionsByP = []
percentages = [0.50, 0.70, 0.85, 0.90, 0.99]

for p in percentages:
    cutIndex = np.argwhere(cumulativeExplainedVariance > p).min()
    retainedComponents = vectors[:, -cutIndex:].T

    projected = subStandardizeData @ retainedComponents.T
    backProjected = projected @ retainedComponents

    backProjected = np.clip(backProjected * std + mean, 0, 255).astype(np.uint8).reshape(-1, 28, 28)

    psnr = [round(peak_signal_noise_ratio(a, b), 2) for a, b in zip(subData, backProjected)]
    mse = [round(mean_squared_error(a, b), 2) for a, b in zip(subData, backProjected)]
    ssim = [round(structural_similarity(a, b), 2) for a, b in zip(subData, backProjected)]

    backProjectedByP.append(backProjected)
    annotationsByP.append([f'{p}\n{m}\n{s}' for p, m, s in zip(psnr, mse, ssim)])
    dataDescriptionsByP.append(f'{p*100}%({cutIndex})')

    # display each sub reconstruction
    # annotations = [f'psnr: {p}\nmse: {m}\nssim: {s}' for p, m, s in zip(psnr, mse, ssim)]
    # displayRowComparaison(np.stack((subData, backProjected)), ['data', 'backProjected'],  subTargets.astype('U'), annotations)

# add ground truth data
backProjectedByP.append(subData)
annotationsByP.append(['' for _ in range(len(subTargets))])
dataDescriptionsByP.append('data')

displayRowComparaison(np.stack(backProjectedByP), dataDescriptionsByP,  subTargets.astype('U'), np.array(annotationsByP), byRowsAnnotations=True, vAxesPad=0.8)
