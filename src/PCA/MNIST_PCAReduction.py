import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

# Compute covariance matrix
# cm = (data.T @ data) / (data.shape[0]-1)
cm = np.cov(data, rowvar=False)

values, vectors = eigh(cm) # finding eigen-values and corresponding eigen-vectors 

fig = go.Figure(data=[go.Scatter(x=np.arange(0, len(values)), y=values[::-1], mode='lines', marker=dict(size=8,opacity=0.5), name="data" )])
fig.update_layout( xaxis_title="components", yaxis_title="eigen values")
fig.show()

cumulativeExplainedVariance = np.cumsum(values[::-1] / np.sum(values))

p = 0.99
cutIndex = np.argwhere(cumulativeExplainedVariance > p).min()

fig = go.Figure(data=[go.Scatter(x=np.arange(0, len(cumulativeExplainedVariance)), y=cumulativeExplainedVariance, mode='lines', marker=dict(size=8,opacity=0.5))])
fig.add_shape(type='line', x0=0, y0=p, x1=cutIndex, y1=p, line=dict(color='Red', dash='dot',), xref='x', yref='y')
fig.add_shape(type='line', x0=cutIndex, y0=0, x1=cutIndex, y1=p, line=dict(color='Red', dash='dot',), xref='x', yref='y')
fig.update_layout(title=f'The first {cutIndex} main components explain {p*100}% of the data variance.', xaxis_title="components", yaxis_title="Cumulative explained variance")
fig.show()

# subsample
samplesNum = 20
subData = data[:samplesNum]
subTargets = targets[:samplesNum]
subDataAsImg = subData.astype(np.uint8).reshape(-1, 28, 28)

annotationsByP = []
backProjectedByP = []
dataDescriptionsByP = []
percentages = [0.50, 0.70, 0.85, 0.90, 0.99]

for p in percentages:
    cutIndex = np.argwhere(cumulativeExplainedVariance > p).min()
    retainedComponents = vectors[:, -cutIndex:].T

    projected = subData @ retainedComponents.T
    backProjected = projected @ retainedComponents

    backProjected = np.clip(backProjected, 0, 255).astype(np.uint8).reshape(-1, 28, 28)
    
    psnr = [round(peak_signal_noise_ratio(a, b), 2) for a, b in zip(subDataAsImg, backProjected)]
    mse = [round(mean_squared_error(a, b), 2) for a, b in zip(subDataAsImg, backProjected)]
    ssim = [round(structural_similarity(a, b), 2) for a, b in zip(subDataAsImg, backProjected)]

    backProjectedByP.append(backProjected)
    annotationsByP.append([f'{p}\n{m}\n{s}' for p, m, s in zip(psnr, mse, ssim)])
    dataDescriptionsByP.append(f'{p*100}%({cutIndex})')

    # display each sub reconstruction
    # annotations = [f'psnr: {p}\nmse: {m}\nssim: {s}' for p, m, s in zip(psnr, mse, ssim)]
    # displayRowComparaison(np.stack((subData, backProjected)), ['data', 'backProjected'],  subTargets.astype('U'), annotations)

# add ground truth data
backProjectedByP.append(subDataAsImg)
annotationsByP.append(['' for _ in range(len(subTargets))])
dataDescriptionsByP.append('data')

displayRowComparaison(np.stack(backProjectedByP), dataDescriptionsByP,  subTargets.astype('U'), np.array(annotationsByP), byRowsAnnotations=True, vAxesPad=0.8)
