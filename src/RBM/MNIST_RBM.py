import torch
import torchvision

import math
import numpy as np
import matplotlib.pyplot as plt

import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.utils import make_grid

from RBM import RBM
from DAE import DAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- Load MNIST ----- #
flattenTransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x: torch.flatten(x))])
MNISTTrain = torchvision.datasets.MNIST(root="./dataset/", train=True, transform=flattenTransform, download=True)
MNISTTest = torchvision.datasets.MNIST(root="./dataset/", train=False, transform=flattenTransform, download=True)

batchSize = 128
trainData = DataLoader(MNISTTrain, batch_size=batchSize, shuffle=True)
testData = DataLoader(MNISTTest, batch_size=batchSize, shuffle=True)

# use much lower learning for last gaussian layer to avoid exploding gradient
# use a Gaussian distribution for the last hidden layer to let it take advantage of continuous values
# 784-1000-500-250-3
RMBLayersTrainingInfos = [
    { "hiddenDim": 1000, "numEpochs": 30, "learningRate": 0.1, "displayDim": (28, 28), "useGaussian": False}, 
    { "hiddenDim": 500, "numEpochs": 30, "learningRate": 0.1, "displayDim": (25, 40), "useGaussian": False},
    { "hiddenDim": 250, "numEpochs": 30, "learningRate": 0.1, "displayDim": (25, 20), "useGaussian": False},
    { "hiddenDim": 3, "numEpochs": 60, "learningRate": 0.01, "displayDim": (25, 10), "useGaussian": True}
]

def trainRBM(rbm, dataLoader, numEpochs, learningRate, weightDecay=2e-4):
    loss = torch.nn.MSELoss(reduction='mean')

    for epoch in range(numEpochs):
        trainLoss = 0
        for i, (data, _) in enumerate(dataLoader): # we will use only data
            Vp0 = data.to(device)
            # Vs0 = torch.bernoulli(Vp0)
            # V: Visible | H: Hidden
            # s: sampling | p: probabilities
            # 0: start | k:end

            Vpk, Vsk = rbm.gibbsSampling(Vp0, iterations = 5) #Vs0

            Hp0, _ = rbm.sampleHidden(Vp0) #Vs0
            Hpk, _ = rbm.sampleHidden(Vsk)

            #Vs0
            rbm.contrastiveDivergence(Vp0, Vpk, Hp0, Hpk, learningRate = learningRate, weightDecay = weightDecay, momentumDamping = 0.5 if epoch < 5 else 0.9)

            trainLoss += loss(Vp0, Vpk) # track loss of probabilities

        print(f"epoch {epoch+1}: {trainLoss/len(dataLoader)}")
    return

def genNewDataSet(rbm, device, dataLoader):
    # rederive new data loader based on hidden activations of trained model
    newData = []
    for data, _ in dataLoader:
        Hp, _ = rbm.sampleHidden(data.to(device))
        newData.append(Hp.detach().cpu().numpy())
    newData = np.concatenate(newData)
    fakesLabels = np.zeros((len(newData), 1))
    return DataLoader(TensorDataset(torch.Tensor(newData).to(device), torch.Tensor(fakesLabels).to(device)), batch_size=dataLoader.batch_size, shuffle=True)

def displayOutput(a, b, dim=(28, 28), maxDisplay = 10, title=None, fileName=None):
    inputs = [a, b]
    fig, axs = plt.subplots(1,2)
    for i in range(2):
        viewAsImage = inputs[i].view(inputs[i].shape[0], 1, dim[0], dim[1])
        sideSize = min(int(math.sqrt(len(viewAsImage))), maxDisplay) # use sqrt to compute max squared size length before makeGrid
        img = make_grid(viewAsImage[:sideSize*sideSize].data, nrow = sideSize).detach().cpu().numpy()
        axs[i].imshow(np.transpose(img, (1, 2, 0)))

    if title is not None: fig.suptitle(title)
    if fileName is not None: plt.savefig(fileName)
    plt.show()

# ----- Training RBM ----- #

loss = torch.nn.MSELoss(reduction='mean')
dataLoader = trainData # get initial iteration of new training data
visibleDim = next(iter(dataLoader))[0].shape[1] # set starting visible dim
hiddenDim = None
RBMLayers = [] # trained RBM models

for configs in RMBLayersTrainingInfos:
    # update hidenDim
    hiddenDim = configs["hiddenDim"]
    # create rbm layers
    rbm = RBM(device, visibleDim, hiddenDim, gaussianHiddenDistribution=configs["useGaussian"], useMomentum = True)

    # # display sample output
    # data = next(iter(dataLoader))[0].to(device)
    # reconstructedVp, _ = rbm.reconstruct(data)
    # displayOutput(data, reconstructedVp, configs["displayDim"], title=f'MSE: {loss(data, reconstructedVp).item()}')

    trainRBM(rbm, dataLoader, configs["numEpochs"], configs["learningRate"], weightDecay=2e-4)

    # display sample output
    data = next(iter(dataLoader))[0].to(device)
    reconstructedVp, _ = rbm.reconstruct(data)
    displayOutput(data, reconstructedVp, configs["displayDim"], title=f'MSE: {loss(data, reconstructedVp).item()}')

    RBMLayers.append(rbm)
    dataLoader = genNewDataSet(rbm, device, dataLoader)
    visibleDim = hiddenDim # update new visibleDim for next RBM

# ----- Build & fine-tune autoencoder ----- #

dataLoader = trainData
learningRate = 1e-3
DAE = DAE(RBMLayers).to(device)
loss = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(DAE.parameters(), learningRate)
numEpochs = 50

trackLoss = True
epochsLoss = []

for epoch in range(numEpochs):
    epochLoss = 0
    for batchIdx, (data, _) in enumerate(dataLoader):
        data = data.to(device)

        optimizer.zero_grad() # zero the parameters gradients

        outputs = DAE(data) # forward

        lossValue = loss(data, outputs) # compute loss

        if(trackLoss):  # record loss
            epochLoss += lossValue.item()

        lossValue.backward() # backward
        optimizer.step()

    epochLoss /= len(dataLoader)
    if(trackLoss):  # record loss
        epochsLoss.append(epochLoss)
    
    if epoch % (numEpochs/10) == (numEpochs/10-1):
        print(f"Epoch[{epoch + 1:>4}] Complete: Avg. Loss: {epochLoss:.8f}")
        # displayOutput(data, outputs, (28, 28), title=f'MSE: {loss(data, outputs).item()}')


it = iter(dataLoader)

for i in range(4):
    data, _ = next(it)
    data = data.to(device)
    outputs = DAE(data)
    displayOutput(data, outputs, (28, 28), title=f'MSE: {loss(data, outputs).item()}')
