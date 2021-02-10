import torch
import torch.nn as nn
import torch.nn.functional as F

class DAE(nn.Module):

    def __init__(self, models):
        super(DAE, self).__init__()

        # extract weights from each model
        encoders = []
        encoderBiases = []
        decoders = []
        decoderBiases = []
        for model in models:
            encoders.append(nn.Parameter(model.W.clone()))
            encoderBiases.append(nn.Parameter(model.hBias.clone()))
            decoders.append(nn.Parameter(model.W.clone()))
            decoderBiases.append(nn.Parameter(model.vBias.clone()))

        # build encoders and decoders
        self.encoders = nn.ParameterList(encoders)
        self.encoderBiases = nn.ParameterList(encoderBiases)
        self.decoders = nn.ParameterList(reversed(decoders))
        self.decoderBiases = nn.ParameterList(reversed(decoderBiases))

    def forward(self, v):
        return self.decode(self.encode(v)) # decode

    def encode(self, v):  # for visualization, encode without sigmoid
        Vp = v
        activation = v
        for i in range(len(self.encoders)):
            W = self.encoders[i]
            hBias = self.encoderBiases[i]
            activation = torch.mm(Vp, W) + hBias
            Vp = torch.sigmoid(activation)

        # for the last layer, we want to return the activation directly rather than the sigmoid
        return activation

    def decode(self, h):
        Hp = h
        for i in range(len(self.encoders)):
            W = self.decoders[i]
            vBias = self.decoderBiases[i]
            activation = torch.mm(Hp, W.t()) + vBias
            Hp = torch.sigmoid(activation)
        return Hp

class Naive_DAE(nn.Module):
    def __init__(self, layers):
        super(Naive_DAE, self).__init__()

        self.layers = layers
        encoders = []
        decoders = []
        prevLayer = layers[0]
        for layer in layers[1:]:
            encoders.append(nn.Linear(in_features=prevLayer, out_features=layer))
            decoders.append(nn.Linear(in_features=layer, out_features=prevLayer))
            prevLayer = layer
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        for i, encoder in enumerate(self.encoders):
            x = enc(x)
            if i != len(self.encoders) - 1:
                x = torch.sigmoid(x)
        return x
    
    def decode(self, x):
        for decoder in self.decoders:
            x = torch.sigmoid(decoder(x))
        return x