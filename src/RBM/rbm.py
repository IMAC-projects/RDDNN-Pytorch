import torch

class RBM():
    def __init__(self, device, visibleDim, hiddenDim, gaussianHiddenDistribution=False):

        self.device = device
        self.visibleDim = visibleDim
        self.hiddenDim = hiddenDim
        self.gaussianHiddenDistribution = gaussianHiddenDistribution

        # intialize parameters
        self.W = torch.randn(visibleDim, hiddenDim).to(self.device) * 0.1
        self.hBias = torch.zeros(hiddenDim).to(self.device)
        self.vBias = torch.zeros(visibleDim).to(self.device)

    def sampleHidden(self, v):
        activation = v.mm(self.W) + self.hBias
        if self.gaussianHiddenDistribution:
            return activation, torch.normal(activation, torch.tensor([1]).to(self.device))
        else:
            p = torch.sigmoid(activation)
            return p, torch.bernoulli(p)

    def sampleVisible(self, h):
        activation = h.mm(self.W.t()) + self.vBias
        p = torch.sigmoid(activation)
        return p, torch.bernoulli(p)

    def gibbsSampling(self, v, iterations = 1):
        Vs = v
        for i in range(iterations):
            Hp, Hs = self.sampleHidden(Vs)
            Vp, Vs = self.sampleVisible(Hs)
            
        return Hp, Hs, Vp, Vs

    def contrastiveDivergence(self, Vs0, Vsk, Hp0, Hpk, lr):
        #"/ len(Vs0)"" for batch size nomalisation
        self.W -= lr * (torch.mm(Vsk.t(), Hpk) - torch.mm(Vs0.t(), Hp0)) / len(Vs0)
        self.hBias -= lr * torch.mean((Hpk - Hp0), axis=0)
        self.vBias -= lr * torch.mean((Vsk - Vs0), axis=0)

        # self.W -= self.W * weight_decay # L2 weight decay
        return