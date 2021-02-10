import torch

class RBM():
    def __init__(self, device, visibleDim, hiddenDim, gaussianHiddenDistribution=False, useMomentum = True):

        self.device = device
        self.visibleDim = visibleDim
        self.hiddenDim = hiddenDim
        self.gaussianHiddenDistribution = gaussianHiddenDistribution

        # intialize parameters
        self.W = torch.randn(visibleDim, hiddenDim).to(self.device) * 0.1
        self.hBias = torch.zeros(hiddenDim).to(self.device)
        self.vBias = torch.zeros(visibleDim).to(self.device)

        self.useMomentum = useMomentum
        if self.useMomentum:  # parameters for learning with momentum
            self.WMomentum = torch.zeros(visibleDim, hiddenDim).to(self.device)
            self.hBiasMomentum = torch.zeros(hiddenDim).to(self.device)
            self.vBiasMomentum = torch.zeros(visibleDim).to(self.device)

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
            _, Hs = self.sampleHidden(Vs)
            Vp, Vs = self.sampleVisible(Hs)  
        return Vp, Vs

    def reconstruct(self, v):
        _, Hs = self.sampleHidden(v)
        return self.sampleVisible(Hs)

    def contrastiveDivergence(self, Vs0, Vsk, Hp0, Hpk, learningRate, weightDecay=0, momentumDamping = 0.5):

        if(self.useMomentum):
            # Damping
            self.WMomentum *= momentumDamping
            self.hBiasMomentum *= momentumDamping
            self.vBiasMomentum *= momentumDamping

            self.WMomentum += (torch.mm(Vsk.t(), Hpk) - torch.mm(Vs0.t(), Hp0)) / len(Vs0)
            self.hBiasMomentum += torch.mean((Hpk - Hp0), axis=0)
            self.vBiasMomentum += torch.mean((Vsk - Vs0), axis=0)

            self.W -= learningRate * self.WMomentum
            self.hBias -= learningRate * self.hBiasMomentum
            self.vBias -= learningRate * self.vBiasMomentum

        else:
            #"/ len(Vs0)"" for batch size nomalisation
            self.W -= learningRate * (torch.mm(Vsk.t(), Hpk) - torch.mm(Vs0.t(), Hp0)) / len(Vs0)
            self.hBias -= learningRate * torch.mean((Hpk - Hp0), axis=0)
            self.vBias -= learningRate * torch.mean((Vsk - Vs0), axis=0)

        self.W *= (1 - weightDecay) # L2 weight decay
        return