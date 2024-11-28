import torch


class GeodesicExtensionModel(torch.nn.Module):
    def __init__(self, model, diameter, dist_scaler = 0.1):
        super().__init__()
        self.model = model
        self.diameter = diameter
        self.dist_scaler = dist_scaler
    def forward(self, x, y):
        dist = torch.linalg.norm(y-x)
        rescale = 1+ dist /self.diameter
        return dist +   (self.model(x, y) - self.dist_scaler * dist) / rescale**2
