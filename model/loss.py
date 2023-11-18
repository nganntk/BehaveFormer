import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, logger=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if logger:
            self.logger = logger
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(dim=1).sqrt()
    
    def calc_cosine(self, x1, x2):
        dot_product_sum = (x1*x2).sum(dim=1)
        norm_multiply = (x1.pow(2).sum(dim=1).sqrt()) * (x2.pow(2).sum(dim=1).sqrt())
        return dot_product_sum / norm_multiply
    
    def calc_manhattan(self, x1, x2):
        return (x1-x2).abs().sum(dim=1)
    
    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        
        if (not(anchor.isnan().any()) and not(positive.isnan().any()) and not(negative.isnan().any())):
            if (losses.isnan().any()) and self.logger:
                self.logger.info("losses has NaN")

        return losses.mean()