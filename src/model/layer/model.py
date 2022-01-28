import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_size, fn, sn, tn, dp):
        super(Attention, self).__init__()
        self.input_size = input_size#202# 229
        self.D1 = fn
        self.D2 = sn
        self.D3 = tn
        self.K = 1
        self.Dropout = dp
        
        
        
        self.feature_extractor = nn.Sequential(
            # first layer 
            nn.Linear(self.input_size, self.D1), 
            nn.ReLU(),
            nn.Dropout(p=self.Dropout ),

            # second layer
            nn.Linear(self.D1, self.D2),
            nn.ReLU(),
            nn.Dropout(p=self.Dropout ),

            # third layer
            nn.Linear(self.D2, self.D3),
            nn.ReLU(),
            nn.Dropout(p=self.Dropout )
        )
        
        self.attention = nn.Sequential(
            nn.Linear(self.D3, self.D3),
            nn.Tanh()
        )

        self.attention2 = nn.Sequential(
            nn.Linear(self.D3, self.D3),
            nn.Sigmoid()
        )
        
        self.last = nn.Sequential(
            nn.Linear(self.D3, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.D3 * self.K, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):

        H = self.feature_extractor(x)
        
        A1 = self.attention(H)
        A2 = self.attention2(H)

        A = torch.mul(A1, A2)
        A = self.last(A)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, H)
        
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        
        return Y_prob, Y_hat, A
    
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        prob, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat, prob
    
    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return neg_log_likelihood, A