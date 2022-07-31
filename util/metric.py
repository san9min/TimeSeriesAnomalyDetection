import torch

def fbeta_score (y_pred, y_true,beta=1, eps=1e-11):
    beta2 = beta ** 2
    y_pred = torch.FloatTensor(y_pred)
    y_true = torch.FloatTensor(y_true).squeeze()
    
    y_pred = y_pred.float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum()
    precision = true_positive.div(y_pred.sum().add(eps))
    recall = true_positive.div(y_true.sum().add(eps))

    return torch.mean((precision * recall * (1 + beta2)).div(precision.mul(beta2) + recall + eps)).item()

