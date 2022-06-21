import torch
import torch.nn as nn

class MetricLoss(nn.Module):
    def __init__(self, centor, r1, r2):
        super(MetricLoss, self).__init__()
        self.centor = centor.requires_grad_(False)
        self.centor = self.centor.cuda()
        self.r1 = r1
        self.r2 = r2
    
    def forward(self, x, labels):
        batch_size = x.size(0)
        # d = torch.pow(x, 2).sum(dim=1, keepdim=False).clamp(min=1e-12, max=1e+12)
        d = torch.norm(x - self.centor, dim=1).clamp(min=1e-12, max=1e+12)
        # d = torch.norm(x, dim=1).clamp(min=1e-12, max=1e+12)
        
        real_mask = labels == 0          
        d_real = d[real_mask]
        d_real -= self.r1
        real_loss = d_real[d_real > 0].sum() / max(real_mask.size(0), 1) 
        
        fake_mask = labels == 1
        d_fake = d[fake_mask]
        d_fake = self.r2 - d_fake
        fake_loss = d_fake[d_fake > 0].sum() / max(fake_mask.size(0), 1)

        return (fake_loss + real_loss) / batch_size

class L2Loss(nn.Module):
    def __init__(self, norm=False):
        super(L2Loss, self).__init__()
        self.norm = norm
    
    def forward(self, x, y):
        batch_size = x.size(0)
        if self.norm:
            d = torch.norm( 
                x/torch.norm(x, dim=1,keepdim=True).clamp(min=1e-12, max=1e+12) - y/torch.norm(y, dim=1,keepdim=True).clamp(min=1e-12, max=1e+12), dim=1).clamp(min=1e-12, max=1e+12)
        else:
            d = torch.norm(x-y, dim=1).clamp(min=1e-12, max=1e+12)
        loss = d.sum() / batch_size
        return loss

if __name__ == "__main__":
    a = torch.ones((3, 3)) * 2
    Loss = MetricLoss(2048, 1, 15).cuda()
    label = torch.ones(3)-1
    a = a.cuda()
    label = label.cuda()
    loss = Loss(a, label)
    print(loss)
