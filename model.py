import torch
import torch.nn as nn
import torchvision.models as tv_models

class CustomCrossEntropyLoss(nn.Module):
   
    def __init__(self, epsilon: float = 0.1, num_classes: int = 10):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        
        B, C = logits.shape

        z_max = logits.max(dim=1, keepdim=True).values          
        shifted = logits - z_max                                 
        log_sum_exp = shifted.exp().sum(dim=1, keepdim=True).log()
        log_p = shifted - log_sum_exp                            

        correct_log_p = log_p.gather(1, targets.view(-1, 1)).squeeze(1)  
        ce_loss = -correct_log_p.mean()

        smooth_loss = -log_p.mean()   

        loss = (1.0 - self.epsilon) * ce_loss + self.epsilon * smooth_loss

        return loss

def build_mobilenet(alpha=1.0, num_classes=10):
    model = tv_models.mobilenet_v2(
        weights=None,
        width_mult=alpha,
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features=in_features,
        out_features=num_classes,
        bias=True,
    )

    return model