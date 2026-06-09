import torch
import torch.nn as nn
import torchvision.models as tv_models

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        max_vals = logits.max(dim=1, keepdim=True).values

        shifted = logits - max_vals
        log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1))

        log_probs = shifted - log_sum_exp.unsqueeze(1)

        correct_log_probs = log_probs.gather(
            dim=1,
            index=targets.unsqueeze(1)
        ).squeeze(1)

        loss = -correct_log_probs.mean()

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