"""UDA-DDA transfer network adapted for quasi-online calibration."""

from __future__ import annotations

import torch.nn as nn

import backbone


class Transfer_Net(nn.Module):
    """Feature extractor plus classifier from the original UDA-DDA codebase."""

    def __init__(
        self,
        num_class=3,
        base_net="CFE",
        transfer_loss="mmd",
        use_bottleneck=False,
        width=32,
        confidence_threshold=0.35,
    ):
        super().__init__()
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        self.confidence_threshold = confidence_threshold
        self.feature_dim = 64
        self.num_class = num_class
        self.base_net = base_net
        self.width = width
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_class),
        )

    def extract_features(self, x):
        return self.base_network(x)

    def forward(self, x, return_features=False):
        features = self.extract_features(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def predict(self, x):
        return self.forward(x)

    def model_config(self):
        return {
            "num_class": self.num_class,
            "base_net": self.base_net,
            "transfer_loss": self.transfer_loss,
            "use_bottleneck": self.use_bottleneck,
            "width": self.width,
            "confidence_threshold": self.confidence_threshold,
        }
