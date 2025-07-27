import torch
import torch.nn as nn
import torch.nn.functional as F

class UFE(nn.Module):
    def __init__(self, in_chans, n_classes, attn_dropout=0.2, p_drop=0.3):
        super().__init__()
        kernel_sizes = [15, 25, 51, 65]

        # Multi-branch temporal convs with bottlenecks
        self.temp_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 8, (1, k), padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(8),
                nn.ELU(),
                nn.Conv2d(8, 12, (1, 1), bias=False),
                nn.BatchNorm2d(12),
                nn.ELU(),
                nn.Dropout2d(p=p_drop)
            )
            for k in kernel_sizes
        ])

        # Depthwise + Pointwise Spatial Filtering
        self.depthwise_conv = nn.Conv2d(
            in_channels=12 * len(kernel_sizes),
            out_channels=12 * len(kernel_sizes),
            kernel_size=(in_chans, 1),
            groups=12 * len(kernel_sizes),  # depthwise
            bias=False
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=12 * len(kernel_sizes),
            out_channels=24,  # output channels
            kernel_size=(1, 1),
            bias=False
        )
        self.bn_sep = nn.BatchNorm2d(24)
        self.drop_sep = nn.Dropout2d(p=p_drop)

        # Positional encodings for attention
        self.pos_enc = nn.Parameter(torch.randn(1, 256, 24))  # assumes T <= 256

        self.attention = nn.MultiheadAttention(
            embed_dim=24, num_heads=3, dropout=attn_dropout, batch_first=True
        )
        self.attn_ln = nn.LayerNorm(24)
        self.attn_dropout = nn.Dropout(p=p_drop)

        # Conv encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(24),
            nn.ELU(),
            nn.Conv1d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(24),
            nn.ELU(),
            nn.Dropout(p=p_drop)
        )

        # Projection to 32
        self.fc_project = nn.Sequential(
            nn.Linear(24, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(p=p_drop)
        )

        # Classifier
        self.fc_classify = nn.Sequential(
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(p=p_drop),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        B, C, T = x.shape
        x = x.unsqueeze(1)  # (B, 1, C, T)

        # Temporal convs
        temp_feats = [conv(x) for conv in self.temp_convs]  # list of (B, 12, C, T)
        x = torch.cat(temp_feats, dim=1)  # (B, 48, C, T)

        # Depthwise Separable Spatial conv
        x = self.depthwise_conv(x)  # (B, 48, 1, T)
        x = self.pointwise_conv(x)  # (B, 24, 1, T)
        x = self.bn_sep(x)
        x = F.elu(x)
        x = self.drop_sep(x)

        # Pooling
        avg = nn.AdaptiveAvgPool2d((1, T))(x)
        var = nn.AdaptiveMaxPool2d((1, T))(x)
        x = avg + var  # (B, 24, 1, T)

        # Attention
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T, 24)
        if x.shape[1] < self.pos_enc.shape[1]:
            pe = self.pos_enc[:, :x.shape[1], :]
            x = x + pe
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_ln(x + self.attn_dropout(attn_out))

        # Conv encoder
        x = x.permute(0, 2, 1)  # (B, 24, T)
        x = self.encoder(x)  # (B, 24, T)

        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, 24)

        # Project to 32-D
        x = self.fc_project(x)  # (B, 32)

        out = self.fc_classify(x)  # (B, n_classes)
        return out, x
