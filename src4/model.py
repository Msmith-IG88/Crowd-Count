import torch
import torch.nn as nn

class DilatedTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, 
                              padding=dilation*(kernel_size-1)//2, 
                              dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        
    def forward(self, x):
        residual = x
        x = self.relu(self.bn(self.conv(x)))
        if self.downsample:
            residual = self.downsample(residual)
        return x + residual

class CrowdTCN(nn.Module):
    def __init__(self, input_size=64):
        super().__init__()
        self.initial_conv = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        
        # Dilated TCN blocks
        self.tcn_blocks = nn.Sequential(
            DilatedTCNBlock(64, 128, dilation=1),
            DilatedTCNBlock(128, 128, dilation=2),
            DilatedTCNBlock(128, 128, dilation=4),
            DilatedTCNBlock(128, 128, dilation=8),
        )
        
        # Multi-scale feature fusion
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128, 128//8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128//8, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.output = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, bins]
        x = x.permute(0, 2, 1)  # [batch, bins, seq_len]
        x = self.initial_conv(x)
        
        # Process through TCN blocks
        features = []
        for block in self.tcn_blocks:
            x = block(x)
            features.append(x)
        
        # Multi-scale fusion
        fused = sum(features)
        
        # Attention weighting
        attn_weights = self.attention(fused)
        weighted = fused * attn_weights
        
        # Output prediction
        residual = self.output(weighted).squeeze(1)  # [batch, seq_len]
        return residual
    
class DopplerCrowdTCN(nn.Module):
    def __init__(self, range_bins=64, doppler_bins=32):
        super().__init__()
        self.input_channels = range_bins * doppler_bins
        
        self.initial_fc = nn.Linear(self.input_channels, 256)
        self.tcn_blocks = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.final_fc = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch, seq_len, range_bins, doppler_bins)
        batch, seq_len, range_bins, doppler_bins = x.shape
        x = x.view(batch, seq_len, -1)  # flatten spatial dims
        x = self.initial_fc(x)  # (batch, seq_len, 256)
        x = x.permute(0, 2, 1)  # (batch, 256, seq_len)
        x = self.tcn_blocks(x)  # (batch, 256, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 256)
        out = self.final_fc(x)  # (batch, seq_len, 1)
        out = out.squeeze(-1)
        return out