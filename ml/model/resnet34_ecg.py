"""
ResNet-34 for ECG classification (Tarabanis et al. 2025 방법론)

- 34-layer ResNet with 16 residual connections
- 1D convolution for ECG signal processing
- Input: (batch, n_leads, 5000) at 500Hz
- Optional: concat tabular features after CNN
"""
import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block (Hu et al. 2018) for 1D signals"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1)
        return x * w


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class SEResidualBlock1D(nn.Module):
    """Residual Block with Squeeze-and-Excitation (Kwon 2024)"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, se_reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels, reduction=se_reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet34ECG(nn.Module):
    """
    ResNet-34 for 1D ECG signals.
    Layer config: [3, 4, 6, 3] = 16 residual blocks = 32 conv layers + 1 initial conv + 1 FC = 34 layers
    """

    def __init__(self, n_leads=3, n_classes=2, dropout=0.3):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Conv1d(n_leads, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual layers: [3, 4, 6, 3] blocks
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, n_classes)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = [ResidualBlock1D(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """ECG -> feature vector (512-dim)"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1)
        return x

    def forward(self, x):
        """x: (batch, n_leads, seq_len)"""
        feat = self.forward_features(x)
        feat = self.dropout(feat)
        return self.fc(feat)


class ResNet34ECGWithTabular(nn.Module):
    """ResNet-34 + tabular features (Late Fusion)"""

    def __init__(self, n_leads=3, n_numeric=4, n_patient=2, n_classes=2, dropout=0.3):
        super().__init__()
        self.resnet = ResNet34ECG(n_leads=n_leads, n_classes=n_classes, dropout=dropout)
        # Remove the original FC
        self.resnet.fc = nn.Identity()

        n_tabular = n_numeric + n_patient
        self.tabular_fc = nn.Sequential(
            nn.Linear(n_tabular, 32), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, waveform, numeric, patient):
        """
        waveform: (batch, seq_len, n_leads) -> transpose to (batch, n_leads, seq_len)
        numeric: (batch, n_numeric)
        patient: (batch, n_patient)
        """
        x = waveform.transpose(1, 2)  # (B, leads, 5000)
        ecg_feat = self.resnet.forward_features(x)  # (B, 512)
        ecg_feat = self.resnet.dropout(ecg_feat)

        tab = torch.cat([numeric, patient], dim=1)
        tab_feat = self.tabular_fc(tab)  # (B, 32)

        combined = torch.cat([ecg_feat, tab_feat], dim=1)  # (B, 544)
        return self.classifier(combined)


class SEResNet34ECG(nn.Module):
    """SE-ResNet-34 for 1D ECG signals (Kwon 2024 style)"""

    def __init__(self, n_leads=3, n_classes=2, dropout=0.3, se_reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(n_leads, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1, se_reduction=se_reduction)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2, se_reduction=se_reduction)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2, se_reduction=se_reduction)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2, se_reduction=se_reduction)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, n_classes)
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride, se_reduction):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = [SEResidualBlock1D(in_channels, out_channels, stride, downsample, se_reduction)]
        for _ in range(1, blocks):
            layers.append(SEResidualBlock1D(out_channels, out_channels, se_reduction=se_reduction))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        feat = self.dropout(feat)
        return self.fc(feat)


class SEResNet34ECGWithTabular(nn.Module):
    """SE-ResNet-34 + tabular features (Late Fusion)"""

    def __init__(self, n_leads=3, n_numeric=4, n_patient=2, n_classes=2, dropout=0.3):
        super().__init__()
        self.resnet = SEResNet34ECG(n_leads=n_leads, n_classes=n_classes, dropout=dropout)
        self.resnet.fc = nn.Identity()

        n_tabular = n_numeric + n_patient
        self.tabular_fc = nn.Sequential(
            nn.Linear(n_tabular, 32), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, waveform, numeric, patient):
        x = waveform.transpose(1, 2)
        ecg_feat = self.resnet.forward_features(x)
        ecg_feat = self.resnet.dropout(ecg_feat)
        tab = torch.cat([numeric, patient], dim=1)
        tab_feat = self.tabular_fc(tab)
        combined = torch.cat([ecg_feat, tab_feat], dim=1)
        return self.classifier(combined)
