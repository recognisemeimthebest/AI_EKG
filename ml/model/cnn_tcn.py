"""
CNN-TCN 부정맥 3-class 분류 모델

구조:
  Waveform (5000,12) → CNN Block → TCN Block → GAP ─┐
  Numeric (4) + Patient (2) → FC ───────────────────┤→ Concat → FC → 3-class
"""
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """1D CNN으로 ECG 파형에서 형태학적 특징 추출"""

    def __init__(self, in_channels=12, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            # Block 1: (B, 12, 5000) → (B, 32, 2500)
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            # Block 2: (B, 32, 2500) → (B, 64, 625)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            # Block 3: (B, 64, 625) → (B, 64, 156)
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


class TCNLayer(nn.Module):
    """단일 TCN 레이어: Dilated Causal Conv + Residual"""

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # same padding
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        return out + residual  # Residual connection


class TCNBlock(nn.Module):
    """TCN 블록: 여러 dilation rate로 다중 시간 스케일 포착"""

    def __init__(self, channels=64, n_layers=4, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i  # 1, 2, 4, 8
            layers.append(TCNLayer(channels, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CNNTCN(nn.Module):
    """
    CNN-TCN 부정맥 분류 모델

    Args:
        n_leads: ECG 리드 수 (기본 12)
        n_numeric: 수치 피처 수 (기본 4: rr_interval, qrs_duration, p_duration, qrs_axis)
        n_patient: 환자 피처 수 (기본 2: age, gender)
        n_classes: 분류 클래스 수 (기본 3)
        dropout: 드롭아웃 비율 (기본 0.3)
    """

    def __init__(self, n_leads=12, n_numeric=6, n_patient=2,
                 n_classes=3, dropout=0.3):
        super().__init__()

        # 파형 처리 경로
        self.cnn = CNNBlock(in_channels=n_leads, dropout=dropout)
        self.tcn = TCNBlock(channels=64, n_layers=4, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling

        # 수치+환자 피처 처리 경로
        n_aux = n_numeric + n_patient
        self.aux_fc = nn.Sequential(
            nn.Linear(n_aux, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, waveform, numeric, patient):
        """
        Args:
            waveform: (B, 5000, 12) → 내부에서 (B, 12, 5000)으로 변환
            numeric:  (B, 4)
            patient:  (B, 2)
        Returns:
            logits: (B, 3)
        """
        # 파형: (B, 5000, 12) → (B, 12, 5000) for Conv1d
        x = waveform.transpose(1, 2)
        x = self.cnn(x)       # (B, 64, 156)
        x = self.tcn(x)       # (B, 64, 156)
        x = self.gap(x)       # (B, 64, 1)
        x = x.squeeze(-1)     # (B, 64)

        # 보조 피처
        aux = torch.cat([numeric, patient], dim=1)  # (B, 6)
        aux = self.aux_fc(aux)                       # (B, 16)

        # Concat + 분류
        combined = torch.cat([x, aux], dim=1)  # (B, 80)
        logits = self.classifier(combined)      # (B, 3)
        return logits


if __name__ == "__main__":
    # 모델 확인
    model = CNNTCN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {n_params:,}")
    print(f"모델 크기 (FP32): {n_params * 4 / 1024 / 1024:.1f} MB")

    # 더미 입력으로 forward 테스트
    waveform = torch.randn(4, 5000, 12)
    numeric = torch.randn(4, 4)
    patient = torch.randn(4, 2)
    logits = model(waveform, numeric, patient)
    print(f"입력: waveform={waveform.shape}, numeric={numeric.shape}, patient={patient.shape}")
    print(f"출력: {logits.shape}")
    print(f"출력 예시: {logits[0].detach()}")
