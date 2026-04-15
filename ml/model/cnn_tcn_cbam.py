"""
CNN-TCN-CBAM 부정맥 3-class 분류 모델

구조:
  Waveform (5000,12) → CNN Block → TCN Block → CBAM → GAP ─┐
  Numeric (4) + Patient (2) → FC ──────────────────────────┤→ Concat → FC → 3-class

CBAM: Channel Attention + Temporal Attention
  - Channel: 어떤 피처맵(리드 특성)이 중요한지
  - Temporal: 10초 중 어느 구간이 중요한지
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
        return out + residual


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


class ChannelAttention(nn.Module):
    """
    채널 어텐션: 어떤 피처맵(채널)이 중요한지 학습
    GAP + GMP → FC → Sigmoid로 채널별 가중치 생성
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.shared_fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
        )

    def forward(self, x):
        # x: (B, C, T)
        avg_pool = x.mean(dim=2)  # (B, C)
        max_pool = x.amax(dim=2)  # (B, C)
        attn = torch.sigmoid(self.shared_fc(avg_pool) + self.shared_fc(max_pool))
        return x * attn.unsqueeze(2)  # (B, C, T)


class TemporalAttention(nn.Module):
    """
    시간축 어텐션: 10초 중 어느 구간이 중요한지 학습
    채널 방향 avg/max pool → Conv1d → Sigmoid로 시간별 가중치 생성
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x):
        # x: (B, C, T)
        avg_pool = x.mean(dim=1, keepdim=True)  # (B, 1, T)
        max_pool = x.amax(dim=1, keepdim=True)  # (B, 1, T)
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, T)
        attn = torch.sigmoid(self.conv(pooled))  # (B, 1, T)
        return x * attn  # (B, C, T)


class CBAM1D(nn.Module):
    """CBAM: Channel Attention → Temporal Attention (순차 적용)"""

    def __init__(self, channels, reduction=4, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.temporal_attn = TemporalAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.temporal_attn(x)
        return x


class CNNTCN_CBAM(nn.Module):
    """
    CNN-TCN-CBAM 부정맥 분류 모델

    Args:
        n_leads: ECG 리드 수 (기본 12)
        n_numeric: 수치 피처 수 (기본 4)
        n_patient: 환자 피처 수 (기본 2)
        n_classes: 분류 클래스 수 (기본 3)
        dropout: 드롭아웃 비율 (기본 0.3)
    """

    def __init__(self, n_leads=12, n_numeric=4, n_patient=2,
                 n_classes=3, dropout=0.3):
        super().__init__()

        # 파형 처리 경로
        self.cnn = CNNBlock(in_channels=n_leads, dropout=dropout)
        self.tcn = TCNBlock(channels=64, n_layers=4, dropout=dropout)
        self.cbam = CBAM1D(channels=64, reduction=4, kernel_size=7)
        self.gap = nn.AdaptiveAvgPool1d(1)

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
        # 파형: (B, 5000, 12) → (B, 12, 5000)
        x = waveform.transpose(1, 2)
        x = self.cnn(x)       # (B, 64, 156)
        x = self.tcn(x)       # (B, 64, 156)
        x = self.cbam(x)      # (B, 64, 156) — 어텐션 적용
        x = self.gap(x)       # (B, 64, 1)
        x = x.squeeze(-1)     # (B, 64)

        # 보조 피처
        aux = torch.cat([numeric, patient], dim=1)
        aux = self.aux_fc(aux)

        # Concat + 분류
        combined = torch.cat([x, aux], dim=1)
        logits = self.classifier(combined)
        return logits


if __name__ == "__main__":
    model = CNNTCN_CBAM()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {n_params:,}")
    print(f"모델 크기 (FP32): {n_params * 4 / 1024 / 1024:.2f} MB")

    # 더미 입력으로 forward 테스트
    waveform = torch.randn(4, 5000, 12)
    numeric = torch.randn(4, 4)
    patient = torch.randn(4, 2)
    logits = model(waveform, numeric, patient)
    print(f"입력: waveform={waveform.shape}, numeric={numeric.shape}, patient={patient.shape}")
    print(f"출력: {logits.shape}")

    # CBAM 파라미터만 확인
    cbam_params = sum(p.numel() for p in model.cbam.parameters())
    print(f"CBAM 추가 파라미터: {cbam_params:,}")
