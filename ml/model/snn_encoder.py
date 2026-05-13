"""
SNN(Siamese/Supervised Contrastive) 용 ECG CNN Encoder.

입력:  (B, T, 3)     # T샘플 × 500Hz × 3-lead (I, II, III=II-I). GAP 덕에 T 유연.
출력:  (B, 128)       # L2 정규화된 임베딩

프로토타입 단계이므로 cnn_tcn.CNNBlock 스타일을 따르되 단순화함.
임베딩은 contrastive 학습 안정성을 위해 L2 정규화하여 hypersphere 위에 둔다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGEncoder(nn.Module):
    """ECG 3-lead → 128차원 임베딩. AdaptiveAvgPool로 입력 길이 자유."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()

        # (B, 3, 2500) → (B, 32, 1250)
        # → (B, 64, 312) → (B, 128, 78) → (B, 128, 19)
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        # Projection head — SimCLR/SupCon 관행: backbone feature를 별도 head로 임베딩
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T, C) 형태. 내부에서 (B, C, T)로 변환.
        Returns:
            embed: (B, embed_dim), L2-normalized.
        """
        if waveform.dim() != 3:
            raise ValueError(f"waveform must be 3D (B,T,C), got {waveform.shape}")
        x = waveform.transpose(1, 2)        # (B, C, T)
        x = self.backbone(x)                # (B, 128, T')
        x = self.gap(x).squeeze(-1)         # (B, 128)
        z = self.projection(x)              # (B, embed_dim)
        z = F.normalize(z, dim=1, p=2)      # 단위 구면으로 투영
        return z


if __name__ == "__main__":
    model = ECGEncoder()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터: {n_params:,}  ({n_params * 4 / 1024 / 1024:.2f} MB FP32)")
    dummy = torch.randn(8, 2500, 3)
    out = model(dummy)
    print(f"입력 {tuple(dummy.shape)} → 출력 {tuple(out.shape)}")
    print(f"L2 norm sample: {out.norm(dim=1)[:3].tolist()}  (≈1.0 이어야 함)")
