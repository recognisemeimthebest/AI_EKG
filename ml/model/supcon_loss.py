"""
Supervised Contrastive Loss (SupCon).

Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
https://arxiv.org/abs/2004.11362

핵심 아이디어:
  - 같은 클래스의 모든 샘플 쌍을 양성(positive)으로 간주.
  - softmax-cross-entropy 형태로 양성 vs 배치 내 모든 다른 샘플(음성+양성) 비교.
  - temperature가 작을수록 hard positive/negative에 집중.

본 구현은 single-view(증강 1개) 버전이며, 입력 임베딩은 사전 L2 정규화 가정.
"""
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (single-view).

    Args:
        temperature: 온도 스케일링 (기본 0.1).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D), L2-normalized 권장.
            labels:   (B,)
        Returns:
            scalar loss
        """
        if features.dim() != 2:
            raise ValueError(f"features must be 2D (B,D), got {features.shape}")
        if features.size(0) != labels.size(0):
            raise ValueError("features and labels batch size mismatch")

        device = features.device
        batch_size = features.size(0)

        # (B, B) 유사도 행렬 — 정규화된 임베딩의 dot product = cosine
        sim = torch.matmul(features, features.T) / self.temperature

        # 수치 안정화: 행별 max 차감 (log-sum-exp trick)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - sim_max.detach()

        # 양성 마스크: 같은 라벨이지만 자기 자신 제외
        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask  # 자기 자신은 양성에서 제외
        positive_mask = positive_mask.clamp(min=0)

        # 분모용 마스크: 자기 자신만 제외 (모든 다른 샘플)
        logits_mask = 1.0 - self_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 양성이 0개인 anchor는 제외하여 NaN 방지
        pos_per_anchor = positive_mask.sum(dim=1)
        valid = pos_per_anchor > 0

        if valid.sum() == 0:
            # 배치에 양성 쌍이 전혀 없으면 0 반환 (gradient 없음)
            return torch.zeros([], device=device, requires_grad=True)

        mean_log_prob_pos = (
            (positive_mask * log_prob).sum(dim=1)[valid] / pos_per_anchor[valid]
        )

        loss = -mean_log_prob_pos.mean()
        return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    feats = torch.randn(16, 128)
    feats = torch.nn.functional.normalize(feats, dim=1)
    labels = torch.randint(0, 2, (16,))
    loss_fn = SupConLoss(temperature=0.1)
    loss = loss_fn(feats, labels)
    print(f"loss = {loss.item():.4f}  (랜덤 임베딩 → 약 log(15) ≈ 2.7 부근)")
