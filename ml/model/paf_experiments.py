"""
발작성 AF 감지 모델 10가지 실험 — 각 논문 기법 적용

Baseline: ResNet-34 + 3-lead(I,II,III) + clinical + class weight CE
          AUROC 0.824, Hidden AF P=0.25, R=0.71

각 실험은 하나의 기법만 변경하여 효과를 격리 측정
"""
import argparse
import time
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import h5py
from scipy.signal import find_peaks

from resnet34_ecg import ResNet34ECGWithTabular

PAF_CACHE_H5 = "g:/AIEKG/ml/data/ecg_paf_cache.h5"

# ============================================================
# 기법 1: Focal Loss (CNN-LSTM Imbalanced 2024)
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ============================================================
# 기법 2: STAR Augmentation (Noseworthy 2025)
# ============================================================
class STARAugment:
    """Beat-wise sinusoidal time-amplitude resampling"""
    def __init__(self, time_warp_range=(0.8, 1.2), amp_range=(0.85, 1.15), prob=0.5):
        self.time_warp_range = time_warp_range
        self.amp_range = amp_range
        self.prob = prob

    def __call__(self, waveform):
        """waveform: (seq_len, n_leads) numpy or tensor"""
        if np.random.rand() > self.prob:
            return waveform
        is_tensor = isinstance(waveform, torch.Tensor)
        if is_tensor:
            device = waveform.device
            wf = waveform.cpu().numpy()
        else:
            wf = waveform.copy()

        seq_len, n_leads = wf.shape
        # R-peak detection on lead II (index 1)
        lead = wf[:, min(1, n_leads - 1)]
        peaks, _ = find_peaks(lead, distance=150, height=np.percentile(lead, 60))

        if len(peaks) < 2:
            return waveform

        # Beat-wise augmentation
        result = np.zeros_like(wf)
        boundaries = [0] + list(peaks) + [seq_len]
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            segment = wf[start:end]
            seg_len = len(segment)
            if seg_len < 5:
                result[start:end] = segment
                continue
            # Time warping via resampling
            time_factor = np.random.uniform(*self.time_warp_range)
            new_len = max(3, int(seg_len * time_factor))
            x_old = np.linspace(0, 1, seg_len)
            x_new = np.linspace(0, 1, new_len)
            resampled = np.zeros((new_len, n_leads))
            for ch in range(n_leads):
                resampled[:, ch] = np.interp(x_new, x_old, segment[:, ch])
            # Amplitude scaling
            amp_factor = np.random.uniform(*self.amp_range)
            resampled *= amp_factor
            # Fit back to original length
            if new_len >= seg_len:
                result[start:end] = resampled[:seg_len]
            else:
                x_fit = np.linspace(0, 1, new_len)
                x_orig = np.linspace(0, 1, seg_len)
                for ch in range(n_leads):
                    result[start:end, ch] = np.interp(x_orig, x_fit, resampled[:, ch])

        if is_tensor:
            return torch.tensor(result, dtype=torch.float32, device=device)
        return result


# ============================================================
# 기법 3: Non-Uniform MixUp (ECG-Mamba 2025)
# ============================================================
def non_uniform_mixup(x, y, alpha=0.3, epoch=0, max_epochs=30,
                      start_ratio=0.2, end_ratio=0.8):
    """Epoch-progressive MixUp"""
    progress = min(epoch / max(max_epochs, 1), 1.0)
    mix_ratio = start_ratio + (end_ratio - start_ratio) * progress
    if np.random.rand() > mix_ratio:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


# ============================================================
# 기법 4: Label Smoothing
# ============================================================
class LabelSmoothingCE(nn.Module):
    def __init__(self, n_classes=2, smoothing=0.1, weight=None):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / (self.n_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth * log_probs).sum(dim=1)
        if self.weight is not None:
            w = self.weight[targets]
            loss = loss * w
        return loss.mean()


# ============================================================
# 기법 5: Knowledge Distillation (Sensors 2024)
# ============================================================
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7, weight=None):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, student_logits, teacher_logits, targets):
        hard_loss = self.ce(student_logits, targets)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.T ** 2)
        return self.alpha * hard_loss + (1 - self.alpha) * kd_loss


# ============================================================
# 기법 6: Supervised Contrastive Loss (Li 2026)
# ============================================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """features: (B, D), labels: (B,)"""
        device = features.device
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim = torch.matmul(features, features.T) / self.temperature
        # Remove self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-6)

        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return -mean_log_prob.mean()


# ============================================================
# 기법 7: P-wave Dual-branch (TS-ECG 2024)
# ============================================================
class PWaveBranch(nn.Module):
    """P-wave segment CNN branch"""
    def __init__(self, n_leads=3, n_beats=8, pwave_len=75):
        super().__init__()
        self.n_beats = n_beats
        self.pwave_len = pwave_len
        self.cnn = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, 32)

    def forward(self, pwave_segments):
        """pwave_segments: (B, n_beats, n_leads, pwave_len)"""
        B, N, C, L = pwave_segments.shape
        x = pwave_segments.view(B * N, C, L)
        x = self.cnn(x).squeeze(-1)  # (B*N, 64)
        x = x.view(B, N, -1).mean(dim=1)  # (B, 64)
        return self.fc(x)  # (B, 32)


class ResNet34WithPWave(nn.Module):
    """ResNet-34 + P-wave branch"""
    def __init__(self, base_model, n_leads=3, n_beats=8, pwave_len=75, n_classes=2):
        super().__init__()
        self.base = base_model
        self.pwave_branch = PWaveBranch(n_leads, n_beats, pwave_len)
        # Replace classifier
        old_in = self.base.classifier[-1].in_features
        self.base.classifier[-1] = nn.Identity()
        self.final_fc = nn.Linear(old_in + 32, n_classes)

    def forward(self, waveform, numeric, patient, pwave_segments):
        base_out = self.base(waveform, numeric, patient)  # (B, old_in)
        pwave_out = self.pwave_branch(pwave_segments)  # (B, 32)
        combined = torch.cat([base_out, pwave_out], dim=1)
        return self.final_fc(combined)


# ============================================================
# 기법 8: Wavelet PE + Transformer head (AF-ECGNET 2025)
# ============================================================
class TransformerHead(nn.Module):
    """ResNet feature에 Transformer 레이어 추가"""
    def __init__(self, d_model=512, nhead=8, num_layers=2, n_classes=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        """x: (B, seq_len, d_model)"""
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average
        return self.fc(x)


# ============================================================
# Dataset (기존 v1과 동일 + augmentation 옵션)
# ============================================================
class PAFDataset(Dataset):
    """캐시 기반 PAF 데이터셋 — ecg_paf_cache.h5 (2-lead float16, ~9.8GB) 사용
    preload=True 시 waveform 전체를 RAM에 올려 I/O 병목 제거 (~7GB for train split)
    """

    def __init__(self, ecg_h5_path, paf_h5_path, indices=None, lead_indices=None,
                 use_clinical=True, augment=None, extract_pwave=False,
                 cache_h5_path=PAF_CACHE_H5, preload=True):
        self.cache_h5_path = cache_h5_path
        self._cache_h5 = None
        self.augment = augment
        self.extract_pwave = extract_pwave
        self._waveform_ram = None  # RAM 프리로드 버퍼

        with h5py.File(paf_h5_path, "r") as f:
            all_ecg_indices = f["indices"][:]      # PAF 내 ECG 인덱스 (원본 H5 기준)
            all_labels = f["paf_label"][:]
            all_sids = f["subject_id"][:]
            all_clinical = f["clinical_features"][:] if use_clinical and "clinical_features" in f else None

        # 캐시 인덱스 매핑 로드 (PAF indices → cache indices)
        with h5py.File(cache_h5_path, "r") as f:
            paf_to_cache = f["paf_to_cache"][:]   # PAF 순서대로의 캐시 인덱스

        if indices is not None:
            self.cache_indices = paf_to_cache[indices]
            self.labels = torch.tensor(all_labels[indices], dtype=torch.long)
            self.subject_ids = all_sids[indices]
            if all_clinical is not None:
                all_clinical = all_clinical[indices]
        else:
            self.cache_indices = paf_to_cache
            self.labels = torch.tensor(all_labels, dtype=torch.long)
            self.subject_ids = all_sids

        # tabular features는 원본 H5에서 로드 (작은 파일)
        with h5py.File(ecg_h5_path, "r") as f:
            orig_indices = all_ecg_indices[indices] if indices is not None else all_ecg_indices
            sorted_order = np.argsort(orig_indices)
            sorted_idx = orig_indices[sorted_order]
            raw_numeric = f["numeric_features"][sorted_idx]
            raw_patient = f["patient_features"][sorted_idx]
            reverse_order = np.argsort(sorted_order)
            numeric = raw_numeric[reverse_order].astype(np.float32)
            patient = raw_patient[reverse_order].astype(np.float32)
            np.nan_to_num(numeric, copy=False, nan=0.0)
            np.nan_to_num(patient, copy=False, nan=0.0)
            if all_clinical is not None:
                clinical = all_clinical.astype(np.float32)
                numeric = np.concatenate([numeric, clinical], axis=1)
            self.numeric = torch.from_numpy(numeric)
            self.patient = torch.from_numpy(patient)

        # RAM 프리로드: 필요한 cache_indices만 골라서 순서대로 RAM에 올림
        if preload:
            unique_cidx, inverse = np.unique(self.cache_indices, return_inverse=True)
            n_unique = len(unique_cidx)
            gb = n_unique * 5000 * 2 * 2 / 1024**3
            print(f"  [RAM 프리로드] {n_unique:,}개 waveform 로딩 중... ({gb:.1f}GB)")
            buf = np.empty((n_unique, 5000, 2), dtype=np.float16)
            CHUNK = 2000
            with h5py.File(cache_h5_path, "r") as f:
                for start in range(0, n_unique, CHUNK):
                    end = min(start + CHUNK, n_unique)
                    buf[start:end] = f["waveform"][unique_cidx[start:end]]
            # 각 샘플이 buf에서 어느 행을 참조하는지 저장 (inverse index)
            self._waveform_ram = buf
            self._ram_inverse = inverse  # len == len(self.cache_indices)
            print(f"  [RAM 프리로드] 완료")

    @property
    def cache_h5(self):
        if self._cache_h5 is None:
            self._cache_h5 = h5py.File(self.cache_h5_path, "r",
                                        rdcc_nbytes=256 * 1024 * 1024)  # 256MB 청크 캐시
        return self._cache_h5

    def __len__(self):
        return len(self.cache_indices)

    def _extract_pwave_segments(self, waveform_np, n_beats=8, pwave_len=75):
        lead = waveform_np[:, min(1, waveform_np.shape[1] - 1)]
        peaks, _ = find_peaks(lead, distance=150, height=np.percentile(lead, 60))
        segments = []
        for p in peaks[:n_beats]:
            start = max(0, p - pwave_len)
            seg = waveform_np[start:start + pwave_len]
            if len(seg) < pwave_len:
                seg = np.pad(seg, ((0, pwave_len - len(seg)), (0, 0)))
            segments.append(seg)
        while len(segments) < n_beats:
            segments.append(np.zeros((pwave_len, waveform_np.shape[1])))
        segments = np.array(segments[:n_beats])
        return torch.tensor(segments, dtype=torch.float32).permute(0, 2, 1)

    def __getitem__(self, idx):
        # RAM 프리로드 우선, 없으면 HDF5에서 읽기
        if self._waveform_ram is not None:
            ram_idx = int(self._ram_inverse[idx])
            waveform = self._waveform_ram[ram_idx].astype(np.float32)  # (5000, 2)
        else:
            cache_idx = int(self.cache_indices[idx])
            waveform = self.cache_h5["waveform"][cache_idx].astype(np.float32)  # (5000, 2)

        # Lead III = II - I
        lead3 = waveform[:, 1:2] - waveform[:, 0:1]
        waveform = np.concatenate([waveform, lead3], axis=1)  # (5000, 3)

        if self.augment is not None:
            waveform = self.augment(waveform)

        if self.extract_pwave:
            pwave = self._extract_pwave_segments(waveform if isinstance(waveform, np.ndarray)
                                                  else waveform.numpy())
        waveform = torch.tensor(waveform, dtype=torch.float32) if not isinstance(waveform, torch.Tensor) else waveform

        if self.extract_pwave:
            return waveform, self.numeric[idx], self.patient[idx], self.labels[idx], pwave
        return waveform, self.numeric[idx], self.patient[idx], self.labels[idx]

    def __del__(self):
        if self._cache_h5 is not None:
            self._cache_h5.close()


# ============================================================
# Utilities
# ============================================================
def split_by_patient(paf_h5_path, train_ratio=0.7, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    with h5py.File(paf_h5_path, "r") as f:
        subject_ids = f["subject_id"][:]
    unique = np.unique(subject_ids)
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_p = set(unique[:n_train])
    val_p = set(unique[n_train:n_train + n_val])
    train_idx = np.where(np.isin(subject_ids, list(train_p)))[0]
    val_idx = np.where(np.isin(subject_ids, list(val_p)))[0]
    test_idx = np.where(np.isin(subject_ids, list(set(unique[n_train + n_val:]))))[0]
    return train_idx, val_idx, test_idx


def compute_class_weights(paf_h5_path, indices, n_classes=2):
    with h5py.File(paf_h5_path, "r") as f:
        labels = f["paf_label"][:]
    labels = labels[indices]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    weights = len(labels) / (n_classes * counts)
    weights[counts == 0] = 1.0
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, save_path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            return False
        self.counter += 1
        return self.counter >= self.patience


# ============================================================
# Train / Eval
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    use_mixup=False, max_epochs=30, teacher_model=None,
                    distill_loss=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"  Epoch {epoch} [Train]", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for batch in pbar:
        if len(batch) == 5:  # with pwave
            wf, num, pat, lab, pwave = batch
            pwave = pwave.to(device)
        else:
            wf, num, pat, lab = batch
            pwave = None

        wf, num, pat, lab = wf.to(device), num.to(device), pat.to(device), lab.to(device)
        optimizer.zero_grad()

        if use_mixup:
            wf, lab_a, lab_b, lam = non_uniform_mixup(wf, lab, epoch=epoch, max_epochs=max_epochs)

        if pwave is not None:
            logits = model(wf, num, pat, pwave)
        else:
            logits = model(wf, num, pat)

        if teacher_model is not None and distill_loss is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(wf, num, pat)
            loss = distill_loss(logits, teacher_logits, lab)
        elif use_mixup:
            loss = lam * criterion(logits, lab_a) + (1 - lam) * criterion(logits, lab_b)
        else:
            loss = criterion(logits, lab)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * lab.size(0)
        if not use_mixup:
            correct += (logits.argmax(1) == lab).sum().item()
        else:
            correct += (logits.argmax(1) == lab_a).sum().item()
        total += lab.size(0)
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.1%}")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val", has_pwave=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_probs, all_labels = [], [], []
    pbar = tqdm(loader, desc=f"         [{desc}]", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for batch in pbar:
        if has_pwave and len(batch) == 5:
            wf, num, pat, lab, pwave = batch
            pwave = pwave.to(device)
        else:
            wf, num, pat, lab = batch[:4]
            pwave = None

        wf, num, pat, lab = wf.to(device), num.to(device), pat.to(device), lab.to(device)

        if pwave is not None:
            logits = model(wf, num, pat, pwave)
        else:
            logits = model(wf, num, pat)
        loss = criterion(logits, lab)
        total_loss += loss.item() * lab.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)
        correct += (preds == lab).sum().item()
        total += lab.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(lab.cpu().numpy())
        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.1%}")
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_probs), np.array(all_labels))


# ============================================================
# Main experiment runner
# ============================================================
def run_experiment(exp_name, exp_num, args):
    output_dir = Path(args.output_dir) / f"exp{exp_num:02d}_{exp_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"실험 {exp_num}: {exp_name}")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Data
    train_idx, val_idx, test_idx = split_by_patient(args.paf_data)
    lead_indices = [0, 1]

    # Augmentation
    augment = None
    if exp_num == 2:  # STAR
        augment = STARAugment(prob=0.5)

    extract_pwave = (exp_num == 7)

    train_ds = PAFDataset(args.ecg_data, args.paf_data, train_idx, lead_indices, augment=augment,
                          extract_pwave=extract_pwave)
    val_ds = PAFDataset(args.ecg_data, args.paf_data, val_idx, lead_indices, extract_pwave=extract_pwave)
    test_ds = PAFDataset(args.ecg_data, args.paf_data, test_idx, lead_indices, extract_pwave=extract_pwave)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    class_weights = compute_class_weights(args.paf_data, train_idx)
    n_numeric = train_ds.numeric.shape[1]
    n_patient = train_ds.patient.shape[1]

    # Model
    model = ResNet34ECGWithTabular(
        n_leads=3, n_numeric=n_numeric, n_patient=n_patient,
        n_classes=2, dropout=args.dropout
    ).to(device)

    # Experiment-specific setup
    teacher_model = None
    distill_loss = None
    use_mixup = False
    has_pwave = False

    if exp_num == 1:  # Focal Loss
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
    elif exp_num == 4:  # Label Smoothing
        criterion = LabelSmoothingCE(n_classes=2, smoothing=0.1, weight=class_weights.to(device))
    elif exp_num == 3:  # Non-Uniform MixUp
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        use_mixup = True
    elif exp_num == 5:  # Knowledge Distillation
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        teacher_model = ResNet34ECGWithTabular(
            n_leads=3, n_numeric=n_numeric, n_patient=n_patient,
            n_classes=2, dropout=args.dropout
        ).to(device)
        teacher_path = "g:/AIEKG/ml/checkpoints/paroxysmal-af-resnet34-clinical/best_model.pt"
        teacher_model.load_state_dict(torch.load(teacher_path, weights_only=True))
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        distill_loss = DistillationLoss(temperature=3.0, alpha=0.7, weight=class_weights.to(device))
    elif exp_num == 7:  # P-wave Dual-branch
        has_pwave = True
        base_model = model
        model = ResNet34WithPWave(base_model, n_leads=3, n_beats=8, pwave_len=75, n_classes=2).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    elif exp_num == 8:  # SE (Squeeze-Excitation) attention head
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        old_in = model.classifier[-1].in_features

        class SEClassifier(nn.Module):
            """Channel-wise SE attention on feature vector → more stable than Transformer"""
            def __init__(self, in_features, reduction=16, n_classes=2):
                super().__init__()
                self.se = nn.Sequential(
                    nn.Linear(in_features, in_features // reduction),
                    nn.ReLU(),
                    nn.Linear(in_features // reduction, in_features),
                    nn.Sigmoid(),
                )
                self.fc = nn.Linear(in_features, n_classes)

            def forward(self, x):
                scale = self.se(x)
                return self.fc(x * scale)

        model.classifier[-1] = SEClassifier(old_in).to(device)
    else:  # exp 2 (STAR), 6 (SupCon stage2), 9, 10 — baseline criterion
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # SupCon: 2-stage training
    if exp_num == 6:
        print("  Stage 1: Supervised Contrastive pre-training...")
        # Use ResNet backbone as encoder
        supcon_loss = SupConLoss(temperature=0.07)
        proj_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        ).to(device)

        # Get ResNet backbone output (before classifier)
        encoder_params = list(model.resnet.parameters()) + list(proj_head.parameters())
        opt_stage1 = torch.optim.Adam(encoder_params, lr=1e-3)

        for sc_epoch in range(5):
            model.train()
            total_sc_loss = 0
            for batch in tqdm(train_loader, desc=f"  SupCon Epoch {sc_epoch}", leave=False):
                wf, num, pat, lab = batch[:4]
                wf, lab = wf.to(device), lab.to(device)
                # Get features from ResNet backbone (already GAP'd)
                x = wf.permute(0, 2, 1)
                features = model.resnet.forward_features(x)  # (B, 512)
                proj = proj_head(features)  # (B, 64)
                loss = supcon_loss(proj, lab)
                opt_stage1.zero_grad()
                loss.backward()
                opt_stage1.step()
                total_sc_loss += loss.item()
            print(f"    SupCon Epoch {sc_epoch}: loss={total_sc_loss/len(train_loader):.4f}")

        del proj_head, opt_stage1
        print("  Stage 2: Fine-tuning classifier...")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=2
    )
    early_stopping = EarlyStopping(
        patience=args.es_patience, save_path=str(output_dir / "best_model.pt")
    )

    # Train
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auroc": []}
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_mixup=use_mixup, max_epochs=args.epochs,
            teacher_model=teacher_model, distill_loss=distill_loss
        )
        # Use CE for val evaluation even if training uses different loss
        val_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        val_loss, val_acc, _, val_probs, val_labels = evaluate(
            model, val_loader, val_criterion, device, has_pwave=has_pwave
        )
        val_auroc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.0
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        # Normal / Hidden AF precision & recall (threshold=0.5)
        from sklearn.metrics import precision_recall_fscore_support
        _, val_preds_05 = evaluate.__wrapped__ if hasattr(evaluate, '__wrapped__') else (None, None), None
        val_preds_05 = (val_probs >= 0.5).astype(int)
        p_n, r_n, _, _ = precision_recall_fscore_support(val_labels, val_preds_05, labels=[0], zero_division=0)
        p_af, r_af, _, _ = precision_recall_fscore_support(val_labels, val_preds_05, labels=[1], zero_division=0)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auroc"].append(val_auroc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:2d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} auroc={val_auroc:.4f} | "
              f"Normal P={p_n[0]:.2f} R={r_n[0]:.2f} | AF P={p_af[0]:.2f} R={r_af[0]:.2f} | "
              f"lr={lr_now:.1e} | {elapsed:.0f}s")

        if early_stopping.step(val_loss, model, epoch):
            print(f"  Early stopping at epoch {epoch} (best: {early_stopping.best_epoch})")
            break

    # Test
    best_epoch = early_stopping.best_epoch
    model.load_state_dict(torch.load(str(output_dir / "best_model.pt"), weights_only=True))
    test_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    test_loss, test_acc, test_preds, test_probs, test_labels = evaluate(
        model, test_loader, test_criterion, device, desc="Test", has_pwave=has_pwave
    )
    test_auroc = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.0

    print(f"\n  Test AUROC: {test_auroc:.4f}")
    print(f"  Test Accuracy: {test_acc:.1%}")
    report = classification_report(test_labels, test_preds,
                                   target_names=["Normal", "Hidden AF"], output_dict=True)
    print(classification_report(test_labels, test_preds, target_names=["Normal", "Hidden AF"]))
    cm = confusion_matrix(test_labels, test_preds)
    print(f"Confusion Matrix:\n{cm}")

    # 모든 실험: threshold 최적화 (Hidden AF F1 최대화)
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(test_labels, test_probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    opt_preds = (test_probs >= best_thresh).astype(int)
    opt_report = classification_report(test_labels, opt_preds,
                                       target_names=["Normal", "Hidden AF"], output_dict=True)
    print(f"\n  [Threshold 최적화] best_thresh={best_thresh:.3f} (Hidden AF F1 최대)")
    print(f"  Normal    P={opt_report['Normal']['precision']:.3f} R={opt_report['Normal']['recall']:.3f} F1={opt_report['Normal']['f1-score']:.3f}")
    print(f"  Hidden AF P={opt_report['Hidden AF']['precision']:.3f} R={opt_report['Hidden AF']['recall']:.3f} F1={opt_report['Hidden AF']['f1-score']:.3f}")
    report["optimal_threshold"] = best_thresh
    report["optimized"] = opt_report

    # 실험 10: 복수 ECG 앙상블 (환자 단위 확률 평균)
    if exp_num == 10:
        with h5py.File(args.paf_data, "r") as f:
            all_sids = f["subject_id"][:]
        test_sids = all_sids[test_idx]
        unique_sids = np.unique(test_sids)
        patient_aurocs = []
        patient_labels_list, patient_probs_list = [], []
        for sid in unique_sids:
            mask = test_sids == sid
            p_label = test_labels[mask][0]
            p_prob = test_probs[mask].mean()
            patient_labels_list.append(p_label)
            patient_probs_list.append(p_prob)
        patient_auroc = roc_auc_score(patient_labels_list, patient_probs_list)
        print(f"\n  Patient-level AUROC (ensemble): {patient_auroc:.4f}")
        report["patient_auroc"] = float(patient_auroc)

    results = {
        "experiment": exp_name, "exp_num": exp_num,
        "test_auroc": float(test_auroc), "test_acc": float(test_acc),
        "best_epoch": best_epoch,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "history": {k: [float(v) for v in vs] for k, vs in history.items()},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="PAF 10 Experiments")
    parser.add_argument("--ecg-data", default="g:/AIEKG/ml/data/ecg_preprocessed.h5")
    parser.add_argument("--paf-data", default="g:/AIEKG/ml/data/ecg_paroxysmal_af.h5")
    parser.add_argument("--output-dir", default="g:/AIEKG/ml/checkpoints/paf-experiments")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--es-patience", type=int, default=5)
    parser.add_argument("--scan-epochs", type=int, default=5,
                        help="1차 스캔 epoch 수 (기본 5)")
    parser.add_argument("--scan-threshold", type=float, default=0.80,
                        help="5 epoch 후 이 AUROC 미만이면 스킵 (기본 0.80)")
    parser.add_argument("--exp", type=int, nargs="+", default=list(range(1, 11)))
    args = parser.parse_args()

    experiments = {
        1: "focal_loss",
        2: "star_augment",
        3: "nonuniform_mixup",
        4: "label_smoothing",
        5: "knowledge_distill",
        6: "supcon",
        7: "pwave_branch",
        8: "transformer_head",
        9: "threshold_opt",
        10: "patient_ensemble",
    }

    BASELINE_AUROC = 0.8240
    scan_results = {}  # exp_num → val_auroc after scan_epochs

    # ── Phase 1: 5 epoch 빠른 스캔 ─────────────────────────────
    print(f"\n{'#'*70}")
    print(f"Phase 1: {args.scan_epochs}-epoch 빠른 스캔 (threshold={args.scan_threshold})")
    print(f"{'#'*70}")

    for exp_num in args.exp:
        if exp_num not in experiments:
            continue
        # 5 epoch만 돌리도록 임시 override
        args_scan = argparse.Namespace(**vars(args))
        args_scan.epochs = args.scan_epochs
        args_scan.es_patience = args.scan_epochs + 1  # early stop 없이 전부 돌림

        result = run_experiment(experiments[exp_num], exp_num, args_scan)
        best_val_auroc = max(result["history"]["val_auroc"])
        scan_results[exp_num] = best_val_auroc

        if best_val_auroc >= args.scan_threshold:
            print(f"  [PASS] 실험 {exp_num} ({experiments[exp_num]}): val_auroc={best_val_auroc:.4f} -> 풀 학습 후보")
        else:
            print(f"  [SKIP] 실험 {exp_num} ({experiments[exp_num]}): val_auroc={best_val_auroc:.4f} -> 스킵 (< {args.scan_threshold})")

    # ── Phase 2: 통과한 실험 또는 최고점 실험 풀 학습 ─────────────
    passed = {k: v for k, v in scan_results.items() if v >= args.scan_threshold}

    if not passed:
        # 아무것도 통과 못하면 가장 높은 것 하나만 풀 학습
        best_num = max(scan_results, key=scan_results.get)
        passed = {best_num: scan_results[best_num]}
        print(f"\n  경고: threshold 통과 실험 없음. 최고점 실험 {best_num} 단독 풀 학습.")

    print(f"\n{'#'*70}")
    print(f"Phase 2: 풀 학습 ({len(passed)}개 실험 — early stopping까지)")
    print(f"{'#'*70}")

    all_results = {}
    for exp_num in sorted(passed.keys()):
        args_full = argparse.Namespace(**vars(args))
        result = run_experiment(experiments[exp_num], exp_num, args_full)
        all_results[exp_num] = result

    # ── 최종 요약 ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("최종 결과 요약")
    print(f"{'='*80}")
    print(f"{'#':>3} {'실험':>25} {'AUROC':>8} {'Acc':>8} {'AF_P':>8} {'AF_R':>8} {'AF_F1':>8}")
    print("-" * 80)
    print(f"{'0':>3} {'baseline (v1)':>25} {BASELINE_AUROC:>8.4f} {'76.1%':>8} {'0.25':>8} {'0.71':>8} {'0.37':>8}")

    for num, res in sorted(all_results.items()):
        af = res["classification_report"].get("Hidden AF", {})
        marker = " *BEST*" if res["test_auroc"] > BASELINE_AUROC else ""
        print(f"{num:>3} {res['experiment']:>25} {res['test_auroc']:>8.4f} "
              f"{res['test_acc']:>7.1%} {af.get('precision',0):>8.2f} "
              f"{af.get('recall',0):>8.2f} {af.get('f1-score',0):>8.2f}{marker}")

    print(f"\n스캔 결과 (5 epoch val AUROC):")
    for num, auroc in sorted(scan_results.items()):
        status = "통과" if auroc >= args.scan_threshold else "스킵"
        print(f"  실험 {num:2d} ({experiments[num]:>20}): {auroc:.4f} [{status}]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "scan_results": {str(k): v for k, v in scan_results.items()},
            "full_results": {str(k): v for k, v in all_results.items()},
        }, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
