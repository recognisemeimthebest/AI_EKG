"""
CNN-TCN 부정맥 분류 모델 학습 스크립트

사용법:
  python train.py                          # 기본 설정
  python train.py --batch-size 32          # 배치 크기 조정
  python train.py --epochs 50 --lr 0.0005  # 하이퍼파라미터 변경
"""
import argparse
import os
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from cnn_tcn import CNNTCN
from cnn_tcn_cbam import CNNTCN_CBAM
from dataset import ECGDataset, split_by_patient, compute_class_weights


# ======================== Early Stopping ========================
class EarlyStopping:
    """val loss가 patience 에폭 동안 개선되지 않으면 학습 중단"""

    def __init__(self, patience=10, min_delta=1e-4, save_path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            return False
        self.counter += 1
        return self.counter >= self.patience


# ======================== 학습/검증 함수 ========================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch} [Train]", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for waveform, numeric, patient, label in pbar:
        waveform = waveform.to(device)
        numeric = numeric.to(device)
        patient = patient.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(waveform, numeric, patient)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        correct += (logits.argmax(1) == label).sum().item()
        total += label.size(0)

        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.1%}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"         [{desc}]", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for waveform, numeric, patient, label in pbar:
        waveform = waveform.to(device)
        numeric = numeric.to(device)
        patient = patient.to(device)
        label = label.to(device)

        logits = model(waveform, numeric, patient)
        loss = criterion(logits, label)

        total_loss += loss.item() * label.size(0)
        preds = logits.argmax(1)
        correct += (preds == label).sum().item()
        total += label.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.1%}")

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ======================== 메인 ========================
def main():
    parser = argparse.ArgumentParser(description="CNN-TCN 학습")
    parser.add_argument("--data", type=str, default="g:/AIEKG/ml/data/ecg_preprocessed.h5")
    parser.add_argument("--output-dir", type=str, default="g:/AIEKG/ml/checkpoints")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0=메인 스레드, Windows에서는 0 권장)")
    parser.add_argument("--resume", type=str, default=None,
                        help="체크포인트 경로 (이어서 학습)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="학습 데이터 subset 크기 (빠른 테스트용)")
    parser.add_argument("--shutdown", action="store_true",
                        help="학습 완료 후 PC 자동 종료")
    parser.add_argument("--model", type=str, default="cnn-tcn",
                        choices=["cnn-tcn", "cnn-tcn-cbam"],
                        help="모델 선택 (기본: cnn-tcn)")
    parser.add_argument("--leads", type=str, default="0,1",
                        help="사용할 리드 (예: '0,1' = Lead I,II). 'all'이면 전체 12-lead")
    parser.add_argument("--add-lead3", action="store_true", default=True,
                        help="Lead III = Lead II - Lead I 자동 계산 추가 (기본: True)")
    parser.add_argument("--no-bp", action="store_true",
                        help="BP 피처 제외")
    args = parser.parse_args()

    # 리드 파싱
    lead_indices = None
    if args.leads and args.leads != "all":
        lead_indices = [int(x) for x in args.leads.split(",")]

    lead_count = len(lead_indices) if lead_indices else 12
    if args.add_lead3 and lead_indices and 0 in lead_indices and 1 in lead_indices:
        lead_count += 1
    bp_tag = "" if args.no_bp else "-bp"
    lead_tag = f"{lead_count}lead{bp_tag}"
    output_dir = Path(args.output_dir) / f"{args.model}-{lead_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ============================
    # 데이터 분할
    # ============================
    print("\n[1/4] 데이터 분할 중 (환자 단위)...")
    train_idx, val_idx, test_idx = split_by_patient(args.data)
    if args.max_train_samples and len(train_idx) > args.max_train_samples:
        rng = np.random.RandomState(42)
        train_idx = rng.choice(train_idx, args.max_train_samples, replace=False)
        # val/test도 비율 맞춰 줄이기
        val_size = max(1000, int(args.max_train_samples * 0.2))
        test_size = max(1000, int(args.max_train_samples * 0.2))
        val_idx = rng.choice(val_idx, min(val_size, len(val_idx)), replace=False)
        test_idx = rng.choice(test_idx, min(test_size, len(test_idx)), replace=False)
    print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    # 데이터셋/로더
    use_bp = not args.no_bp
    train_ds = ECGDataset(args.data, train_idx, lead_indices=lead_indices, use_bp=use_bp)
    val_ds = ECGDataset(args.data, val_idx, lead_indices=lead_indices, use_bp=use_bp)
    test_ds = ECGDataset(args.data, test_idx, lead_indices=lead_indices, use_bp=use_bp)
    if args.add_lead3 and lead_indices is not None and 0 in lead_indices and 1 in lead_indices:
        train_ds.add_lead3 = True
        val_ds.add_lead3 = True
        test_ds.add_lead3 = True

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # 클래스 가중치
    class_weights = compute_class_weights(args.data, train_idx)
    print(f"  Class weights: {class_weights.tolist()}")

    # ============================
    # 모델 생성
    # ============================
    # 피처 수 자동 감지
    n_numeric = train_ds.numeric.shape[1]
    n_leads = len(lead_indices) if lead_indices else 12
    if train_ds.add_lead3:
        n_leads += 1
    print(f"\n[2/4] 모델 생성 중... ({args.model}, leads={n_leads}, numeric={n_numeric})")
    if args.model == "cnn-tcn-cbam":
        model = CNNTCN_CBAM(n_leads=n_leads, n_numeric=n_numeric, dropout=args.dropout).to(device)
    else:
        model = CNNTCN(n_leads=n_leads, n_numeric=n_numeric, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터: {n_params:,}")
    print(f"  모델 크기: {n_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(
        patience=args.patience,
        save_path=str(output_dir / "best_model.pt")
    )

    # 체크포인트에서 이어서 학습
    start_epoch = 1
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    if args.resume:
        print(f"\n  체크포인트 로드: {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", history)
        early_stopping.best_loss = ckpt.get("best_val_loss", float("inf"))
        early_stopping.best_epoch = ckpt.get("best_epoch", 0)
        early_stopping.counter = ckpt.get("es_counter", 0)
        print(f"  에폭 {start_epoch}부터 이어서 학습")

    # ============================
    # 학습 루프
    # ============================
    print(f"\n[3/4] 학습 시작 (에폭 {start_epoch}~{args.epochs}, patience {args.patience})...")
    print(f"  Batch size: {args.batch_size}, LR: {args.lr}, Dropout: {args.dropout}")
    print("-" * 70)
    print(f"{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>10} | "
          f"{'Val Loss':>10} {'Val Acc':>10} | {'LR':>10} {'Time':>6}")
    print("-" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc="Val")

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"{epoch:>5} | {train_loss:>10.4f} {train_acc:>9.1%} | "
              f"{val_loss:>10.4f} {val_acc:>9.1%} | {lr:>10.6f} {elapsed:>5.0f}s")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        scheduler.step(val_loss)

        # 매 에폭 체크포인트 저장 (중단 후 이어서 학습 가능)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_val_loss": early_stopping.best_loss,
            "best_epoch": early_stopping.best_epoch,
            "es_counter": early_stopping.counter,
            "args": vars(args),
        }, str(output_dir / "checkpoint.pt"))

        if early_stopping.step(val_loss, model, epoch):
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best epoch: {early_stopping.best_epoch}")
            break

    # ============================
    # 테스트 평가
    # ============================
    print(f"\n[4/4] 테스트 평가 중...")
    model.load_state_dict(torch.load(str(output_dir / "best_model.pt"), weights_only=True))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device, desc="Test")

    class_names = ["Normal", "AFib", "Other"]
    report = classification_report(labels, preds, target_names=class_names)
    cm = confusion_matrix(labels, preds)

    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.1%}")
    print(f"\n  Classification Report:")
    print(report)
    print(f"  Confusion Matrix:")
    print(cm)

    # ============================
    # 결과 저장
    # ============================
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_epoch": early_stopping.best_epoch,
        "n_params": n_params,
        "args": vars(args),
        "history": history,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    results_path = output_dir / "train_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  모델 저장: {output_dir / 'best_model.pt'}")
    print(f"  결과 저장: {results_path}")
    print("  완료!")

    if args.shutdown:
        print("\n  60초 후 PC 종료됩니다... (취소: shutdown /a)")
        os.system("shutdown /s /t 60")


if __name__ == "__main__":
    main()
