"""
deep_learning.py — Missed-Lead Detector
PyTorch Deep Learning model for tabular lead classification.
GPU-accelerated feedforward network with BatchNorm, Dropout, residual connections.
"""

import os
import numpy as np
import pandas as pd
import pickle
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, ConfusionMatrixDisplay,
                              precision_recall_curve, roc_curve)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# DEVICE SELECTION — GPU if available, else CPU
# ─────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[deep_learning] Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[deep_learning] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[deep_learning] CUDA: {torch.version.cuda}")


# ─────────────────────────────────────────────────────────
# 1. NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """A residual block with BatchNorm, ReLU, and Dropout."""

    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(x + self.block(x)))


class LeadClassifier(nn.Module):
    """
    Deep feedforward network for lead classification.

    Architecture:
        Input (9 features)
        -> Linear(9, 128) + BatchNorm + GELU + Dropout
        -> Linear(128, 256) + BatchNorm + GELU + Dropout
        -> ResidualBlock(256)
        -> ResidualBlock(256)
        -> Linear(256, 128) + BatchNorm + GELU + Dropout
        -> ResidualBlock(128)
        -> Linear(128, 64) + BatchNorm + GELU + Dropout
        -> Linear(64, 1) -> Sigmoid
    """

    def __init__(self, input_dim=9, dropout=0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)

        self.network = nn.Sequential(
            # Expansion layer
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            # Wide layer
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),

            # Residual blocks
            ResidualBlock(256, dropout),
            ResidualBlock(256, dropout),

            # Contraction layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            ResidualBlock(128, dropout),

            # Output layers
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """He initialization for ReLU/GELU-compatible layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_bn(x)
        return self.network(x).squeeze(-1)

    def predict_proba_np(self, X_np):
        """Predict probabilities from numpy array (for sklearn compatibility)."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_np).to(DEVICE)
            probs = self(X_tensor).cpu().numpy()
        return np.column_stack([1 - probs, probs])

    def predict_np(self, X_np, threshold=0.5):
        """Predict classes from numpy array."""
        probs = self.predict_proba_np(X_np)[:, 1]
        return (probs >= threshold).astype(int)


# ─────────────────────────────────────────────────────────
# 2. TRAINING LOOP
# ─────────────────────────────────────────────────────────
def compute_class_weights(y):
    """Compute balanced class weights for imbalanced data."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    weight_neg = len(y) / (2.0 * n_neg)
    weight_pos = len(y) / (2.0 * n_pos)
    return torch.FloatTensor([weight_pos, weight_neg]).to(DEVICE)


def train_deep_model(X_train, y_train, X_val, y_val,
                     epochs=200, lr=1e-3, batch_size=64,
                     dropout=0.3, patience=30, verbose=True):
    """
    Train the deep learning model with early stopping and LR scheduling.

    Returns:
        model (nn.Module), history (dict)
    """
    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.FloatTensor(y_train).to(DEVICE)
    X_vl = torch.FloatTensor(X_val).to(DEVICE)
    y_vl = torch.FloatTensor(y_val).to(DEVICE)

    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=False)

    # Initialize model
    model = LeadClassifier(input_dim=X_train.shape[1], dropout=dropout).to(DEVICE)

    # Class-weighted loss
    class_weights = compute_class_weights(y_train)
    pos_weight = class_weights[1] / class_weights[0]
    criterion = nn.BCELoss(reduction="none")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=10, min_lr=1e-6)

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}
    best_val_loss = float("inf")
    best_auc = 0.0
    best_state = None
    no_improve = 0

    if verbose:
        print(f"\n[deep_learning] Training on {DEVICE} | "
              f"epochs={epochs} | lr={lr} | batch={batch_size}")
        print(f"[deep_learning] Model parameters: "
              f"{sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, epochs + 1):
        # ── Training phase ──
        model.train()
        train_losses = []
        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            # Apply class weights
            weights = torch.where(y_batch == 1, class_weights[1], class_weights[0])
            loss = (loss * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation phase ──
        model.eval()
        with torch.no_grad():
            val_preds = model(X_vl)
            val_loss_raw = criterion(val_preds, y_vl)
            val_weights = torch.where(y_vl == 1, class_weights[1], class_weights[0])
            val_loss = (val_loss_raw * val_weights).mean().item()
            val_probs = val_preds.cpu().numpy()

        train_loss = np.mean(train_losses)
        current_lr = optimizer.param_groups[0]["lr"]

        # Compute AUC
        try:
            val_auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            val_auc = 0.5

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        scheduler.step(val_loss)

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 25 == 0 or epoch == 1):
            marker = " *" if no_improve == 0 else ""
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"LR: {current_lr:.2e}{marker}")

        if no_improve >= patience:
            if verbose:
                print(f"\n[deep_learning] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    if verbose:
        print(f"[deep_learning] Best Val AUC: {best_auc:.4f} | "
              f"Best Val Loss: {best_val_loss:.4f}")

    return model, history


# ─────────────────────────────────────────────────────────
# 3. VISUALIZATION
# ─────────────────────────────────────────────────────────
def plot_training_history(history, out_dir):
    """Plot training curves: loss, AUC, learning rate."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curve
    axes[0].plot(history["train_loss"], label="Train Loss", color="#2196F3")
    axes[0].plot(history["val_loss"], label="Val Loss", color="#F44336")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC curve
    axes[1].plot(history["val_auc"], label="Val AUC", color="#4CAF50", linewidth=2)
    axes[1].set_title("Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].plot(history["lr"], label="Learning Rate", color="#FF9800")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Deep Learning Model — Training History", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dl_training_history.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print("[deep_learning] Saved dl_training_history.png")


def plot_dl_confusion_matrix(y_true, y_pred, out_dir):
    """Plot confusion matrix for the deep learning model."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Replied", "Missed"]).plot(ax=ax)
    ax.set_title("Deep Learning Model — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dl_confusion_matrix.png"), dpi=150)
    plt.close()
    print("[deep_learning] Saved dl_confusion_matrix.png")


def plot_dl_roc_curve(y_true, y_probs, out_dir):
    """Plot ROC curve for the deep learning model."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#E91E63", linewidth=2, label=f"DL Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Deep Learning Model — ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dl_roc_curve.png"), dpi=150)
    plt.close()
    print("[deep_learning] Saved dl_roc_curve.png")


# ─────────────────────────────────────────────────────────
# 4. SAVE / LOAD
# ─────────────────────────────────────────────────────────
def save_dl_model(model, scaler, history, out_dir):
    """Save the trained DL model, scaler, and training history."""
    # Save model state dict (PyTorch)
    model_path = os.path.join(out_dir, "dl_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": 9,
            "dropout": 0.3,
        },
        "device": str(DEVICE),
    }, model_path)

    # Save scaler (sklearn)
    scaler_path = os.path.join(out_dir, "dl_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save history
    history_path = os.path.join(out_dir, "dl_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    print(f"[deep_learning] Model saved to {model_path}")
    print(f"[deep_learning] Scaler saved to {scaler_path}")
    print(f"[deep_learning] History saved to {history_path}")


def load_dl_model(model_path, scaler_path):
    """Load a saved DL model and scaler."""
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = checkpoint["model_config"]
    model = LeadClassifier(input_dim=config["input_dim"],
                           dropout=config["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# ─────────────────────────────────────────────────────────
# 5. STANDALONE TRAINING (when run directly)
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Lazy import to avoid circular dependency when run standalone
    import sys as _sys
    _SRC = os.path.dirname(os.path.abspath(__file__))
    if _SRC not in _sys.path:
        _sys.path.insert(0, _SRC)
    from train_model import load_and_engineer, cluster_leads
    from sklearn.model_selection import train_test_split

    BASE = os.path.dirname(__file__)
    DATA = os.path.join(BASE, "..", "data", "leads.csv")
    MODELS = os.path.join(BASE, "..", "models")
    OUT = os.path.join(BASE, "..", "outputs")

    print("=" * 60)
    print("  MISSED-LEAD DETECTOR — DEEP LEARNING TRAINING")
    print("=" * 60)

    # Load and engineer features
    X, y, df = load_and_engineer(DATA)
    df = cluster_leads(df)

    # Split data
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Further split training into train/val for early stopping
    X_tr_final, X_val, y_tr_final, y_val = train_test_split(
        X_tr_s, y_tr.values, test_size=0.15, random_state=42, stratify=y_tr.values
    )

    print(f"\n[data] Train: {len(X_tr_final)} | Val: {len(X_val)} | Test: {len(X_te_s)}")
    print(f"[data] Positive rate: {y.mean():.1%}")

    # Train deep learning model
    model, history = train_deep_model(
        X_tr_final, y_tr_final, X_val, y_val,
        epochs=200, lr=1e-3, batch_size=64,
        dropout=0.3, patience=30, verbose=True
    )

    # Evaluate on test set
    y_pred = model.predict_np(X_te_s)
    y_probs = model.predict_proba_np(X_te_s)[:, 1]
    auc = roc_auc_score(y_te, y_probs)

    report = classification_report(y_te, y_pred, target_names=["Replied", "Missed"])
    print(f"\n[deep_learning] Test AUC: {auc:.4f}")
    print(f"\n[deep_learning] Classification Report:\n{report}")

    # Save everything
    save_dl_model(model, scaler, history, MODELS)

    # Generate visualizations
    plot_training_history(history, OUT)
    plot_dl_confusion_matrix(y_te.values, y_pred, OUT)
    plot_dl_roc_curve(y_te.values, y_probs, OUT)

    # Save classification report
    with open(os.path.join(OUT, "dl_classification_report.txt"), "w") as f:
        f.write(report)

    print("\n[deep_learning] All artefacts saved to /models and /outputs")
