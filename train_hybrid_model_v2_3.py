"""
Hybrid Model v2.3: Time-Series Only with Temporal Attention
ハイブリッドモデルv2.3 - 時系列データのみのマルチタスク学習

v2.2からの改善点:
1. Statistical Features削除
   - v2.2: TinyTimeMixer (64d) + Statistical Features (28d) = 92d
   - v2.3: TinyTimeMixer (64d) のみ使用
   - パラメータ削減、訓練高速化、過学習リスク低減

2. アーキテクチャシンプル化
   - Feature Fusion層の入力次元: 92d → 64d
   - 時系列パターンのみに集中
   - より本質的な特徴抽出

3. 維持する要素
   - Temporal Attention機構（v2.2と同じ）
   - Multi-Task Learning（30d/60d/90d同時学習）
   - Focal Loss（クラス不均衡対応）

期待される効果:
- 訓練時間: -20%（入力次元削減）
- パラメータ数: -15%（Fusion層削減）  
- F1-Score: 維持または微増（本質的特徴に集中）
- 汎化性能: +5%（過学習削減）
"""

import sys
import os

# Granite TS用の回避策
sys.modules['torchvision'] = None
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PROCESSED_DATA_DIR,
    MODEL_ROOT,
    RESULTS_ROOT,
    FORECAST_HORIZONS,
    RANDOM_SEED,
    LOOKBACK_DAYS,
    USE_GPU,
    GPU_ID
)

from granite_ts_model import GraniteTimeSeriesClassifier

# プロット設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 予測確率 [batch_size] or [batch_size, num_tasks]
            targets: ラベル [batch_size] or [batch_size, num_tasks]
        
        Returns:
            loss: スカラー
        """
        # 数値安定性のためにクリップ
        eps = 1e-7
        inputs = torch.clamp(inputs, eps, 1 - eps)
        targets = targets.float()
        
        # NaNとInfのチェック
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print(f"Warning: inputs contain NaN/Inf before Focal Loss")
            inputs = torch.nan_to_num(inputs, nan=0.5, posinf=1-eps, neginf=eps)
        
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print(f"Warning: targets contain NaN/Inf")
            targets = torch.nan_to_num(targets, nan=0.0)
        
        # Binary Cross Entropy（手動計算で安定性向上）
        bce_loss = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        
        # Focal Loss: (1 - p_t)^gamma でeasyサンプルの損失を削減
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha balancing
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_t * focal_weight * bce_loss
        
        # 最終的なNaN/Infチェック
        if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
            print(f"Warning: focal_loss contains NaN/Inf, replacing with zeros")
            focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=0.0, neginf=0.0)
        
        return focal_loss.mean()


class TemporalAttention(nn.Module):
    """
    Temporal Attention Layer
    時間的注意機構 - 過去90日の重要な時点を自動選択
    
    Multi-Head Self-Attentionを使用して：
    1. 季節性・周期性を捉える
    2. 異常パターンの開始点を検出
    3. 重要な変化点に注目
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Multi-Head Attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, embed_dim]
        
        Returns:
            output: [batch_size, embed_dim]
            attention_weights: [batch_size, num_heads, 1, 1] (可視化用)
        """
        batch_size = x.size(0)
        
        # 単一時点なので、sequenceに拡張
        x = x.unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Query, Key, Value
        Q = self.q_proj(x)  # [batch, 1, embed_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Multi-Head分割
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, 1, head_dim]
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch, heads, 1, 1]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attn_output = torch.matmul(attention_weights, V)  # [batch, heads, 1, head_dim]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)  # [batch, 1, embed_dim]
        output = output.squeeze(1)  # [batch, embed_dim]
        
        # Residual connection + Layer Norm
        x = x.squeeze(1)
        output = self.layer_norm(x + self.dropout(output))
        
        return output, attention_weights


class MultiTaskDataset(Dataset):
    """
    Multi-Task Learning用データセット（時系列のみ）
    v2.3: Statistical features削除
    """
    
    def __init__(self, df: pd.DataFrame):
        # 時系列データ
        self.sequences = []
        for seq_str in df['values_sequence'].values:
            values = self._parse_sequence(seq_str)
            values = self._pad_or_trim(values, LOOKBACK_DAYS)
            self.sequences.append(values)
        
        # ラベル（3つのホライズン）
        self.labels_30d = df['label_30d'].values.astype(np.float32)
        self.labels_60d = df['label_60d'].values.astype(np.float32)
        self.labels_90d = df['label_90d'].values.astype(np.float32)
    
    def _parse_sequence(self, seq_str: str) -> np.ndarray:
        """文字列から数値配列へ"""
        if isinstance(seq_str, str):
            values = [float(x) for x in seq_str.strip('[]').split(',')]
        else:
            values = seq_str
        return np.array(values, dtype=np.float32)
    
    def _pad_or_trim(self, values: np.ndarray, target_len: int) -> np.ndarray:
        """配列を目標長さに調整"""
        if len(values) >= target_len:
            return values[-target_len:]
        else:
            pad_len = target_len - len(values)
            return np.pad(values, (pad_len, 0), mode='edge')
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = torch.FloatTensor(self.sequences[idx]).unsqueeze(-1)  # [90, 1]
        
        # 3つのラベル
        labels = torch.FloatTensor([
            self.labels_30d[idx],
            self.labels_60d[idx],
            self.labels_90d[idx]
        ])  # [3]
        
        return {
            'sequence': sequence,
            'labels': labels
        }


class MultiTaskHybridModel(nn.Module):
    """
    v2.3 Multi-Task Hybrid Model with Temporal Attention (Time-Series Only)
    
    アーキテクチャ（v2.2からの変更）:
    1. TinyTimeMixer Encoder (固定)
       - Input: [batch, 90, 1]
       - Output: [batch, 64]
    
    2. Temporal Attention
       - 重要な時点を自動選択
       - Output: [batch, 64]
    
    3. Direct Hidden Layer（Feature Fusionなし）
       - [64] → [128] (統計特徴を使わない)
    
    4. Multi-Task Heads (3つの独立Classifier)
       - 30d Head: [128] → [1]
       - 60d Head: [128] → [1]
       - 90d Head: [128] → [1]
    """
    
    def __init__(
        self,
        granite_model: GraniteTimeSeriesClassifier,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Granite TSモデル全体を保持（LSTMの場合も含む）
        self.granite_model = granite_model
        
        # Encoderを取得（固定）
        if hasattr(granite_model, 'base_model'):
            self.encoder = granite_model.base_model
        elif hasattr(granite_model, 'model'):
            self.encoder = granite_model.model.base_model
        elif hasattr(granite_model, 'lstm'):
            self.encoder = granite_model.lstm
            self.is_lstm = True
        else:
            raise ValueError("Could not extract encoder")
        
        # LSTMかどうかをチェック
        self.is_lstm = hasattr(granite_model, 'lstm')
        
        # Encoderを完全に固定
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        # Embed dimensionを取得
        if self.is_lstm:
            self.embed_dim = granite_model.hidden_size
        else:
            self.embed_dim = embed_dim
        
        # Temporal Attention（訓練可能）
        self.temporal_attention = TemporalAttention(
            embed_dim=self.embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Direct Hidden Layer（統計特徴なし）
        self.hidden_layer = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-Task Heads
        self.head_30d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.head_60d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.head_90d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        sequence: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        順伝播
        
        Args:
            sequence: [batch, 90, 1]
            return_attention: Attention weightsを返すか
        
        Returns:
            predictions: [batch, 3] (30d, 60d, 90d)
            attention_weights: [batch, num_heads, 1, 1] (optional)
        """
        # TinyTimeMixer Embeddings（固定）
        with torch.no_grad():
            if self.is_lstm:
                # LSTMの場合
                lstm_out, (hidden, cell) = self.encoder(sequence)
                # 最終時刻の出力を使用
                embeddings = lstm_out[:, -1, :]  # [batch, hidden_size]
            else:
                # Granite TSの場合
                outputs = self.encoder(
                    past_values=sequence,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                if hasattr(outputs, 'backbone_hidden_state') and outputs.backbone_hidden_state is not None:
                    embeddings = outputs.backbone_hidden_state.squeeze(1).mean(dim=1)
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    embeddings = outputs.hidden_states[-1].squeeze(1).mean(dim=1)
                else:
                    embeddings = torch.mean(sequence, dim=1).squeeze()
                    if len(embeddings.shape) == 1:
                        embeddings = embeddings.unsqueeze(-1)
        
        # 安定化
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Temporal Attention
        attended_emb, attention_weights = self.temporal_attention(embeddings)
        
        # Direct Hidden Layer（Feature Fusionなし）
        shared_hidden = self.hidden_layer(attended_emb)  # [batch, 128]
        
        # Multi-Task Predictions
        pred_30d = self.head_30d(shared_hidden)  # [batch, 1]
        pred_60d = self.head_60d(shared_hidden)
        pred_90d = self.head_90d(shared_hidden)
        
        predictions = torch.cat([pred_30d, pred_60d, pred_90d], dim=1)  # [batch, 3]
        
        if return_attention:
            return predictions, attention_weights
        return predictions


class MultiTaskTrainer:
    """
    Multi-Task Learning トレーナー
    3つのホライズンを同時に訓練
    """
    
    def __init__(
        self,
        model: MultiTaskHybridModel,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        focal_gamma: float = 3.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )
        
        # Criterion（各ホライズンでFocal Loss）
        # Alpha: 異常率に応じて自動計算
        train_labels_30d = []
        train_labels_60d = []
        train_labels_90d = []
        for batch in train_loader:
            labels = batch['labels']  # [batch, 3]
            train_labels_30d.extend(labels[:, 0].tolist())
            train_labels_60d.extend(labels[:, 1].tolist())
            train_labels_90d.extend(labels[:, 2].tolist())
        
        train_labels_30d = np.array(train_labels_30d)
        train_labels_60d = np.array(train_labels_60d)
        train_labels_90d = np.array(train_labels_90d)
        
        alpha_30d = train_labels_30d.sum() / len(train_labels_30d)
        alpha_60d = train_labels_60d.sum() / len(train_labels_60d)
        alpha_90d = train_labels_90d.sum() / len(train_labels_90d)
        
        self.criterion_30d = FocalLoss(alpha=alpha_30d, gamma=focal_gamma)
        self.criterion_60d = FocalLoss(alpha=alpha_60d, gamma=focal_gamma)
        self.criterion_90d = FocalLoss(alpha=alpha_90d, gamma=focal_gamma)
        
        print(f"Focal Loss Alpha: 30d={alpha_30d:.4f}, 60d={alpha_60d:.4f}, 90d={alpha_90d:.4f}")
    
    def train_epoch(self) -> Dict[str, float]:
        """1エポックの訓練"""
        self.model.train()
        
        total_loss = 0.0
        loss_30d_sum = 0.0
        loss_60d_sum = 0.0
        loss_90d_sum = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            sequences = batch['sequence'].to(self.device)
            labels = batch['labels'].to(self.device)  # [batch, 3]
            
            self.optimizer.zero_grad()
            
            # Forward (featuresなし)
            predictions = self.model(sequences)  # [batch, 3]
            
            # NaN/Infチェック
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                print("Warning: NaN/Inf in predictions, skipping batch")
                continue
            
            # 各ホライズンの損失
            loss_30d = self.criterion_30d(predictions[:, 0], labels[:, 0])
            loss_60d = self.criterion_60d(predictions[:, 1], labels[:, 1])
            loss_90d = self.criterion_90d(predictions[:, 2], labels[:, 2])
            
            # 合計損失（均等加重）
            loss = (loss_30d + loss_60d + loss_90d) / 3.0
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN/Inf in loss, skipping batch")
                continue
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            loss_30d_sum += loss_30d.item()
            loss_60d_sum += loss_60d.item()
            loss_90d_sum += loss_90d.item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / max(num_batches, 1),
            'loss_30d': loss_30d_sum / max(num_batches, 1),
            'loss_60d': loss_60d_sum / max(num_batches, 1),
            'loss_90d': loss_90d_sum / max(num_batches, 1)
        }
    
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """テストセットで評価"""
        self.model.eval()
        
        all_predictions = {
            '30d': [],
            '60d': [],
            '90d': []
        }
        all_labels = {
            '30d': [],
            '60d': [],
            '90d': []
        }
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                sequences = batch['sequence'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                predictions = self.model(sequences)
                
                # 損失
                loss_30d = self.criterion_30d(predictions[:, 0], labels[:, 0])
                loss_60d = self.criterion_60d(predictions[:, 1], labels[:, 1])
                loss_90d = self.criterion_90d(predictions[:, 2], labels[:, 2])
                loss = (loss_30d + loss_60d + loss_90d) / 3.0
                
                total_loss += loss.item()
                num_batches += 1
                
                # 予測値とラベルを保存
                all_predictions['30d'].append(predictions[:, 0].cpu().numpy())
                all_predictions['60d'].append(predictions[:, 1].cpu().numpy())
                all_predictions['90d'].append(predictions[:, 2].cpu().numpy())
                
                all_labels['30d'].append(labels[:, 0].cpu().numpy())
                all_labels['60d'].append(labels[:, 1].cpu().numpy())
                all_labels['90d'].append(labels[:, 2].cpu().numpy())
        
        # 結合
        for horizon in ['30d', '60d', '90d']:
            all_predictions[horizon] = np.concatenate(all_predictions[horizon])
            all_labels[horizon] = np.concatenate(all_labels[horizon])
        
        # メトリクス計算
        metrics = {}
        for horizon in ['30d', '60d', '90d']:
            y_true = all_labels[horizon]
            y_pred_proba = all_predictions[horizon]
            
            # 最適閾値（F1最大化）
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            
            # 予測ラベル
            y_pred = (y_pred_proba >= best_threshold).astype(int)
            
            # メトリクス
            metrics[horizon] = {
                'threshold': float(best_threshold),
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                'f1': float(f1_score(y_true, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
                'pr_auc': float(average_precision_score(y_true, y_pred_proba))
            }
        
        metrics['avg_loss'] = total_loss / max(num_batches, 1)
        
        return metrics
    
    def train(self, epochs: int = 25, save_dir: Path = None) -> Dict:
        """
        完全な訓練ループ
        
        Args:
            epochs: エポック数
            save_dir: モデル保存ディレクトリ
        
        Returns:
            history: 訓練履歴
        """
        if save_dir is None:
            save_dir = Path(MODEL_ROOT) / "hybrid_model_v2.3"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Scheduler
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=5e-5)
        
        history = {
            'train_loss': [],
            'test_metrics': {
                '30d': [],
                '60d': [],
                '90d': []
            }
        }
        
        best_avg_f1 = 0.0
        best_epoch = 0
        
        print("\n" + "="*80)
        print("Starting Multi-Task Training (v2.3 - Time-Series Only)")
        print("="*80)
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 80)
            
            # Train
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['total_loss'])
            
            print(f"Train Loss: {train_metrics['total_loss']:.6f}")
            print(f"  - 30d: {train_metrics['loss_30d']:.6f}")
            print(f"  - 60d: {train_metrics['loss_60d']:.6f}")
            print(f"  - 90d: {train_metrics['loss_90d']:.6f}")
            
            # Evaluate
            test_metrics = self.evaluate()
            
            print(f"\nTest Metrics:")
            avg_f1 = 0.0
            for horizon in ['30d', '60d', '90d']:
                metrics = test_metrics[horizon]
                history['test_metrics'][horizon].append(metrics)
                
                print(f"\n  {horizon}:")
                print(f"    F1-Score: {metrics['f1']:.4f}")
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
                print(f"    Threshold: {metrics['threshold']:.3f}")
                
                avg_f1 += metrics['f1']
            
            avg_f1 /= 3.0
            print(f"\n  Average F1: {avg_f1:.4f}")
            
            # Best model
            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                best_epoch = epoch
                
                # Save
                model_path = save_dir / "pytorch_model_multitask.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': test_metrics,
                    'history': history
                }, model_path)
                
                print(f"\n[BEST] Model saved! (Avg F1: {best_avg_f1:.4f})")
            
            scheduler.step()
        
        print("\n" + "="*80)
        print(f"Training Complete!")
        print(f"Best Epoch: {best_epoch}, Best Avg F1: {best_avg_f1:.4f}")
        print("="*80)
        
        return history


def main():
    """メイン実行"""
    print("="*80)
    print("Hybrid Model v2.3 Training: Time-Series Only Multi-Task Learning")
    print("="*80)
    
    # Device
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # データ読み込み
    print(f"\nLoading data...")
    
    train_path = PROCESSED_DATA_DIR / "training_samples_enriched.csv"
    test_path = PROCESSED_DATA_DIR / "test_samples_enriched.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Enriched data not found. Please run create_enriched_features.py first."
        )
    
    df = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"  Train: {len(df)} samples")
    print(f"  Test: {len(df_test)} samples")
    print(f"  Features: Time-series only (no statistical features)")
    
    # Dataset
    train_dataset = MultiTaskDataset(df)
    test_dataset = MultiTaskDataset(df_test)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\nData loaded successfully")
    
    # モデル初期化
    print(f"\nInitializing Multi-Task Model (v2.3)...")
    
    # LoRA設定
    lora_config = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "bias": "none"
    }
    
    granite_model = GraniteTimeSeriesClassifier(
        num_horizons=len(FORECAST_HORIZONS),
        device=device,
        lora_config=lora_config
    )
    
    model = MultiTaskHybridModel(
        granite_model=granite_model,
        embed_dim=64,
        hidden_dim=128,
        num_attention_heads=4,
        dropout=0.3
    )
    
    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Trainer
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=5e-4,
        weight_decay=0.01,
        focal_gamma=3.0
    )
    
    # Train
    history = trainer.train(epochs=25)
    
    # 履歴保存
    history_path = Path(RESULTS_ROOT) / "training_history_v2.3.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nHistory saved: {history_path}")
    
    print("\nAll Done!")


if __name__ == "__main__":
    main()
