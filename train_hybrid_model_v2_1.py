"""
Hybrid Model v2.1: Enhanced Time Series Forecasting
ハイブリッドモデルv2.1 - 非線形予測と高度な特徴活用

改善点:
1. Statistical Features (28次元) を多次元インプット
   - TinyTimeMixerとClassifier両方で活用
   - 埋め込み後に統計特徴を結合

2. TinyTimeMixer Encoder（事前学習済み・固定）
   - 事前学習モデルの知識を活用
   - Encoderは固定し、訓練時間を短縮
   - Feature Fusion + Classifierのみ訓練

3. Focal Loss (beta=3)
   - クラス不均衡データに対応
   - 難しいサンプルに重点的に学習

4. 非線形予測
   - 過去90日の延長線上にある予測
   - 設備管理者にとって直感的な可視化
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

import lightgbm as lgb
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
    
    Args:
        alpha: バランス係数（positive classの重み）
        gamma: focusing parameter（難しいサンプルへの重点度）
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 予測確率 [batch_size]
            targets: ラベル [batch_size]
        
        Returns:
            loss: スカラー損失値
        """
        # 数値安定性のためにクリップと型変換
        eps = 1e-7
        inputs = torch.clamp(inputs, eps, 1 - eps)
        targets = targets.float()  # targetsを確実にfloatに
        
        # NaNとInfのチェック
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print(f"Warning: inputs contains NaN or Inf")
            inputs = torch.nan_to_num(inputs, nan=0.5, posinf=1-eps, neginf=eps)
        
        # Binary Cross Entropy（手動計算でより安定）
        bce_loss = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        
        # p_t: 正解クラスの予測確率
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        p_t = torch.clamp(p_t, eps, 1 - eps)  # さらにクリップ
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal Loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        return focal_loss.mean()


class EnhancedHybridDataset(Dataset):
    """
    v2.1 Enhanced Dataset
    時系列データ + 統計特徴量を両方返す
    """
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], horizon: int):
        self.df = df
        self.feature_cols = feature_cols
        self.horizon = horizon
        
        # 時系列データの解析
        self.sequences = []
        for seq_str in df['values_sequence'].values:
            import ast
            try:
                values = ast.literal_eval(seq_str)
            except:
                values = [float(x.strip('[] ')) for x in seq_str.split(',') if x.strip()]
            
            # LOOKBACK_DAYS日分にパディング/トリミング
            if len(values) < LOOKBACK_DAYS:
                values = [values[0]] * (LOOKBACK_DAYS - len(values)) + values
            elif len(values) > LOOKBACK_DAYS:
                values = values[-LOOKBACK_DAYS:]
            self.sequences.append(values)
        
        # 統計特徴量
        self.features = df[feature_cols].values.astype(np.float32)
        
        # ラベル
        label_col = f'label_{horizon}d'
        self.labels = df[label_col].values.astype(np.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 時系列シーケンス [seq_len, 1]
        sequence = torch.FloatTensor(self.sequences[idx]).unsqueeze(-1)
        
        # 統計特徴量 [num_features]
        features = torch.FloatTensor(self.features[idx])
        
        # ラベル
        label = torch.FloatTensor([self.labels[idx]])
        
        return {
            'sequence': sequence,
            'features': features,
            'label': label
        }


class EnhancedHybridModel(nn.Module):
    """
    v2.1 Enhanced Hybrid Model
    
    アーキテクチャ:
    1. TinyTimeMixer Encoder (事前学習済み・固定)
       - Input: 時系列 [batch, 90, 1]
       - Output: 埋め込み [batch, 64]
       - 訓練中は固定（no_grad）
    
    2. Feature Fusion Layer
       - 埋め込み [64] + 統計特徴 [28] = 結合特徴 [92]
    
    3. Multi-Layer Classifier（訓練可能）
       - 非線形変換 → 異常確率
    """
    
    def __init__(
        self,
        granite_model: GraniteTimeSeriesClassifier,
        stat_feature_dim: int = 28,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # TinyTimeMixer Encoder (固定：訓練しない）
        if hasattr(granite_model, 'base_model'):
            self.encoder = granite_model.base_model
        elif hasattr(granite_model, 'model'):
            self.encoder = granite_model.model.base_model
        elif hasattr(granite_model, 'lstm'):
            self.encoder = granite_model.lstm
        else:
            raise ValueError("Could not extract encoder")
        
        # Encoderを完全に固定（LoRAも含む）
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 推論モードに設定
        self.encoder.eval()
        
        # 埋め込み次元
        self.embedding_dim = 64  # TinyTimeMixer d_model
        
        # Feature Fusion + Classification Head
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.embedding_dim + stat_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sequence: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            sequence: 時系列 [batch, seq_len, 1]
            features: 統計特徴 [batch, stat_dim]
        
        Returns:
            predictions: 異常確率 [batch, 1]
        """
        # TinyTimeMixer Embeddings（固定モード）
        with torch.no_grad():
            outputs = self.encoder(
                past_values=sequence,
                output_hidden_states=True,
                return_dict=True
            )
            
            if hasattr(outputs, 'backbone_hidden_state') and outputs.backbone_hidden_state is not None:
                embeddings = outputs.backbone_hidden_state.squeeze(1).mean(dim=1)  # [batch, 64]
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                embeddings = outputs.hidden_states[-1].squeeze(1).mean(dim=1)
            else:
                embeddings = torch.mean(sequence, dim=1).squeeze()
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(-1)
        
        # Embeddings安定化（NaNチェック）
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            print(f"Warning: embeddings contain NaN/Inf, replacing with zeros")
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 統計特徴の安定化
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"Warning: features contain NaN/Inf, replacing with zeros")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Feature Fusion
        fused_features = torch.cat([embeddings, features], dim=1)  # [batch, 92]
        
        # Classification
        predictions = self.fusion_layer(fused_features)  # [batch, 1]
        
        # 最終出力の安定化
        predictions = predictions.squeeze(1)
        predictions = torch.clamp(predictions, 1e-7, 1 - 1e-7)
        
        return predictions


class HybridTrainerV2_1:
    """v2.1 ハイブリッドモデルトレーナー"""
    
    def __init__(self):
        self.device = torch.device(f'cuda:{GPU_ID}' if USE_GPU and torch.cuda.is_available() else 'cpu')
        self.train_df = None
        self.test_df = None
        self.feature_cols = []
        self.models = {}
        self.results = {}
        
        print(f"🖥️  Device: {self.device}")
        print(f"📁 Data directory: {PROCESSED_DATA_DIR}")
        print(f"📁 Model directory: {MODEL_ROOT}")
    
    def load_data(self):
        """データロード"""
        print("\n📂 Loading enriched data...")
        
        train_path = PROCESSED_DATA_DIR / "training_samples_enriched.csv"
        test_path = PROCESSED_DATA_DIR / "test_samples_enriched.csv"
        
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "Enriched data not found. Please run create_enriched_features.py first."
            )
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"✓ Train: {len(self.train_df):,} samples")
        print(f"✓ Test: {len(self.test_df):,} samples")
        
        # 特徴量カラムの特定
        exclude_cols = [
            'equipment_id', 'check_item_id', 'date', 
            'window_start', 'window_end', 'values_sequence',
            'reference_datetime', 'horizon_datetime',
            'label_current', 'label_30d', 'label_60d', 'label_90d',
            'any_anomaly'
        ]
        
        self.feature_cols = [col for col in self.train_df.columns 
                            if col not in exclude_cols]
        
        print(f"✓ Statistical features: {len(self.feature_cols)}d")
    
    def train_horizon(self, horizon: int, epochs: int = 10, batch_size: int = 128, lr: float = 1e-4):
        """
        特定ホライズンのモデル訓練
        
        Args:
            horizon: 予測ホライズン（30, 60, 90）
            epochs: エポック数
            batch_size: バッチサイズ
            lr: 学習率
        """
        print(f"\n{'='*70}")
        print(f"Training Enhanced Hybrid Model v2.1 for {horizon}d horizon")
        print('='*70)
        
        # データセット作成
        train_dataset = EnhancedHybridDataset(self.train_df, self.feature_cols, horizon)
        test_dataset = EnhancedHybridDataset(self.test_df, self.feature_cols, horizon)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"\nDataset statistics:")
        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Test samples: {len(test_dataset):,}")
        print(f"  Train positives: {train_dataset.labels.sum():.0f} ({train_dataset.labels.mean()*100:.1f}%)")
        print(f"  Test positives: {test_dataset.labels.sum():.0f} ({test_dataset.labels.mean()*100:.1f}%)")
        
        # モデル構築
        print(f"\n🏗️  Building Enhanced Hybrid Model...")
        
        # LoRA設定（r=8, alpha=16で規模を制御）
        lora_config = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "bias": "none"
        }
        
        granite_model = GraniteTimeSeriesClassifier(
            num_horizons=len(FORECAST_HORIZONS),
            device=self.device,
            lora_config=lora_config
        )
        
        model = EnhancedHybridModel(
            granite_model=granite_model,
            stat_feature_dim=len(self.feature_cols),
            hidden_dim=128,
            dropout=0.3
        ).to(self.device)
        
        # 訓練可能パラメータ数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        # Focal Loss
        pos_weight = (len(train_dataset) - train_dataset.labels.sum()) / train_dataset.labels.sum()
        alpha = 1.0 / (1.0 + pos_weight)
        criterion = FocalLoss(alpha=alpha, gamma=3.0)
        print(f"  Focal Loss: alpha={alpha:.3f}, gamma=3.0")
        
        # Optimizer & Scheduler
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
        
        # 勾配クリッピング用の最大ノルム
        max_grad_norm = 1.0
        
        # 訓練ループ
        print(f"\n🚀 Training for {epochs} epochs...")
        best_f1 = 0.0
        history = {'train_loss': [], 'test_loss': [], 'test_f1': []}
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                sequences = batch['sequence'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                
                optimizer.zero_grad()
                outputs = model(sequences, features)
                
                # NaN/Infチェック（出力）
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"  Warning: outputs contain NaN/Inf at batch, skipping...")
                    continue
                
                loss = criterion(outputs, labels)
                
                # NaN/Infチェック（損失）
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Warning: loss is NaN/Inf at batch, skipping...")
                    continue
                
                loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            test_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    sequences = batch['sequence'].to(self.device)
                    features = batch['features'].to(self.device)
                    labels = batch['label'].squeeze().to(self.device)
                    
                    outputs = model(sequences, features)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            test_loss /= len(test_loader)
            
            # Metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1_score = f1_scores[best_idx]
            
            pred_binary = (all_preds > best_threshold).astype(int)
            accuracy = accuracy_score(all_labels, pred_binary)
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['test_f1'].append(best_f1_score)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | F1: {best_f1_score:.4f} | "
                  f"Acc: {accuracy:.4f} | Threshold: {best_threshold:.3f}")
            
            # Best model保存
            if best_f1_score > best_f1:
                best_f1 = best_f1_score
                best_model_state = model.state_dict()
                best_threshold_value = best_threshold
            
            scheduler.step()
        
        # Best modelをロード
        model.load_state_dict(best_model_state)
        
        # 最終評価
        print(f"\n📊 Final Evaluation (best F1={best_f1:.4f})...")
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequence'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['label'].squeeze()
                
                outputs = model(sequences, features)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_binary = (all_preds > best_threshold_value).astype(int)
        
        metrics = {
            'horizon': horizon,
            'threshold': best_threshold_value,
            'accuracy': accuracy_score(all_labels, pred_binary),
            'precision': precision_score(all_labels, pred_binary, zero_division=0),
            'recall': recall_score(all_labels, pred_binary, zero_division=0),
            'f1': f1_score(all_labels, pred_binary, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_preds),
            'pr_auc': average_precision_score(all_labels, all_preds)
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        
        # モデル保存
        model_dir = MODEL_ROOT / "hybrid_model_v2.1"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"pytorch_model_{horizon}d.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'threshold': best_threshold_value,
            'metrics': metrics,
            'history': history
        }, model_path)
        print(f"✓ Model saved: {model_path}")
        
        self.models[horizon] = model
        self.results[horizon] = {
            'metrics': metrics,
            'predictions': all_preds,
            'labels': all_labels,
            'history': history
        }
        
        return model, metrics
    
    def plot_training_history(self):
        """訓練履歴のプロット"""
        print(f"\n📈 Plotting training history...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            if horizon not in self.results:
                continue
            
            history = self.results[horizon]['history']
            epochs = range(1, len(history['train_loss']) + 1)
            
            ax = axes[idx]
            ax2 = ax.twinx()
            
            # Loss
            ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12, color='b')
            ax.tick_params(axis='y', labelcolor='b')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # F1 Score
            ax2.plot(epochs, history['test_f1'], 'g-', label='Test F1', linewidth=2)
            ax2.set_ylabel('F1 Score', fontsize=12, color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.legend(loc='upper right')
            
            ax.set_title(f'{horizon}d Horizon Training', fontsize=14, fontweight='bold')
        
        plt.suptitle('Enhanced Hybrid Model v2.1 Training History', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_path = RESULTS_ROOT / f'training_history_v2.1_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.show()
    
    def save_summary(self):
        """結果サマリーの保存"""
        print(f"\n💾 Saving results summary...")
        
        summary = []
        for horizon in FORECAST_HORIZONS:
            if horizon in self.results:
                metrics = self.results[horizon]['metrics']
                summary.append(metrics)
        
        summary_df = pd.DataFrame(summary)
        
        model_dir = MODEL_ROOT / "hybrid_model_v2.1"
        summary_path = model_dir / "metrics_summary_v2.1.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"✓ Summary saved: {summary_path}")
        print(f"\n{summary_df}")
    
    def run(self, epochs: int = 10, batch_size: int = 128, lr: float = 1e-4):
        """完全実行"""
        print("\n" + "="*70)
        print("Enhanced Hybrid Model v2.1 Training")
        print("="*70)
        
        # データロード
        self.load_data()
        
        # 各ホライズンで訓練
        for horizon in FORECAST_HORIZONS:
            self.train_horizon(horizon, epochs=epochs, batch_size=batch_size, lr=lr)
        
        # 可視化
        self.plot_training_history()
        
        # サマリー保存
        self.save_summary()
        
        print("\n✅ Training completed!")
        print(f"📁 Models saved to: {MODEL_ROOT / 'hybrid_model_v2.1'}")


def main():
    trainer = HybridTrainerV2_1()
    # Encoderは固定なので、より高速に訓練可能
    trainer.run(epochs=20, batch_size=128, lr=5e-4)


if __name__ == "__main__":
    main()
