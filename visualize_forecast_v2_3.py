"""
Forecast Visualization for v2.3 Multi-Task Model (Time-Series Only)
v2.3マルチタスクモデルの予測可視化（時系列データのみ）

v2.2からの変更点:
- Statistical features (28次元) を削除
- TinyTimeMixer embeddings (64次元) のみを使用
- よりシンプルな予測パイプライン

機能:
1. 3つのホライズン（30d、60d、90d）を同時表示
2. Attention weightsの可視化
3. 非線形予測の生成  
4. 多様なサンプル選択（TN, TP, FP, FN）
"""

import sys
import os
sys.modules['torchvision'] = None
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import (
    PROCESSED_DATA_DIR,
    MODEL_ROOT,
    RESULTS_ROOT,
    FORECAST_HORIZONS,
    LOOKBACK_DAYS
)

from granite_ts_model import GraniteTimeSeriesClassifier

# プロット設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class TemporalAttention(nn.Module):
    """Temporal Attention Layer（推論用）"""
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attn_output = torch.matmul(attention_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.embed_dim)
        output = self.out_proj(attn_output)
        output = output.squeeze(1)
        
        x = x.squeeze(1)
        output = self.layer_norm(x + self.dropout(output))
        
        return output, attention_weights


class MultiTaskHybridModel(nn.Module):
    """v2.3 Multi-Task Model（推論用、時系列のみ）"""
    
    def __init__(
        self,
        granite_model: GraniteTimeSeriesClassifier,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        if hasattr(granite_model, 'base_model'):
            self.encoder = granite_model.base_model
        elif hasattr(granite_model, 'model'):
            self.encoder = granite_model.model.base_model
        elif hasattr(granite_model, 'lstm'):
            self.encoder = granite_model.lstm
            self.is_lstm = True
        else:
            raise ValueError("Could not extract encoder")
        
        self.is_lstm = hasattr(granite_model, 'lstm')
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        if self.is_lstm:
            self.embed_dim = granite_model.hidden_size
        else:
            self.embed_dim = embed_dim
        
        self.temporal_attention = TemporalAttention(
            embed_dim=self.embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Direct Hidden Layer（Statistical features なし）
        self.hidden_layer = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
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
    
    def forward(self, sequence: torch.Tensor, return_attention: bool = False):
        with torch.no_grad():
            if self.is_lstm:
                lstm_out, (hidden, cell) = self.encoder(sequence)
                embeddings = lstm_out[:, -1, :]
            else:
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
        
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        attended_emb, attention_weights = self.temporal_attention(embeddings)
        
        # Direct Hidden Layer
        shared_hidden = self.hidden_layer(attended_emb)
        
        pred_30d = self.head_30d(shared_hidden)
        pred_60d = self.head_60d(shared_hidden)
        pred_90d = self.head_90d(shared_hidden)
        
        predictions = torch.cat([pred_30d, pred_60d, pred_90d], dim=1)
        
        if return_attention:
            return predictions, attention_weights
        return predictions


class ForecastVisualizerV2_3:
    """v2.3モデルの予測可視化（時系列のみ）"""
    
    def __init__(self, model_path: Path, device: torch.device):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.thresholds = {}
    
    def load_model(self):
        """モデル読み込み"""
        print(f"Loading v2.3 Multi-Task Model (Time-Series Only)...")
        
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
        
        self.model = MultiTaskHybridModel(
            granite_model=granite_model,
            embed_dim=64,
            hidden_dim=128,
            num_attention_heads=4,
            dropout=0.3
        )
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        metrics = checkpoint['metrics']
        self.thresholds['30d'] = metrics['30d']['threshold']
        self.thresholds['60d'] = metrics['60d']['threshold']
        self.thresholds['90d'] = metrics['90d']['threshold']
        
        print(f"  [OK] Model loaded")
        print(f"  [OK] 30d: F1={metrics['30d']['f1']:.4f}, Threshold={self.thresholds['30d']:.3f}")
        print(f"  [OK] 60d: F1={metrics['60d']['f1']:.4f}, Threshold={self.thresholds['60d']:.3f}")
        print(f"  [OK] 90d: F1={metrics['90d']['f1']:.4f}, Threshold={self.thresholds['90d']:.3f}")
    
    def predict_sample(
        self,
        sequence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        単一サンプルの予測（時系列のみ）
        
        Returns:
            predictions: [3] (30d, 60d, 90d probabilities)
            pred_labels: [3] (binary predictions)
            attention_weights: [num_heads, 1, 1]
        """
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            predictions, attention_weights = self.model(sequence_tensor, return_attention=True)
        
        predictions = predictions.cpu().numpy()[0]
        
        pred_labels = np.array([
            1 if predictions[0] >= self.thresholds['30d'] else 0,
            1 if predictions[1] >= self.thresholds['60d'] else 0,
            1 if predictions[2] >= self.thresholds['90d'] else 0
        ])
        
        attention_weights = attention_weights.cpu().numpy()[0]
        
        return predictions, pred_labels, attention_weights
    
    def generate_simple_forecast(
        self,
        sequence: np.ndarray,
        horizon_days: int,
        num_points: int = 30
    ) -> np.ndarray:
        """
        シンプルな非線形予測生成（統計特徴なし）
        時系列パターンのみから予測
        """
        recent_values = sequence[-30:]
        
        # 2次多項式フィット
        x = np.arange(len(recent_values))
        try:
            coeffs = np.polyfit(x, recent_values, 2)
            trend = np.poly1d(coeffs)
        except:
            trend = lambda x: np.mean(recent_values)
        
        # 基本予測
        forecast_x = np.arange(len(recent_values), len(recent_values) + num_points)
        base_forecast = trend(forecast_x)
        
        # 時系列統計
        recent_mean = np.mean(sequence[-horizon_days:])
        recent_std = np.std(sequence[-horizon_days:])
        
        # 減衰係数（平均回帰）
        decay_factor = np.exp(-forecast_x / (horizon_days * 2))
        adjusted_forecast = (base_forecast - recent_mean) * decay_factor + recent_mean
        
        # 物理的制約
        overall_mean = np.mean(sequence)
        overall_std = np.std(sequence)
        adjusted_forecast = np.clip(adjusted_forecast, 
                                    overall_mean - 3*overall_std, 
                                    overall_mean + 3*overall_std)
        
        return adjusted_forecast
    
    def plot_multitask_comparison(
        self,
        test_df: pd.DataFrame,
        num_samples: int = 5
    ):
        """
        マルチタスク予測の可視化
        3つのホライズンを3列で表示
        """
        sample_indices = self._select_diverse_samples(test_df, num_samples)
        
        print(f"\nSelected samples: {sample_indices}")
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(18, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, sample_idx in enumerate(sample_indices):
            row_data = test_df.iloc[sample_idx]
            
            sequence = self._parse_sequence(row_data['values_sequence'])
            
            predictions, pred_labels, attention_weights = self.predict_sample(sequence)
            
            horizons = [
                ('30d', predictions[0], pred_labels[0], 
                 int(row_data['label_30d']), 30, axes[idx, 0]),
                ('60d', predictions[1], pred_labels[1], 
                 int(row_data['label_60d']), 60, axes[idx, 1]),
                ('90d', predictions[2], pred_labels[2], 
                 int(row_data['label_90d']), 90, axes[idx, 2])
            ]
            
            for h_name, pred_prob, pred_label, actual_label, h_days, ax in horizons:
                
                # 過去90日
                time_past = np.arange(-LOOKBACK_DAYS, 0)
                ax.plot(time_past, sequence, 'b-', linewidth=1.5, label='Past 90 days', alpha=0.7)
                
                # 非線形予測
                forecast = self.generate_simple_forecast(sequence, h_days, num_points=30)
                time_future = np.arange(0, len(forecast))
                
                color = 'red' if pred_label == 1 else 'green'
                linestyle = '--'
                ax.plot(time_future, forecast, color=color, linestyle=linestyle, 
                       linewidth=2, label=f'Forecast ({h_name})', alpha=0.8)
                
                # 異常期間の背景
                if actual_label == 1:
                    ax.axvspan(0, h_days, alpha=0.15, color='red', label='Actual Anomaly Period')
                
                # 予測開始点
                ax.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
                
                # タイトル
                result = "[OK]" if pred_label == actual_label else "[X]"
                ax.set_title(
                    f'{h_name} | Pred: {pred_prob:.3f} ({"Anomaly" if pred_label==1 else "Normal"}) | '
                    f'Actual: {"Anomaly" if actual_label==1 else "Normal"} {result}',
                    fontsize=10,
                    fontweight='bold'
                )
                
                ax.set_xlabel('Days', fontsize=9)
                ax.set_ylabel('Value', fontsize=9)
                ax.legend(fontsize=8, loc='upper left')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(RESULTS_ROOT) / f"forecast_comparison_v2.3_{timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n[OK] Saved: {save_path}")
        
        return save_path
    
    def _select_diverse_samples(self, test_df: pd.DataFrame, num_samples: int) -> List[int]:
        """多様なサンプル選択"""
        all_predictions = []
        all_labels = []
        
        for idx in range(len(test_df)):
            row = test_df.iloc[idx]
            sequence = self._parse_sequence(row['values_sequence'])
            
            predictions, pred_labels, _ = self.predict_sample(sequence)
            
            all_predictions.append(pred_labels[0])  # 30dで選択
            all_labels.append(int(row['label_30d']))
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # TN, TP, FP, FN
        tn = np.where((all_predictions == 0) & (all_labels == 0))[0]
        tp = np.where((all_predictions == 1) & (all_labels == 1))[0]
        fp = np.where((all_predictions == 1) & (all_labels == 0))[0]
        fn = np.where((all_predictions == 0) & (all_labels == 1))[0]
        
        samples = []
        if len(tn) > 0: samples.append(np.random.choice(tn))
        if len(tp) > 0: samples.append(np.random.choice(tp))
        if len(fp) > 0: samples.append(np.random.choice(fp))
        if len(fn) > 0: samples.append(np.random.choice(fn))
        
        while len(samples) < num_samples and len(samples) < len(test_df):
            rand_idx = np.random.randint(0, len(test_df))
            if rand_idx not in samples:
                samples.append(rand_idx)
        
        return samples[:num_samples]
    
    def _parse_sequence(self, seq_str: str) -> np.ndarray:
        """文字列から配列へ"""
        if isinstance(seq_str, str):
            values = [float(x) for x in seq_str.strip('[]').split(',')]
        else:
            values = seq_str
        
        values = np.array(values, dtype=np.float32)
        
        # Pad/Trim
        if len(values) >= LOOKBACK_DAYS:
            return values[-LOOKBACK_DAYS:]
        else:
            pad_len = LOOKBACK_DAYS - len(values)
            return np.pad(values, (pad_len, 0), mode='edge')


def main():
    """メイン実行"""
    print("="*80)
    print("v2.3 Multi-Task Model Forecast Visualization (Time-Series Only)")
    print("="*80)
    
    device = torch.device("cpu")
    print(f"\nDevice: {device}")
    
    model_path = Path(MODEL_ROOT) / "hybrid_model_v2.3" / "pytorch_model_multitask.pt"
    
    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print("Please train the model first using train_hybrid_model_v2_3.py")
        return
    
    visualizer = ForecastVisualizerV2_3(model_path, device)
    visualizer.load_model()
    
    print(f"\nLoading test data...")
    
    test_path = PROCESSED_DATA_DIR / "test_samples_enriched.csv"
    
    if not test_path.exists():
        print(f"\n[ERROR] Test data not found: {test_path}")
        return
    
    df_test = pd.read_csv(test_path)
    
    print(f"  [OK] Test samples: {len(df_test)}")
    
    print(f"\nGenerating multi-task comparison...")
    save_path = visualizer.plot_multitask_comparison(
        df_test,
        num_samples=5
    )
    
    print(f"\n[OK] Visualization completed!")
    print(f"   Saved to: {save_path}")


if __name__ == "__main__":
    main()
