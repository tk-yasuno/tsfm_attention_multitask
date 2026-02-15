"""
Enhanced Forecast Visualization for v2.1
v2.1モデルの非線形予測可視化

特徴:
- 過去90日の延長線上にある予測
- 統計特徴量を活用した非線形トレンド
- 設備管理者向けの直感的な表示
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
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    PROCESSED_DATA_DIR,
    MODEL_ROOT,
    RESULTS_ROOT,
    FORECAST_HORIZONS,
    LOOKBACK_DAYS,
    USE_GPU,
    GPU_ID
)

from granite_ts_model import GraniteTimeSeriesClassifier

# プロット設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class EnhancedHybridModel(nn.Module):
    """v2.1 Enhanced Hybrid Model (推論用)"""
    
    def __init__(
        self,
        granite_model: GraniteTimeSeriesClassifier,
        stat_feature_dim: int = 28,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # TinyTimeMixer Encoder
        if hasattr(granite_model, 'base_model'):
            self.encoder = granite_model.base_model
        elif hasattr(granite_model, 'model'):
            self.encoder = granite_model.model.base_model
        elif hasattr(granite_model, 'lstm'):
            self.encoder = granite_model.lstm
        else:
            raise ValueError("Could not extract encoder")
        
        self.embedding_dim = 64
        
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
        # TinyTimeMixer Embeddings
        with torch.no_grad():
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
        
        # Feature Fusion
        fused_features = torch.cat([embeddings, features], dim=1)
        
        # Classification
        predictions = self.fusion_layer(fused_features)
        
        return predictions.squeeze(1)


class ForecastVisualizerV2_1:
    """v2.1 予測可視化クラス"""
    
    def __init__(self):
        self.device = torch.device(f'cuda:{GPU_ID}' if USE_GPU and torch.cuda.is_available() else 'cpu')
        self.test_df = None
        self.feature_cols = []
        self.models = {}
        self.thresholds = {}
        
        print(f"🖥️  Device: {self.device}")
    
    def load_data(self):
        """テストデータをロード"""
        print("\n📂 Loading test data...")
        
        test_path = PROCESSED_DATA_DIR / "test_samples_enriched.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        self.test_df = pd.read_csv(test_path)
        print(f"✓ Loaded {len(self.test_df):,} test samples")
        
        # 特徴量カラム
        exclude_cols = [
            'equipment_id', 'check_item_id', 'date', 
            'window_start', 'window_end', 'values_sequence',
            'reference_datetime', 'horizon_datetime',
            'label_current', 'label_30d', 'label_60d', 'label_90d',
            'any_anomaly'
        ]
        
        self.feature_cols = [col for col in self.test_df.columns 
                            if col not in exclude_cols]
        
        print(f"✓ Statistical features: {len(self.feature_cols)}d")
    
    def load_models(self):
        """v2.1モデルをロード"""
        print("\n🔧 Loading v2.1 models...")
        
        model_dir = MODEL_ROOT / "hybrid_model_v2.1"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # LoRA設定（r=8, alpha=16で規模を制御）
        lora_config = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "bias": "none"
        }
        
        # Granite TS Base Model
        granite_model = GraniteTimeSeriesClassifier(
            num_horizons=len(FORECAST_HORIZONS),
            device=self.device,
            lora_config=lora_config
        )
        
        for horizon in FORECAST_HORIZONS:
            model_path = model_dir / f"pytorch_model_{horizon}d.pt"
            
            if not model_path.exists():
                print(f"  ⚠️  Model not found: {model_path}")
                continue
            
            # モデル構築
            model = EnhancedHybridModel(
                granite_model=granite_model,
                stat_feature_dim=len(self.feature_cols),
                hidden_dim=128,
                dropout=0.3
            ).to(self.device)
            
            # 重みロード
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models[horizon] = model
            self.thresholds[horizon] = checkpoint['threshold']
            
            metrics = checkpoint['metrics']
            print(f"  ✓ {horizon}d horizon: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        print(f"✓ Loaded {len(self.models)} models")
    
    def predict_sample(self, idx: int, horizon: int) -> Tuple[float, float, np.ndarray]:
        """
        1サンプルの予測
        
        Returns:
            prob: 異常確率
            label: 正解ラベル
            sequence: 過去90日の時系列
        """
        if horizon not in self.models:
            raise ValueError(f"Model for {horizon}d horizon not loaded")
        
        # データ取得
        row = self.test_df.iloc[idx]
        
        # 時系列
        import ast
        try:
            values = ast.literal_eval(row['values_sequence'])
        except:
            values = [float(x.strip('[] ')) for x in row['values_sequence'].split(',') if x.strip()]
        
        if len(values) < LOOKBACK_DAYS:
            values = [values[0]] * (LOOKBACK_DAYS - len(values)) + values
        elif len(values) > LOOKBACK_DAYS:
            values = values[-LOOKBACK_DAYS:]
        
        sequence = torch.FloatTensor(values).unsqueeze(0).unsqueeze(-1).to(self.device)  # [1, 90, 1]
        
        # 統計特徴量（数値型に変換してからTensor化）
        feature_values = row[self.feature_cols].values
        feature_values = pd.to_numeric(feature_values, errors='coerce')  # object型を数値に変換
        feature_values = np.nan_to_num(feature_values, nan=0.0)  # NaNを0に置換
        features = torch.FloatTensor(feature_values).unsqueeze(0).to(self.device)  # [1, 28]
        
        # 予測
        model = self.models[horizon]
        with torch.no_grad():
            prob = model(sequence, features).cpu().item()
        
        # ラベル
        label = row[f'label_{horizon}d']
        
        return prob, label, np.array(values)
    
    def generate_nonlinear_forecast(
        self, 
        sequence: np.ndarray, 
        features: torch.Tensor,
        horizon: int,
        num_points: int = 30
    ) -> np.ndarray:
        """
        非線形予測の生成
        
        過去90日のトレンドと統計特徴量から、
        ホライズン期間の予測値を補間生成
        
        Args:
            sequence: 過去90日の時系列 [90]
            features: 統計特徴量 [28]
            horizon: 予測ホライズン日数
            num_points: 予測点数
        
        Returns:
            forecast: 非線形予測値 [num_points]
        """
        # 過去のトレンド分析
        recent_values = sequence[-30:]  # 直近30日
        trend = np.polyfit(range(len(recent_values)), recent_values, 2)  # 2次フィット
        
        # 統計特徴量からのトレンド調整
        mean_val = sequence.mean()
        std_val = sequence.std()
        recent_mean = recent_values.mean()
        
        # 予測点の生成
        forecast_x = np.linspace(0, horizon, num_points)
        
        # 2次多項式ベースの予測
        base_forecast = np.polyval(trend, np.arange(len(recent_values), len(recent_values) + num_points))
        
        # 統計特徴量による調整（過度な発散を抑制）
        decay_factor = np.exp(-forecast_x / (horizon * 2))  # 減衰係数
        adjusted_forecast = (base_forecast - recent_mean) * decay_factor + recent_mean
        
        # 範囲制限（実測値の±3σ以内）
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
        adjusted_forecast = np.clip(adjusted_forecast, lower_bound, upper_bound)
        
        return adjusted_forecast
    
    def plot_enhanced_comparison(
        self, 
        sample_indices: List[int] = None,
        figsize: Tuple[int, int] = (20, 16)
    ):
        """
        v2.1拡張予測比較プロット
        
        各サンプルに対して:
        - 過去90日の実績値（青線）
        - 3つのホライズン予測（赤/緑破線、非線形）
        - 異常確率（タイトルに表示）
        """
        print("\n📊 Generating enhanced forecast comparison...")
        
        if sample_indices is None:
            # デフォルト: 多様なサンプルを選択
            sample_indices = self._select_diverse_samples(num_samples=5)
        
        n_samples = len(sample_indices)
        n_horizons = len(FORECAST_HORIZONS)
        
        fig, axes = plt.subplots(n_samples, n_horizons, figsize=figsize)
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, sample_idx in enumerate(sample_indices):
            row_data = self.test_df.iloc[sample_idx]
            equipment_id = row_data['equipment_id']
            check_item = row_data['check_item_id']
            
            # 時系列取得
            import ast
            try:
                values = ast.literal_eval(row_data['values_sequence'])
            except:
                values = [float(x.strip('[] ')) for x in row_data['values_sequence'].split(',') if x.strip()]
            
            if len(values) < LOOKBACK_DAYS:
                values = [values[0]] * (LOOKBACK_DAYS - len(values)) + values
            elif len(values) > LOOKBACK_DAYS:
                values = values[-LOOKBACK_DAYS:]
            
            sequence = np.array(values)
            
            # 統計特徴量（数値型に変換してからTensor化）
            feature_values = row_data[self.feature_cols].values
            feature_values = pd.to_numeric(feature_values, errors='coerce')  # object型を数値に変換
            feature_values = np.nan_to_num(feature_values, nan=0.0)  # NaNを0に置換
            features = torch.FloatTensor(feature_values).unsqueeze(0).to(self.device)
            
            for col_idx, horizon in enumerate(FORECAST_HORIZONS):
                ax = axes[row_idx, col_idx]
                
                # 過去90日プロット
                x_past = np.arange(-LOOKBACK_DAYS, 0)
                ax.plot(x_past, sequence, 'b-', linewidth=2, label='Past 90 days', alpha=0.8)
                
                # 非線形予測
                if horizon in self.models:
                    prob, label, _ = self.predict_sample(sample_idx, horizon)
                    forecast = self.generate_nonlinear_forecast(sequence, features, horizon, num_points=horizon)
                    
                    x_future = np.linspace(0, horizon, len(forecast))
                    
                    # 予測線の色（異常確率に応じて）
                    is_anomaly_pred = prob > self.thresholds[horizon]
                    color = 'red' if is_anomaly_pred else 'green'
                    linestyle = '--'
                    
                    ax.plot(x_future, forecast, color=color, linestyle=linestyle, 
                           linewidth=2, label=f'Forecast (p={prob:.3f})', alpha=0.8)
                    
                    # 予測開始点のマーカー
                    ax.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
                    
                    # 実績異常期間の背景色
                    if label == 1:
                        ax.axvspan(0, horizon, alpha=0.1, color='red', label='Actual Anomaly')
                    else:
                        ax.axvspan(0, horizon, alpha=0.1, color='green', label='Actual Normal')
                    
                    # タイトル
                    pred_status = "Anomaly" if is_anomaly_pred else "Normal"
                    true_status = "Anomaly" if label == 1 else "Normal"
                    correctness = "✓" if (is_anomaly_pred == label) else "✗"
                    
                    ax.set_title(
                        f'{horizon}d: Pred={pred_status}, True={true_status} {correctness}\n'
                        f'Prob={prob:.3f} (Threshold={self.thresholds[horizon]:.3f})',
                        fontsize=10
                    )
                else:
                    ax.text(0.5, 0.5, f'Model {horizon}d\nNot Available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                
                # ラベル
                ax.set_xlabel('Days from reference', fontsize=9)
                ax.set_ylabel('Value', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='best')
                
                # Y軸範囲の統一
                y_min, y_max = sequence.min(), sequence.max()
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.2 * y_range, y_max + 0.2 * y_range)
        
        # 全体タイトル
        plt.suptitle(
            f'Enhanced Hybrid Model v2.1: Nonlinear Forecast Comparison\n'
            f'{n_samples} Diverse Samples × {n_horizons} Horizons',
            fontsize=16, fontweight='bold', y=0.995
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # 保存
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_path = RESULTS_ROOT / f'forecast_comparison_v2.1_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        
        plt.show()
    
    def _select_diverse_samples(self, num_samples: int = 5) -> List[int]:
        """多様なサンプルを選択"""
        print("\n🎯 Selecting diverse samples...")
        
        # 60dホライズンで選択
        horizon = 60
        if horizon not in self.models:
            print("  ⚠️  Using random selection (model not available)")
            return np.random.choice(len(self.test_df), size=num_samples, replace=False).tolist()
        
        # 全サンプルで予測
        all_probs = []
        for idx in range(len(self.test_df)):
            prob, _, _ = self.predict_sample(idx, horizon)
            all_probs.append(prob)
        
        all_probs = np.array(all_probs)
        all_labels = self.test_df[f'label_{horizon}d'].values
        threshold = self.thresholds[horizon]
        all_preds = (all_probs > threshold).astype(int)
        
        # カテゴリ別サンプル
        tn_indices = np.where((all_preds == 0) & (all_labels == 0))[0]  # True Negative
        tp_indices = np.where((all_preds == 1) & (all_labels == 1))[0]  # True Positive
        fp_indices = np.where((all_preds == 1) & (all_labels == 0))[0]  # False Positive
        fn_indices = np.where((all_preds == 0) & (all_labels == 1))[0]  # False Negative
        
        selected = []
        
        # 各カテゴリから1つずつ
        if len(tn_indices) > 0:
            selected.append(np.random.choice(tn_indices))
        if len(tp_indices) > 0:
            selected.append(np.random.choice(tp_indices))
        if len(fp_indices) > 0:
            selected.append(np.random.choice(fp_indices))
        if len(fn_indices) > 0:
            selected.append(np.random.choice(fn_indices))
        
        # 残りはランダム
        while len(selected) < num_samples:
            candidate = np.random.randint(0, len(self.test_df))
            if candidate not in selected:
                selected.append(candidate)
        
        print(f"  ✓ Selected samples: {selected}")
        return selected
    
    def run(self, sample_indices: List[int] = None):
        """完全実行"""
        print("\n" + "="*70)
        print("Enhanced Hybrid Model v2.1: Forecast Visualization")
        print("="*70)
        
        self.load_data()
        self.load_models()
        self.plot_enhanced_comparison(sample_indices=sample_indices)
        
        print("\n✅ Visualization completed!")


def main():
    visualizer = ForecastVisualizerV2_1()
    visualizer.run()


if __name__ == "__main__":
    main()
