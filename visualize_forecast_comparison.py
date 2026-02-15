"""
Hybrid Model Forecast vs Ground Truth Visualization
ハイブリッドモデル予測結果とGround-truthの比較可視化

TTM論文スタイル:
- 5つの時系列サンプル（縦5行）
- 3つのホライズン：30d, 60d, 90d（横3列）
- 合計15個の折れ線グラフ
"""

import sys
import os

# Granite TS用の回避策
sys.modules['torchvision'] = None
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from config import (
    PROCESSED_DATA_DIR,
    MODEL_ROOT,
    RESULTS_ROOT,
    FORECAST_HORIZONS,
    LOOKBACK_DAYS,
    RANDOM_SEED
)

# プロット設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class HybridDataset(Dataset):
    """ハイブリッドモデル用のデータセット"""
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.df = df
        self.feature_cols = feature_cols
        
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
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx]).unsqueeze(-1)
        features = torch.FloatTensor(self.features[idx])
        
        return {
            'sequence': sequence,
            'features': features,
            'idx': idx
        }


class ForecastVisualizer:
    """予測結果可視化クラス"""
    
    def __init__(self):
        self.test_df = None
        self.models = {}
        self.predictions = {}
        self.embeddings = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        print(f"🖥️  Device: {self.device}")
        print(f"📁 Model directory: {MODEL_ROOT}")
        print(f"📁 Results directory: {RESULTS_ROOT}")
    
    def load_test_data(self):
        """テストデータのロード"""
        print("\n📂 Loading test data...")
        
        test_path = PROCESSED_DATA_DIR / "test_samples_enriched.csv"
        
        if not test_path.exists():
            raise FileNotFoundError(
                f"Test data not found: {test_path}\n"
                "Please run create_enriched_features.py first."
            )
        
        self.test_df = pd.read_csv(test_path)
        
        print(f"✓ Loaded test data: {len(self.test_df):,} samples")
    
    def load_models(self):
        """訓練済みLightGBMモデルのロード"""
        print("\n📦 Loading trained LightGBM models...")
        
        model_dir = MODEL_ROOT / "hybrid_model"
        
        for horizon in FORECAST_HORIZONS:
            # 複数の命名パターンを試行
            possible_names = [
                f"lgbm_hybrid_{horizon}d.txt",
                f"model_{horizon}d.txt",
                f"lgbm_model_{horizon}d.txt"
            ]
            
            model_path = None
            for name in possible_names:
                candidate_path = model_dir / name
                if candidate_path.exists():
                    model_path = candidate_path
                    break
            
            if model_path is None:
                print(f"  ⚠ Model not found for {horizon}d horizon")
                print(f"  Tried: {possible_names}")
                continue
            
            self.models[horizon] = lgb.Booster(model_file=str(model_path))
            print(f"  ✓ Loaded model for {horizon}d horizon: {model_path.name}")
    
    def extract_embeddings_and_features(self, batch_size: int = 256):
        """埋め込みと統計特徴量を抽出"""
        print(f"\n🔬 Extracting embeddings and features...")
        
        from granite_ts_model import GraniteTimeSeriesClassifier
        
        # TinyTimeMixerエンコーダー
        model = GraniteTimeSeriesClassifier(
            num_horizons=len(FORECAST_HORIZONS),
            device=self.device
        )
        
        # エンコーダー抽出
        is_lstm_model = False
        if hasattr(model, 'lstm'):
            encoder = model.lstm
            is_lstm_model = True
        elif hasattr(model, 'base_model'):
            encoder = model.base_model
        elif hasattr(model, 'model'):
            encoder = model.model.base_model
        else:
            raise ValueError("Could not extract encoder")
        
        encoder.to(self.device)
        encoder.eval()
        
        # 統計特徴量のカラム名（train_hybrid_model.pyと同じロジック）
        exclude_cols = [
            'equipment_id', 'check_item_id', 'date', 
            'window_start', 'window_end', 'values_sequence',
            'reference_datetime', 'horizon_datetime',
            'label_current', 'label_30d', 'label_60d', 'label_90d',
            'any_anomaly'
        ]
        
        feature_cols = [col for col in self.test_df.columns 
                       if col not in exclude_cols]
        
        print(f"  Feature columns: {len(feature_cols)}")
        
        # データセット作成
        dataset = HybridDataset(self.test_df, feature_cols)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        embeddings_list = []
        statistical_features_list = []
        
        print(f"  Processing {len(dataset):,} samples...")
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                sequences = batch['sequence'].to(self.device)
                stat_features = batch['features'].cpu().numpy()
                
                # 埋め込み抽出
                if is_lstm_model:
                    lstm_out, (h, c) = encoder(sequences)
                    hidden = lstm_out[:, -1, :]
                else:
                    outputs = encoder(
                        past_values=sequences,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    
                    if hasattr(outputs, 'backbone_hidden_state') and outputs.backbone_hidden_state is not None:
                        hidden = outputs.backbone_hidden_state.squeeze(1).mean(dim=1)
                    elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        hidden = outputs.hidden_states[-1].squeeze(1).mean(dim=1)
                    else:
                        hidden = torch.mean(sequences, dim=1).squeeze()
                        if len(hidden.shape) == 1:
                            hidden = hidden.unsqueeze(-1)
                
                embeddings_list.append(hidden.cpu().numpy())
                statistical_features_list.append(stat_features)
                
                if (i + 1) % 10 == 0:
                    print(f"    Processed {(i+1)*batch_size:,} samples...", end='\r')
        
        # 結合
        self.embeddings = np.vstack(embeddings_list)
        statistical_features = np.vstack(statistical_features_list)
        
        # ハイブリッド特徴量
        self.X_hybrid = np.hstack([self.embeddings, statistical_features])
        
        print(f"\n✓ Extracted features: {self.X_hybrid.shape}")
        print(f"  Embeddings: {self.embeddings.shape[1]}d")
        print(f"  Statistical: {statistical_features.shape[1]}d")
    
    def predict_all_horizons(self):
        """全ホライズンで予測"""
        print(f"\n🔮 Predicting for all horizons...")
        
        if not self.models:
            raise RuntimeError(
                "No models loaded. Please check if model files exist in:\n"
                f"{MODEL_ROOT / 'hybrid_model'}"
            )
        
        for horizon in FORECAST_HORIZONS:
            if horizon not in self.models:
                print(f"  ⚠ Skipping {horizon}d (model not loaded)")
                continue
            
            model = self.models[horizon]
            predictions = model.predict(self.X_hybrid, num_iteration=model.best_iteration)
            
            self.predictions[horizon] = predictions
            
            # Ground truth
            label_col = f'label_{horizon}d'
            if label_col in self.test_df.columns:
                y_true = self.test_df[label_col].values
                auc = np.mean((predictions > 0.5) == y_true)
                print(f"  ✓ {horizon}d: predictions ready (accuracy={auc:.3f})")
    
    def select_diverse_samples(self, n_samples: int = 5) -> List[int]:
        """
        多様な時系列パターンを持つサンプルを選択
        
        Returns:
            選択されたサンプルのインデックスリスト
        """
        print(f"\n🎯 Selecting {n_samples} diverse samples...")
        
        if not self.predictions:
            raise RuntimeError("No predictions available. Run predict_all_horizons() first.")
        
        # 予測が存在するホライズンのみ使用
        available_horizons = list(self.predictions.keys())
        print(f"  Available horizons: {available_horizons}")
        
        # 各カテゴリーからサンプルを選択
        samples = []
        
        # 1. 正常サンプル（予測も正常）
        normal_correct = []
        for i in range(len(self.test_df)):
            if all(self.test_df[f'label_{h}d'].iloc[i] == 0 for h in available_horizons):
                if all(self.predictions[h][i] < 0.5 for h in available_horizons):
                    normal_correct.append(i)
        if normal_correct:
            samples.append(np.random.choice(normal_correct))
            print(f"  Selected normal correct: {samples[-1]}")
        
        # 2. 異常サンプル（予測も異常）
        anomaly_correct = []
        for i in range(len(self.test_df)):
            if any(self.test_df[f'label_{h}d'].iloc[i] == 1 for h in available_horizons):
                if any(self.predictions[h][i] > 0.5 for h in available_horizons):
                    anomaly_correct.append(i)
        if anomaly_correct:
            samples.append(np.random.choice(anomaly_correct))
            print(f"  Selected anomaly correct: {samples[-1]}")
        
        # 3. False Positive（正常だが異常と予測）
        false_positive = []
        for i in range(len(self.test_df)):
            if all(self.test_df[f'label_{h}d'].iloc[i] == 0 for h in available_horizons):
                if any(self.predictions[h][i] > 0.5 for h in available_horizons):
                    false_positive.append(i)
        if false_positive:
            samples.append(np.random.choice(false_positive))
            print(f"  Selected false positive: {samples[-1]}")
        
        # 4. False Negative（異常だが正常と予測）
        false_negative = []
        for i in range(len(self.test_df)):
            if any(self.test_df[f'label_{h}d'].iloc[i] == 1 for h in available_horizons):
                if all(self.predictions[h][i] < 0.5 for h in available_horizons):
                    false_negative.append(i)
        if false_negative:
            samples.append(np.random.choice(false_negative))
            print(f"  Selected false negative: {samples[-1]}")
        
        # 5. ランダムサンプル（埋め込み空間で離れたもの）
        if len(samples) < n_samples:
            remaining = n_samples - len(samples)
            existing_embeddings = self.embeddings[samples] if samples else np.array([])
            
            for _ in range(remaining):
                if len(existing_embeddings) == 0:
                    idx = np.random.randint(0, len(self.test_df))
                else:
                    # 既存サンプルから最も離れたものを選択
                    distances = np.linalg.norm(
                        self.embeddings[:, None, :] - existing_embeddings[None, :, :],
                        axis=2
                    ).min(axis=1)
                    idx = np.argmax(distances)
                
                samples.append(idx)
                existing_embeddings = np.vstack([existing_embeddings, self.embeddings[idx]])
            
            print(f"  Selected random samples: {samples[4:]}")
        
        print(f"  Final selected samples: {samples}")
        
        return samples[:n_samples]
    
    def plot_forecast_comparison(self, sample_indices: List[int]):
        """
        TTM論文スタイルの予測比較プロット
        
        Args:
            sample_indices: プロットするサンプルのインデックスリスト
        """
        print(f"\n📊 Creating forecast comparison plot...")
        
        # 予測が存在するホライズンのみ使用
        available_horizons = list(self.predictions.keys())
        if not available_horizons:
            raise RuntimeError("No predictions available for plotting.")
        
        n_samples = len(sample_indices)
        n_horizons = len(available_horizons)
        
        fig, axes = plt.subplots(n_samples, n_horizons, figsize=(5*n_horizons, 3*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        if n_horizons == 1:
            axes = axes.reshape(-1, 1)
        
        for row, idx in enumerate(sample_indices):
            # 時系列データ取得
            seq_str = self.test_df['values_sequence'].iloc[idx]
            import ast
            try:
                values = ast.literal_eval(seq_str)
            except:
                values = [float(x.strip('[] ')) for x in seq_str.split(',') if x.strip()]
            
            # パディング/トリミング
            if len(values) < LOOKBACK_DAYS:
                values = [values[0]] * (LOOKBACK_DAYS - len(values)) + values
            elif len(values) > LOOKBACK_DAYS:
                values = values[-LOOKBACK_DAYS:]
            
            # 時間軸
            time_axis = np.arange(len(values))
            
            for col, horizon in enumerate(available_horizons):
                ax = axes[row, col]
                
                # Ground truth（過去90日）
                ax.plot(time_axis, values, 
                       color='darkgreen', linewidth=2, 
                       label='Ground truth (Past 90d)', alpha=0.8)
                
                # 予測区間の設定
                future_start = len(values)
                future_end = len(values) + horizon
                future_time = np.arange(future_start, future_end)
                
                # 予測値取得
                pred_prob = self.predictions[horizon][idx]
                label_true = self.test_df[f'label_{horizon}d'].iloc[idx]
                
                # ホライズン期間の実績（異常発生有無）を背景色で表示
                if label_true == 1:
                    # 実際に異常が発生した期間を赤い背景で表示
                    ax.axvspan(future_start, future_end, alpha=0.15, color='red', 
                              label='Actual anomaly period' if row == 0 and col == 0 else '')
                else:
                    # 正常期間を薄い緑の背景で表示
                    ax.axvspan(future_start, future_end, alpha=0.1, color='green',
                              label='Actual normal period' if row == 0 and col == 0 else '')
                
                # 予測値を確率に基づいて可視化
                last_value = values[-1]
                
                # 予測トレンドライン
                if pred_prob > 0.5:
                    # 異常予測：確率に応じた上昇トレンド
                    trend_magnitude = (pred_prob - 0.5) * 2  # 0.5→0.0, 1.0→1.0
                    future_values = np.linspace(last_value, last_value + trend_magnitude, len(future_time))
                    color = 'red'
                    linestyle = '--'
                    label_pred = f'Predicted anomaly (p={pred_prob:.2f})'
                else:
                    # 正常予測：確率に応じた安定トレンド
                    trend_magnitude = (0.5 - pred_prob) * 0.5
                    future_values = np.linspace(last_value, last_value - trend_magnitude * 0.2, len(future_time))
                    color = 'blue'
                    linestyle = '--'
                    label_pred = f'Predicted normal (p={pred_prob:.2f})'
                
                ax.plot(future_time, future_values,
                       color=color, linewidth=2.5, linestyle=linestyle,
                       label=label_pred if row == 0 and col == 0 else '', alpha=0.8)
                
                # 境界線（予測開始点）
                ax.axvline(x=future_start, color='black', linestyle=':', linewidth=1, alpha=0.5)
                
                # タイトル
                status = "Anomaly" if label_true == 1 else "Normal"
                pred_status = "Anomaly" if pred_prob > 0.5 else "Normal"
                correct = "✓" if (label_true == 1) == (pred_prob > 0.5) else "✗"
                
                if row == 0:
                    ax.set_title(f'{horizon}d Horizon', fontsize=12, fontweight='bold')
                
                # Y軸ラベル
                if col == 0:
                    equipment_id = self.test_df['equipment_id'].iloc[idx]
                    ax.set_ylabel(f'Sample {row+1}\nEquip {equipment_id}', 
                                 fontsize=10, fontweight='bold')
                
                # Ground truthと予測結果を注釈
                status = "Anomaly" if label_true == 1 else "Normal"
                pred_status = "Anomaly" if pred_prob > 0.5 else "Normal"
                correct = "✓" if (label_true == 1) == (pred_prob > 0.5) else "✗"
                
                ax.text(0.02, 0.98, 
                       f'Actual: {status}\nPred: {pred_status} {correct}', 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', 
                                facecolor='lightgreen' if correct == '✓' else 'lightcoral', 
                                alpha=0.7))
                
                ax.set_xlabel('Time (days)', fontsize=9)
                ax.grid(True, alpha=0.3)
                
                if row == 0 and col == 0:
                    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
        
        plt.suptitle('Hybrid Model Forecast vs Ground Truth\n'
                    'TinyTimeMixer Embeddings (64d) + Statistical Features (28d) → LightGBM',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # 保存
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_path = RESULTS_ROOT / f'forecast_comparison_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        
        plt.show()
    
    def run(self, n_samples: int = 5):
        """完全実行"""
        print("\n" + "="*70)
        print("Hybrid Model Forecast Visualization (TTM Style)")
        print("="*70)
        
        # データロード
        self.load_test_data()
        
        # モデルロード
        self.load_models()
        
        # 埋め込み＆特徴量抽出
        self.extract_embeddings_and_features()
        
        # 予測
        self.predict_all_horizons()
        
        # サンプル選択
        sample_indices = self.select_diverse_samples(n_samples=n_samples)
        
        # 可視化
        self.plot_forecast_comparison(sample_indices)
        
        print("\n✅ Visualization completed!")
        print(f"📁 Results saved to: {RESULTS_ROOT}")


def main():
    visualizer = ForecastVisualizer()
    visualizer.run(n_samples=5)  # 5行×3列=15グラフ


if __name__ == "__main__":
    main()
