"""
Visualize TinyTimeMixer Embeddings for Hybrid Model v2.0
ハイブリッドモデルv2.0のTinyTimeMixer埋め込み(64次元)可視化

目的:
- テストサンプル8,745に対してTinyTimeMixerの64次元埋め込みを可視化
- 正常/異常のクラスター分離を多様な手法で確認

可視化内容:
1. 2D/3D t-SNE: 非線形な構造を捉える
2. 2D/3D UMAP: 大域的・局所的構造を保持
3. 各予測ホライズン(30日、60日、90日)ごとの分布
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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# UMAPインポート
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠ UMAP not available. Install with: pip install umap-learn")

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
            'features': features
        }


class EmbeddingVisualizer:
    """埋め込み可視化クラス"""
    
    def __init__(self):
        self.test_df = None
        self.embeddings = None
        self.labels = {}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        print(f"🖥️  Device: {self.device}")
        print(f"📁 Data directory: {PROCESSED_DATA_DIR}")
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
        
        # ラベル確認
        for horizon in FORECAST_HORIZONS:
            label_col = f'label_{horizon}d'
            if label_col in self.test_df.columns:
                num_positives = self.test_df[label_col].sum()
                num_total = len(self.test_df)
                positive_rate = num_positives / num_total * 100
                self.labels[horizon] = self.test_df[label_col].values
                print(f"  {horizon}d: {num_positives:,} anomalies ({positive_rate:.1f}%)")
    
    def extract_embeddings(self, batch_size: int = 256):
        """Granite TS TinyTimeMixerエンコーダーから埋め込みを抽出"""
        print(f"\n🔬 Extracting embeddings from TinyTimeMixer encoder...")
        
        from granite_ts_model import GraniteTimeSeriesClassifier
        
        model = GraniteTimeSeriesClassifier(
            num_horizons=len(FORECAST_HORIZONS),
            device=self.device
        )
        
        # エンコーダー部分を抽出
        is_lstm_model = False
        if hasattr(model, 'lstm'):
            encoder = model.lstm
            is_lstm_model = True
            embedding_dim = model.hidden_size
            print(f"  Using LSTM encoder (embedding_dim={embedding_dim})")
        elif hasattr(model, 'base_model'):
            encoder = model.base_model
            embedding_dim = 64
            print(f"  Using TinyTimeMixer encoder (embedding_dim={embedding_dim})")
        elif hasattr(model, 'model'):
            encoder = model.model.base_model
            embedding_dim = 64
            print(f"  Using TinyTimeMixer encoder with PEFT (embedding_dim={embedding_dim})")
        else:
            raise ValueError("Could not extract encoder from Granite TS model")
        
        encoder.to(self.device)
        encoder.eval()
        
        # 統計的特徴量のカラム名を取得(数値カラムのみ)
        exclude_cols = ['equipment_id', 'check_item_id', 'values_sequence', 
                       'reference_datetime', 'horizon_datetime']
        feature_cols = []
        for col in self.test_df.columns:
            if col in exclude_cols or col.startswith('label_'):
                continue
            if pd.api.types.is_numeric_dtype(self.test_df[col]):
                feature_cols.append(col)
        
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
        
        print(f"  Processing {len(dataset):,} samples in batches of {batch_size}...")
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                sequences = batch['sequence'].to(self.device)
                
                try:
                    if is_lstm_model:
                        # LSTMの場合
                        lstm_out, (h, c) = encoder(sequences)
                        hidden = lstm_out[:, -1, :]
                    else:
                        # TinyTimeMixerの場合
                        outputs = encoder(
                            past_values=sequences,
                            output_hidden_states=True,
                            return_dict=True
                        )
                        
                        if hasattr(outputs, 'backbone_hidden_state') and outputs.backbone_hidden_state is not None:
                            backbone_hidden = outputs.backbone_hidden_state
                            hidden = backbone_hidden.squeeze(1).mean(dim=1)
                        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                            last_hidden = outputs.hidden_states[-1]
                            hidden = last_hidden.squeeze(1).mean(dim=1)
                        else:
                            hidden = torch.mean(sequences, dim=1).squeeze()
                            if len(hidden.shape) == 1:
                                hidden = hidden.unsqueeze(-1)
                    
                    embeddings_list.append(hidden.cpu().numpy())
                    
                    if (i + 1) % 10 == 0:
                        print(f"    Processed {(i+1)*batch_size:,} samples...", end='\r')
                
                except Exception as e:
                    print(f"\n  ⚠ Error extracting embeddings: {e}")
                    raise
        
        self.embeddings = np.vstack(embeddings_list)
        
        print(f"\n✓ Extracted embeddings: {self.embeddings.shape}")
        print(f"  Embedding dimension: {self.embeddings.shape[1]}")
    
    def visualize_tsne_3d(self, perplexity=30, max_iter=1000):
        """3次元t-SNE可視化"""
        print(f"\n🔄 Running 3D t-SNE...")
        print(f"  Perplexity: {perplexity}, Iterations: {max_iter}")
        
        tsne = TSNE(n_components=3, perplexity=perplexity, max_iter=max_iter,
                    random_state=RANDOM_SEED, verbose=1)
        embeddings_3d = tsne.fit_transform(self.embeddings)
        
        print(f"✓ 3D t-SNE completed: {embeddings_3d.shape}")
        
        self._plot_3d_by_horizon(embeddings_3d, 't-SNE')
        self._plot_3d_combined(embeddings_3d, 't-SNE')
    
    def visualize_umap_2d(self, n_neighbors=15, min_dist=0.1):
        """2次元UMAP可視化"""
        if not UMAP_AVAILABLE:
            print("⚠ UMAP not available. Skipping.")
            return
        
        print(f"\n🔄 Running 2D UMAP...")
        print(f"  n_neighbors: {n_neighbors}, min_dist: {min_dist}")
        
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                           random_state=RANDOM_SEED, verbose=True)
        embeddings_2d = reducer.fit_transform(self.embeddings)
        
        print(f"✓ 2D UMAP completed: {embeddings_2d.shape}")
        
        self._plot_2d_by_horizon(embeddings_2d, 'UMAP')
        self._plot_2d_combined(embeddings_2d, 'UMAP')
    
    def visualize_umap_3d(self, n_neighbors=15, min_dist=0.1):
        """3次元UMAP可視化"""
        if not UMAP_AVAILABLE:
            print("⚠ UMAP not available. Skipping.")
            return
        
        print(f"\n🔄 Running 3D UMAP...")
        print(f"  n_neighbors: {n_neighbors}, min_dist: {min_dist}")
        
        reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                           random_state=RANDOM_SEED, verbose=True)
        embeddings_3d = reducer.fit_transform(self.embeddings)
        
        print(f"✓ 3D UMAP completed: {embeddings_3d.shape}")
        
        self._plot_3d_by_horizon(embeddings_3d, 'UMAP')
        self._plot_3d_combined(embeddings_3d, 'UMAP')
    
    def _plot_2d_by_horizon(self, embeddings_2d, method):
        """2D: 各ホライズンごとにプロット"""
        print(f"\n📊 Plotting 2D {method} by horizon...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            ax = axes[idx]
            labels = self.labels[horizon]
            
            normal_mask = labels == 0
            ax.scatter(embeddings_2d[normal_mask, 0], embeddings_2d[normal_mask, 1],
                      c='blue', alpha=0.5, s=10, label=f'Normal (n={normal_mask.sum():,})',
                      edgecolors='none')
            
            anomaly_mask = labels == 1
            ax.scatter(embeddings_2d[anomaly_mask, 0], embeddings_2d[anomaly_mask, 1],
                      c='red', alpha=0.7, s=15, label=f'Anomaly (n={anomaly_mask.sum():,})',
                      edgecolors='black', linewidths=0.5)
            
            ax.set_title(f'{horizon}d Horizon', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{method} Dim 1', fontsize=11)
            ax.set_ylabel(f'{method} Dim 2', fontsize=11)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'TinyTimeMixer Embeddings ({method} 2D) - n={len(embeddings_2d):,}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_path = RESULTS_ROOT / f'embeddings_{method.lower()}_2d_by_horizon_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.show()
    
    def _plot_2d_combined(self, embeddings_2d, method):
        """2D: 統合プロット"""
        print(f"\n📊 Plotting 2D {method} combined...")
        
        horizon = 60
        labels = self.labels[horizon]
        
        plt.figure(figsize=(10, 8))
        
        normal_mask = labels == 0
        plt.scatter(embeddings_2d[normal_mask, 0], embeddings_2d[normal_mask, 1],
                   c='blue', alpha=0.5, s=10, label=f'Normal (n={normal_mask.sum():,})',
                   edgecolors='none')
        
        anomaly_mask = labels == 1
        plt.scatter(embeddings_2d[anomaly_mask, 0], embeddings_2d[anomaly_mask, 1],
                   c='red', alpha=0.7, s=15, label=f'Anomaly (n={anomaly_mask.sum():,})',
                   edgecolors='black', linewidths=0.5)
        
        plt.title(f'TinyTimeMixer Embeddings ({method} 2D)\n{horizon}d Horizon',
                 fontsize=14, fontweight='bold')
        plt.xlabel(f'{method} Dim 1', fontsize=12)
        plt.ylabel(f'{method} Dim 2', fontsize=12)
        plt.legend(loc='best', framealpha=0.9, fontsize=11)
        plt.grid(True, alpha=0.3)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_path = RESULTS_ROOT / f'embeddings_{method.lower()}_2d_combined_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.show()
    
    def _plot_3d_by_horizon(self, embeddings_3d, method):
        """3D: 各ホライズンごとにプロット"""
        print(f"\n📊 Plotting 3D {method} by horizon...")
        
        fig = plt.figure(figsize=(20, 6))
        
        for idx, horizon in enumerate(FORECAST_HORIZONS):
            ax = fig.add_subplot(1, 3, idx+1, projection='3d')
            labels = self.labels[horizon]
            
            normal_mask = labels == 0
            ax.scatter(embeddings_3d[normal_mask, 0], embeddings_3d[normal_mask, 1],
                      embeddings_3d[normal_mask, 2], c='blue', alpha=0.4, s=8,
                      label=f'Normal (n={normal_mask.sum():,})', edgecolors='none')
            
            anomaly_mask = labels == 1
            ax.scatter(embeddings_3d[anomaly_mask, 0], embeddings_3d[anomaly_mask, 1],
                      embeddings_3d[anomaly_mask, 2], c='red', alpha=0.7, s=12,
                      label=f'Anomaly (n={anomaly_mask.sum():,})', edgecolors='black', linewidths=0.5)
            
            ax.set_title(f'{horizon}d Horizon', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{method} Dim 1', fontsize=10)
            ax.set_ylabel(f'{method} Dim 2', fontsize=10)
            ax.set_zlabel(f'{method} Dim 3', fontsize=10)
            ax.legend(loc='best', framealpha=0.9, fontsize=9)
            ax.view_init(elev=20, azim=45)
        
        plt.suptitle(f'TinyTimeMixer Embeddings ({method} 3D) - n={len(embeddings_3d):,}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_path = RESULTS_ROOT / f'embeddings_{method.lower()}_3d_by_horizon_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.show()
    
    def _plot_3d_combined(self, embeddings_3d, method):
        """3D: 統合プロット(複数視点)"""
        print(f"\n📊 Plotting 3D {method} combined (multiple views)...")
        
        horizon = 60
        labels = self.labels[horizon]
        
        fig = plt.figure(figsize=(18, 6))
        view_angles = [(20, 45), (20, 135), (60, 45)]
        view_names = ['Front-Right', 'Front-Left', 'Top-Right']
        
        for idx, (elev, azim) in enumerate(view_angles):
            ax = fig.add_subplot(1, 3, idx+1, projection='3d')
            
            normal_mask = labels == 0
            ax.scatter(embeddings_3d[normal_mask, 0], embeddings_3d[normal_mask, 1],
                      embeddings_3d[normal_mask, 2], c='blue', alpha=0.4, s=8,
                      label=f'Normal (n={normal_mask.sum():,})', edgecolors='none')
            
            anomaly_mask = labels == 1
            ax.scatter(embeddings_3d[anomaly_mask, 0], embeddings_3d[anomaly_mask, 1],
                      embeddings_3d[anomaly_mask, 2], c='red', alpha=0.7, s=12,
                      label=f'Anomaly (n={anomaly_mask.sum():,})', edgecolors='black', linewidths=0.5)
            
            ax.set_title(f'View: {view_names[idx]}', fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{method} Dim 1', fontsize=10)
            ax.set_ylabel(f'{method} Dim 2', fontsize=10)
            ax.set_zlabel(f'{method} Dim 3', fontsize=10)
            if idx == 0:
                ax.legend(loc='best', framealpha=0.9, fontsize=9)
            ax.view_init(elev=elev, azim=azim)
        
        plt.suptitle(f'TinyTimeMixer Embeddings ({method} 3D)\n{horizon}d Horizon',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_path = RESULTS_ROOT / f'embeddings_{method.lower()}_3d_combined_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.show()
    
    def run(self):
        """完全実行"""
        print("\n" + "="*70)
        print("Granite TS TinyTimeMixer Embedding Visualization")
        print("="*70)
        
        self.load_test_data()
        self.extract_embeddings()
        
        # 3D t-SNE
        self.visualize_tsne_3d()
        
        # 2D UMAP
        self.visualize_umap_2d()
        
        # 3D UMAP
        self.visualize_umap_3d()
        
        print("\n✅ Visualization completed!")
        print(f"📁 Results saved to: {RESULTS_ROOT}")


def main():
    visualizer = EmbeddingVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()
