"""
Feature Engineering for Pump Range Deviation Forecast
特徴量エンジニアリング

現在の生の時系列データに加えて、統計的特徴量を追加することで
モデルの予測精度（特にPrecision）を向上させる。

追加する特徴量：
1. 統計的特徴量（平均、標準偏差、歪度、尖度など）
2. トレンド特徴量（線形回帰の傾き、変化率など）
3. レンジ関連特徴量（上限/下限までの距離、逸脱回数など）
4. 変動性特徴量（ローリング統計量など）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

from config import (
    PROCESSED_DATA_DIR,
    LOOKBACK_DAYS
)


class FeatureEngineer:
    """特徴量エンジニアリングクラス"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_statistical_features(self, sequence: np.ndarray) -> Dict[str, float]:
        """
        統計的特徴量の抽出（NaN/Inf安全版）
        
        Args:
            sequence: 時系列データ [seq_len]
            
        Returns:
            特徴量の辞書
        """
        features = {}
        
        # 入力データのクリーン
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 基本統計量
        features['mean'] = float(np.mean(sequence))
        features['std'] = float(np.std(sequence))
        features['min'] = float(np.min(sequence))
        features['max'] = float(np.max(sequence))
        features['median'] = float(np.median(sequence))
        features['range'] = features['max'] - features['min']
        
        # 四分位数
        features['q25'] = float(np.percentile(sequence, 25))
        features['q75'] = float(np.percentile(sequence, 75))
        features['iqr'] = features['q75'] - features['q25']
        
        # 歪度・尖度（NaN対策）
        if len(sequence) > 3:
            try:
                skew_val = stats.skew(sequence)
                kurt_val = stats.kurtosis(sequence)
                features['skewness'] = float(skew_val) if np.isfinite(skew_val) else 0.0
                features['kurtosis'] = float(kurt_val) if np.isfinite(kurt_val) else 0.0
            except:
                features['skewness'] = 0.0
                features['kurtosis'] = 0.0
        else:
            features['skewness'] = 0.0
            features['kurtosis'] = 0.0
        
        # 変動係数（安全な除算）
        mean_abs = abs(features['mean'])
        if mean_abs > 1e-10:  # より安全な閾値
            cv_val = features['std'] / mean_abs
            features['cv'] = float(cv_val) if np.isfinite(cv_val) else 0.0
        else:
            features['cv'] = 0.0
        
        return features
    
    def extract_trend_features(self, sequence: np.ndarray) -> Dict[str, float]:
        """
        トレンド特徴量の抽出（NaN/Inf安全版）
        
        Args:
            sequence: 時系列データ [seq_len]
            
        Returns:
            特徴量の辞書
        """
        features = {}
        
        # 入力データのクリーン
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 線形回帰による傾き
        X = np.arange(len(sequence)).reshape(-1, 1)
        y = sequence.reshape(-1, 1)
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            slope = float(model.coef_[0][0])
            intercept = float(model.intercept_[0])
            features['trend_slope'] = slope if np.isfinite(slope) else 0.0
            features['trend_intercept'] = intercept if np.isfinite(intercept) else float(sequence[0])
        except:
            features['trend_slope'] = 0.0
            features['trend_intercept'] = float(sequence[0])
        
        # 最近の期間 vs 過去の期間（安全な除算）
        if len(sequence) >= 60:
            recent_mean = float(np.mean(sequence[-30:]))  # 最近30日
            past_mean = float(np.mean(sequence[-60:-30]))  # 過去30日
            
            if abs(past_mean) > 1e-10:
                ratio = recent_mean / past_mean
                features['recent_vs_past_ratio'] = float(ratio) if np.isfinite(ratio) else 1.0
                features['recent_vs_past_diff'] = float(recent_mean - past_mean)
            else:
                features['recent_vs_past_ratio'] = 1.0
                features['recent_vs_past_diff'] = 0.0
        else:
            features['recent_vs_past_ratio'] = 1.0
            features['recent_vs_past_diff'] = 0.0
        
        # 最終値の変化率
        if len(sequence) >= 10:
            recent_slope = float((sequence[-1] - sequence[-10]) / 10)
            features['recent_change_rate'] = recent_slope if np.isfinite(recent_slope) else 0.0
        else:
            features['recent_change_rate'] = 0.0
        
        return features
    
    def extract_range_features(
        self, 
        sequence: np.ndarray,
        upper_limit: float,
        lower_limit: float
    ) -> Dict[str, float]:
        """
        レンジ関連特徴量の抽出（NaN/Inf安全版）
        
        Args:
            sequence: 時系列データ [seq_len]
            upper_limit: 上限値
            lower_limit: 下限値
            
        Returns:
            特徴量の辞書
        """
        features = {}
        
        # 入力データのクリーン
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 上限/下限までの距離
        distance_to_upper = upper_limit - sequence
        distance_to_lower = sequence - lower_limit
        
        features['mean_distance_to_upper'] = float(np.mean(distance_to_upper))
        features['mean_distance_to_lower'] = float(np.mean(distance_to_lower))
        features['min_distance_to_upper'] = float(np.min(distance_to_upper))
        features['min_distance_to_lower'] = float(np.min(distance_to_lower))
        
        # レンジ中心からの距離
        range_center = (upper_limit + lower_limit) / 2
        range_width = upper_limit - lower_limit
        
        features['mean_distance_to_center'] = float(np.mean(np.abs(sequence - range_center)))
        
        if range_width > 1e-10:  # 安全な閾値
            rel_pos = np.mean((sequence - lower_limit) / range_width)
            features['relative_position'] = float(rel_pos) if np.isfinite(rel_pos) else 0.5
        else:
            features['relative_position'] = 0.5
        
        # レンジ内滞在率
        in_range = (sequence >= lower_limit) & (sequence <= upper_limit)
        features['in_range_ratio'] = np.mean(in_range)
        
        # レンジ逸脱回数
        features['out_of_range_count'] = np.sum(~in_range)
        
        # 連続してレンジ内にいる最大日数
        consecutive_in_range = []
        current_streak = 0
        for val in in_range:
            if val:
                current_streak += 1
            else:
                if current_streak > 0:
                    consecutive_in_range.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            consecutive_in_range.append(current_streak)
        
        features['max_consecutive_in_range'] = max(consecutive_in_range) if consecutive_in_range else 0
        
        # 上限/下限超過の割合
        features['above_upper_ratio'] = np.mean(sequence > upper_limit)
        features['below_lower_ratio'] = np.mean(sequence < lower_limit)
        
        return features
    
    def extract_volatility_features(self, sequence: np.ndarray) -> Dict[str, float]:
        """
        変動性特徴量の抽出（NaN/Inf安全版）
        
        Args:
            sequence: 時系列データ [seq_len]
            
        Returns:
            特徴量の辞書
        """
        features = {}
        
        # 入力データのクリーン
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 差分系列
        diff = np.diff(sequence)
        diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        
        features['diff_mean'] = float(np.mean(diff))
        features['diff_std'] = float(np.std(diff))
        features['diff_abs_mean'] = float(np.mean(np.abs(diff)))
        
        # ローリング統計量（7日、14日、30日）
        for window in [7, 14, 30]:
            if len(sequence) >= window:
                rolling_std = []
                for i in range(len(sequence) - window + 1):
                    window_std = float(np.std(sequence[i:i+window]))
                    if np.isfinite(window_std):
                        rolling_std.append(window_std)
                
                if len(rolling_std) > 0:
                    features[f'rolling_std_{window}d_mean'] = float(np.mean(rolling_std))
                    features[f'rolling_std_{window}d_max'] = float(np.max(rolling_std))
                else:
                    features[f'rolling_std_{window}d_mean'] = 0.0
                    features[f'rolling_std_{window}d_max'] = 0.0
            else:
                features[f'rolling_std_{window}d_mean'] = 0.0
                features[f'rolling_std_{window}d_max'] = 0.0
        
        # 最大ドローダウン（ピークからの最大下落）
        cummax = np.maximum.accumulate(sequence)
        drawdown = cummax - sequence
        drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
        features['max_drawdown'] = float(np.max(drawdown))
        features['mean_drawdown'] = float(np.mean(drawdown))
        
        return features
    
    def extract_all_features(
        self,
        sequence: np.ndarray,
        upper_limit: float = None,
        lower_limit: float = None
    ) -> Dict[str, float]:
        """
        全特徴量の抽出
        
        Args:
            sequence: 時系列データ [seq_len]
            upper_limit: 上限値（オプション）
            lower_limit: 下限値（オプション）
            
        Returns:
            全特徴量の辞書
        """
        all_features = {}
        
        # 入力シーケンスのクリーン（最初に実行）
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 各種特徴量を抽出
        all_features.update(self.extract_statistical_features(sequence))
        all_features.update(self.extract_trend_features(sequence))
        
        # レンジ関連特徴量（レンジ情報がある場合のみ）
        if upper_limit is not None and lower_limit is not None:
            all_features.update(self.extract_range_features(sequence, upper_limit, lower_limit))
        
        all_features.update(self.extract_volatility_features(sequence))
        
        # 最終的なNaN/Infチェック（全特徴量に対して）
        for key, value in all_features.items():
            if not np.isfinite(value):
                all_features[key] = 0.0  # NaN/Infは0.0に置換
        
        return all_features
    
    def enrich_training_samples(self, input_path: str = None, output_path: str = None):
        """
        訓練サンプルに特徴量を追加
        
        Args:
            input_path: 入力ファイルパス
            output_path: 出力ファイルパス
        """
        if input_path is None:
            input_path = PROCESSED_DATA_DIR / "training_samples.csv"
        else:
            input_path = Path(input_path)
        
        if output_path is None:
            output_path = PROCESSED_DATA_DIR / "training_samples_enriched.csv"
        else:
            output_path = Path(output_path)
        
        print("="*70)
        print("特徴量エンジニアリング")
        print("="*70)
        
        print(f"\n📂 Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        print(f"✓ Loaded {len(df):,} samples")
        
        print("\n🔧 Extracting features...")
        
        enriched_features = []
        
        for idx, row in df.iterrows():
            # 系列データ取得
            if isinstance(row['values_sequence'], str):
                import ast
                values = ast.literal_eval(row['values_sequence'])
            else:
                values = row['values_sequence']
            
            sequence = np.array(values, dtype=np.float32)
            
            # レンジ情報の取得（存在する場合のみ）
            upper_limit = row.get('upper_limit', None)
            lower_limit = row.get('lower_limit', None)
            
            # 特徴量抽出
            features = self.extract_all_features(
                sequence,
                upper_limit=upper_limit,
                lower_limit=lower_limit
            )
            
            enriched_features.append(features)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(df)} samples...")
        
        # 特徴量をDataFrameに変換
        features_df = pd.DataFrame(enriched_features)
        
        # 元のDataFrameと結合
        enriched_df = pd.concat([df, features_df], axis=1)
        
        # 保存
        enriched_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ Feature extraction completed")
        print(f"  Original features: {len(df.columns)}")
        print(f"  New features: {len(features_df.columns)}")
        print(f"  Total features: {len(enriched_df.columns)}")
        print(f"\n💾 Saved to: {output_path}")
        
        # 特徴量の統計情報
        print("\n📊 Feature statistics:")
        print(features_df.describe())
        
        print("\n" + "="*70)
        print("✅ 完了!")
        print("="*70)
        
        return enriched_df


def main():
    """メイン処理"""
    engineer = FeatureEngineer()
    enriched_df = engineer.enrich_training_samples()
    
    # テストサンプルにも同じ処理を適用
    print("\n\n📂 Processing test samples...")
    test_input = PROCESSED_DATA_DIR / "test_samples.csv"
    test_output = PROCESSED_DATA_DIR / "test_samples_enriched.csv"
    
    if test_input.exists():
        engineer.enrich_training_samples(
            input_path=test_input,
            output_path=test_output
        )
    else:
        print("⚠ Test samples not found. Skipping.")


if __name__ == "__main__":
    main()
