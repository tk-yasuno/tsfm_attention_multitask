"""
Granite Time Series Model with LoRA
Granite TSモデル実装（LoRA統合）

機能:
1. Granite Time Series Foundation Modelのロード
2. LoRAアダプターの適用
3. バイナリ分類ヘッドの追加
4. マルチホライズン予測（30日、60日、90日）
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

# 各依存関係を個別にテスト
GRANITE_TS_AVAILABLE = True
import_errors = []

try:
    from transformers import AutoModel, AutoConfig
except ImportError as e:
    import_errors.append(f"transformers: {e}")
    GRANITE_TS_AVAILABLE = False

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel
    )
except ImportError as e:
    import_errors.append(f"peft: {e}")
    GRANITE_TS_AVAILABLE = False

try:
    # Granite TS specific imports
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
except ImportError as e:
    import_errors.append(f"tsfm_public: {e}")
    GRANITE_TS_AVAILABLE = False

if not GRANITE_TS_AVAILABLE:
    print(f"[WARNING] Some dependencies not available:")
    for err in import_errors:
        print(f"  - {err}")
    print("Run: pip install transformers peft")
    print("For Granite TS: pip install git+https://github.com/ibm-granite/granite-tsfm.git")

from config import (
    GRANITE_MODEL_NAME,
    CONTEXT_LENGTH,
    LORA_CONFIG,
    FORECAST_HORIZONS,
    LOOKBACK_DAYS,
    USE_GPU,
    GPU_ID
)


class GraniteTimeSeriesClassifier(nn.Module):
    """
    Granite Time Series + LoRA による逸脱予測分類器
    """
    
    def __init__(
        self,
        model_name: str = GRANITE_MODEL_NAME,
        num_horizons: int = len(FORECAST_HORIZONS),
        lora_config: Optional[Dict] = None,
        device: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            model_name: Granite TSモデル名
            num_horizons: 予測ホライズン数
            lora_config: LoRA設定
            device: 計算デバイス
        """
        super().__init__()
        
        # デバイス設定
        if device is None:
            if USE_GPU and torch.cuda.is_available():
                self.device = torch.device(f'cuda:{GPU_ID}')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # LoRA設定
        if lora_config is None:
            lora_config = LORA_CONFIG
        
        self.num_horizons = num_horizons
        self.model_name = model_name
        
        # モデル構築
        self._build_model(lora_config)
        
    def _build_model(self, lora_config: Dict):
        """
        モデル構築
        
        Args:
            lora_config: LoRA設定
        """
        print(f"Building Granite TS model: {self.model_name}")
        
        # Granite TSが利用可能かチェック
        if not GRANITE_TS_AVAILABLE:
            print("  [WARNING] Granite TS not available, using fallback LSTM model...")
            self._build_fallback_model()
            return
        
        try:
            # Granite TS TinyTimeMixerモデルのロード
            print("  Loading Granite TS TinyTimeMixer model...")
            
            from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig
            
            # モデル設定
            config = TinyTimeMixerConfig(
                context_length=LOOKBACK_DAYS,
                prediction_length=max(FORECAST_HORIZONS),
                num_input_channels=1,  # 単変量時系列
                decoder_mode='flatten',  # 分類タスク用
                d_model=64,
                num_layers=4,
                dropout=0.1,
            )
            
            # モデル作成
            self.base_model = TinyTimeMixerForPrediction(config)
            self.hidden_size = config.d_model
            
            print(f"  [OK] Granite TS TinyTimeMixer loaded (d_model={self.hidden_size})")
            
            # LoRA適用
            print("  Applying LoRA...")
            
            # TinyTimeMixerの具体的なLinear層を対象にする
            target_modules = [
                "encoder.patcher",  # パッチャー層
                "mlp.fc1",  # MLPの第1層（すべてのMLP内）
                "mlp.fc2",  # MLPの第2層（すべてのMLP内）
                "attn_layer",  # ゲート付き注意機構層
            ]
            print(f"  Target modules for LoRA: {target_modules}")
            
            peft_config = LoraConfig(
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("lora_alpha", 16),
                target_modules=target_modules,
                lora_dropout=lora_config.get("lora_dropout", 0.1),
                bias=lora_config.get("bias", "none"),
            )
            
            self.model = get_peft_model(self.base_model, peft_config)
            
            # 学習可能パラメータ数を表示
            self.model.print_trainable_parameters()
            
        except Exception as e:
            print(f"  [WARNING] Could not load Granite TS model")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            import traceback
            print(f"  Traceback:")
            traceback.print_exc()
            print("  Using fallback LSTM model for MVP testing...")
            self._build_fallback_model()
            return
        
        # 分類ヘッド追加
        self._add_classification_heads()
    
    def _build_fallback_model(self):
        """
        フォールバックモデル（LSTM）
        Granite TSが利用できない場合のシンプルなLSTMモデル
        """
        self.hidden_size = 128
        self.num_layers = 2
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # 正規化層
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Attention層（簡易）
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,
            batch_first=True
        )
        
        print(f"[OK] Fallback LSTM model built (hidden={self.hidden_size}, layers={self.num_layers})")
        
        # 分類ヘッドも追加
        self._add_classification_heads()
    
    def _add_classification_heads(self):
        """分類ヘッド追加（マルチホライズン）"""
        
        # モデルの出力次元を取得
        try:
            hidden_size = self.config.hidden_size
        except:
            hidden_size = self.hidden_size
        
        # 各ホライズンに対する分類ヘッド
        self.classification_heads = nn.ModuleDict({
            f"head_{h}d": nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size // 4, 1),  # バイナリ分類
                nn.Sigmoid()
            )
            for h in FORECAST_HORIZONS
        })
        
        print(f"[OK] Classification heads added for horizons: {FORECAST_HORIZONS}")
    
    def forward(
        self,
        input_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        順伝播
        
        Args:
            input_sequence: 入力系列 [batch_size, seq_len, 1]
            attention_mask: アテンションマスク [batch_size, seq_len]
            
        Returns:
            各ホライズンの予測確率の辞書
        """
        batch_size = input_sequence.size(0)
        
        # ベースモデルがGranite TSかLSTMかで処理を分岐
        if hasattr(self, 'lstm'):
            # フォールバックLSTMモデル
            lstm_out, (hidden, cell) = self.lstm(input_sequence)
            
            # 最終時刻の出力を使用
            last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
            
            # 正規化
            normalized_output = self.layer_norm(last_output)
            
            # Attention（セルフアテンション）
            attn_output, _ = self.attention(
                lstm_out,
                lstm_out,
                lstm_out
            )
            pooled_output = attn_output[:, -1, :]  # 最終時刻
            
        elif hasattr(self, 'model') and hasattr(self.model, 'base_model'):
            # Granite TS TinyTimeMixerモデル
            # 入力形状を調整: [batch_size, seq_len, num_channels]
            if input_sequence.dim() == 2:
                input_sequence = input_sequence.unsqueeze(-1)
            
            # TinyTimeMixerで予測を生成
            outputs = self.model(
                past_values=input_sequence,
                return_dict=True
            )
            
            # 予測値を特徴量として使用
            # TinyTimeMixerの出力: [batch_size, prediction_length, num_channels]
            predictions = outputs.prediction_outputs if hasattr(outputs, 'prediction_outputs') else outputs[0]
            # print(f"DEBUG: predictions shape = {predictions.shape}")
            
            # 時間軸とチャネル軸を平均プーリングして特徴ベクトルに変換
            # [batch_size, pred_len, channels] -> [batch_size, pred_len*channels]
            batch_size = predictions.size(0)
            pooled_output = predictions.reshape(batch_size, -1)  # フラット化
            # print(f"DEBUG: pooled_output shape = {pooled_output.shape}")
            
            # 線形層で次元を調整（一度だけ初期化）
            if not hasattr(self, 'feature_proj'):
                feature_dim = pooled_output.size(-1)
                self.feature_proj = nn.Linear(feature_dim, self.hidden_size).to(self.device)
                # print(f"DEBUG: Created feature_proj: {feature_dim} -> {self.hidden_size}")
            
            pooled_output = self.feature_proj(pooled_output)
            
        else:
            # 一般的なTransformerモデル
            outputs = self.model(
                inputs_embeds=input_sequence,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # プーリング（最終隠れ状態）
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, -1, :]  # [batch_size, hidden_size]
        
        # 各ホライズンの予測
        predictions = {}
        for h in FORECAST_HORIZONS:
            head_name = f"head_{h}d"
            pred = self.classification_heads[head_name](pooled_output)
            predictions[f"prob_{h}d"] = pred.squeeze(-1)  # [batch_size]
        
        return predictions
    
    def predict(
        self,
        input_sequence: np.ndarray,
        return_probs: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        推論
        
        Args:
            input_sequence: 入力系列 [batch_size, seq_len] or [seq_len]
            return_probs: 確率を返すか（Falseの場合はバイナリラベル）
            
        Returns:
            予測結果の辞書
        """
        self.eval()
        
        # 次元調整
        if input_sequence.ndim == 1:
            input_sequence = input_sequence[np.newaxis, :, np.newaxis]  # [1, seq_len, 1]
        elif input_sequence.ndim == 2:
            input_sequence = input_sequence[:, :, np.newaxis]  # [batch_size, seq_len, 1]
        
        # Tensorに変換
        input_tensor = torch.FloatTensor(input_sequence).to(self.device)
        
        # 推論
        with torch.no_grad():
            predictions = self.forward(input_tensor)
        
        # CPU・NumPyに変換
        results = {}
        for key, value in predictions.items():
            probs = value.cpu().numpy()
            
            if return_probs:
                results[key] = probs
            else:
                # 0.5を閾値にバイナリ化
                results[key.replace('prob', 'label')] = (probs > 0.5).astype(int)
        
        return results
    
    def save_model(self, save_path: str):
        """
        モデル保存
        
        Args:
            save_path: 保存先パス
        """
        print(f"Saving model to: {save_path}")
        
        # LoRAアダプターのみ保存（ベースモデルは保存しない）
        if hasattr(self, 'model') and isinstance(self.model, PeftModel):
            self.model.save_pretrained(save_path)
        else:
            # フォールバックモデルの場合は全体を保存
            torch.save(self.state_dict(), f"{save_path}/model.pt")
        
        print("[OK] Model saved successfully")
    
    def load_model(self, load_path: str):
        """
        モデル読み込み
        
        Args:
            load_path: 読み込み元パス
        """
        print(f"Loading model from: {load_path}")
        
        try:
            # LoRAアダプター読み込み
            self.model = PeftModel.from_pretrained(
                self.base_model,
                load_path
            )
            print("[OK] LoRA adapter loaded")
        except:
            # フォールバックモデル読み込み
            state_dict = torch.load(f"{load_path}/model.pt", map_location=self.device)
            self.load_state_dict(state_dict)
            print("[OK] Model loaded")
        
        self.to(self.device)


def test_model():
    """モデルのテスト"""
    print("="*60)
    print("🧪 Testing Granite TS Classifier")
    print("="*60)
    
    # モデル作成
    model = GraniteTimeSeriesClassifier()
    model.to(model.device)
    
    # ダミーデータでテスト
    batch_size = 4
    seq_len = 90
    
    dummy_input = torch.randn(batch_size, seq_len, 1).to(model.device)
    
    print(f"\n📊 Testing with input shape: {dummy_input.shape}")
    
    # 順伝播テスト
    predictions = model(dummy_input)
    
    print("\n[OK] Forward pass successful")
    print("Predictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape} -> {value[0].item():.4f}")
    
    # NumPy入力でのテスト
    numpy_input = np.random.randn(seq_len)
    results = model.predict(numpy_input)
    
    print("\n[OK] Prediction successful")
    print("Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("[OK] Model test complete!")
    print("="*60)


if __name__ == "__main__":
    test_model()
