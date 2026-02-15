# Production Model: Hybrid v2.2

**Status**: ✅ **PRODUCTION READY**  
**Version**: v2.2  
**Last Updated**: 2026-02-15

---

## 🏆 Model Selection Decision

After comprehensive experiments (v2.2, v2.2.2, v2.3), **v2.2** has been confirmed as the production model.

### Performance Summary

| Model | Avg F1 | ROC-AUC | Parameters | Status |
|-------|--------|---------|------------|--------|
| **v2.2** ✅ | **0.2789** | **0.7056** | 195,204 | **PRODUCTION** |
| v2.2.2 | 0.2689 (-3.6%) | 0.6942 | 218,952 | Experimental |
| v2.3 | 0.1973 (-29.2%) | 0.5874 | 191,620 | Experimental |

---

## 📁 Model Files

### v2.2 (PRODUCTION) ✅

**Location**: `hybrid_model_v2.2/`

- **Model Weight**: `pytorch_model_multitask.pt` (195,204 params)
- **Training History**: `../results/training_history_v2.2.json`
- **Training Script**: `../train_hybrid_model_v2_2.py`
- **Visualization Script**: `../visualize_forecast_v2_2.py`

**Architecture**:
```
TinyTimeMixer Encoder (64d, frozen)
    ↓
Temporal Attention (4 heads)
    ↓
Simple Concat Fusion: [Embeddings 64d] + [Statistical Features 28d] = 92d
    ↓
Shared Hidden Layer (92d → 128d)
    ↓
Multi-Task Heads (30d, 60d, 90d)
```

**Performance**:
- 30d: F1=0.2771, ROC-AUC=0.7230, Precision=0.2404, Recall=0.3257
- 60d: F1=0.2756, ROC-AUC=0.6981, Precision=0.2435, Recall=0.3173
- 90d: F1=0.2840, ROC-AUC=0.6958, Precision=0.2634, Recall=0.3101
- **Average**: F1=0.2789, ROC-AUC=0.7056

**Key Features**:
- ✅ Statistical Features (28d) for explicit domain knowledge
- ✅ Simple Concat Fusion (efficient, effective)
- ✅ Optimal parameter count (195,204)
- ✅ Balanced trainable ratio (31.64%)
- ✅ Multi-Task Learning for shared representation
- ✅ Focal Loss for class imbalance handling

---

### v2.2.2 (Experimental)

**Location**: `hybrid_model_v2.2.2/`

**Status**: ❌ Not Recommended for Production

**Reason**: 
- Cross-Attention Fusion added complexity (+12% params)
- Performance degradation (-3.6% F1)
- Overkill for 28-dimensional features

**Lesson Learned**: Simple Concat Fusion > Complex Cross-Attention for low-dimensional features

---

### v2.3 (Experimental)

**Location**: `hybrid_model_v2.3/`

**Status**: ❌ Not Recommended for Production

**Reason**: 
- Removed statistical features (28d)
- Severe performance degradation (-29.2% F1)
- Critical information loss

**Lesson Learned**: Statistical features are essential for anomaly detection

---

## 🚀 Deployment Guide

### Loading the Production Model

```python
import torch
from pathlib import Path
from granite_ts_model import GraniteTimeSeriesClassifier
from train_hybrid_model_v2_2 import MultiTaskHybridModel

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize base model
lora_config = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bias": "none"
}
granite_model = GraniteTimeSeriesClassifier(
    num_horizons=3,
    device=device,
    lora_config=lora_config
)

# Build hybrid model
model = MultiTaskHybridModel(
    granite_model=granite_model,
    stat_feature_dim=28,
    embed_dim=64,
    hidden_dim=128,
    num_attention_heads=4,
    dropout=0.3
)

# Load trained weights
model_path = Path("models/hybrid_model_v2.2/pytorch_model_multitask.pt")
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Get thresholds
thresholds = {
    '30d': checkpoint['metrics']['30d']['threshold'],
    '60d': checkpoint['metrics']['60d']['threshold'],
    '90d': checkpoint['metrics']['90d']['threshold']
}
```

### Inference Example

```python
import numpy as np

# Prepare input
sequence = np.array([...])  # [90] time-series values
features = np.array([...])  # [28] statistical features

# Convert to tensors
sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1)  # [1, 90, 1]
features_tensor = torch.FloatTensor(features).unsqueeze(0)  # [1, 28]

# Inference
with torch.no_grad():
    predictions = model(sequence_tensor.to(device), features_tensor.to(device))
    probs = predictions.cpu().numpy()[0]  # [3] for 30d, 60d, 90d

# Apply thresholds
pred_30d = 1 if probs[0] >= thresholds['30d'] else 0
pred_60d = 1 if probs[1] >= thresholds['60d'] else 0
pred_90d = 1 if probs[2] >= thresholds['90d'] else 0

print(f"30d: {probs[0]:.3f} ({'Anomaly' if pred_30d else 'Normal'})")
print(f"60d: {probs[1]:.3f} ({'Anomaly' if pred_60d else 'Normal'})")
print(f"90d: {probs[2]:.3f} ({'Anomaly' if pred_90d else 'Normal'})")
```

---

## 📊 Evaluation Metrics

### Optimal Thresholds (F1-maximized)

- 30d: 0.230
- 60d: 0.231
- 90d: 0.250

### Performance by Horizon

**30-day Forecast**:
- F1-Score: 0.2771
- Accuracy: 0.8318
- Precision: 0.2404
- Recall: 0.3257
- ROC-AUC: 0.7230

**60-day Forecast**:
- F1-Score: 0.2756
- Accuracy: 0.8349
- Precision: 0.2435
- Recall: 0.3173
- ROC-AUC: 0.6981

**90-day Forecast**:
- F1-Score: 0.2840
- Accuracy: 0.8486
- Precision: 0.2634
- Recall: 0.3101
- ROC-AUC: 0.6958

---

## 🔧 Model Characteristics

### Strengths

1. ✅ **High Interpretability**: Statistical features are explicit and explainable
2. ✅ **Balanced Performance**: Good precision-recall trade-off
3. ✅ **Efficient Architecture**: Optimal parameter count
4. ✅ **Stable Training**: Consistent convergence
5. ✅ **Multi-Horizon**: Simultaneous 30d/60d/90d predictions

### Limitations

1. ⚠️ **Class Imbalance**: Anomaly rate ~9% (Focal Loss mitigates this)
2. ⚠️ **Feature Engineering Required**: 28 statistical features must be computed
3. ⚠️ **Moderate F1-Score**: 0.28 (acceptable for production, room for improvement)

### Recommended Use Cases

- ✅ HVAC equipment anomaly prediction (30-90 days ahead)
- ✅ Time-series with rich statistical features
- ✅ Applications requiring explainable predictions
- ✅ Multi-horizon forecasting tasks

---

## 📚 Documentation

- **Lesson Learned**: [Hybrid_v2.2_v2.3_Lesson.md](../Hybrid_v2.2_v2.3_Lesson.md)
- **Training Log**: [results/training_history_v2.2.json](../results/training_history_v2.2.json)
- **Source Code**: [train_hybrid_model_v2_2.py](../train_hybrid_model_v2_2.py)

---

## 🔄 Version History

- **v2.2** (2026-02-15): Production model confirmed ✅
- **v2.2.2** (2026-02-15): Cross-Attention experiment (not recommended)
- **v2.3** (2026-02-15): No-statistical-features experiment (failed)

---

## 📞 Contact & Support

For questions about the production model:
1. Check [Hybrid_v2.2_v2.3_Lesson.md](../Hybrid_v2.2_v2.3_Lesson.md) for detailed analysis
2. Review training history: `results/training_history_v2.2.json`
3. Run visualization: `python visualize_forecast_v2_2.py`

---

**Last Verified**: 2026-02-15  
**Production Status**: ✅ READY  
**Recommended for**: Deployment
