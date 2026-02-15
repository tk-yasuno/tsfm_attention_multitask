# TSFM Attention Multi-Task: HVAC Anomaly Forecasting

**Time-Series Foundation Model with Temporal Attention and Multi-Task Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 📌 Overview

This project implements a **Hybrid Time-Series Anomaly Forecasting Model** for HVAC equipment, combining IBM Granite Time-Series Foundation Model (TinyTimeMixer) with statistical feature engineering and temporal attention mechanisms.

**Key Innovation**: Multi-task learning for simultaneous 30-day, 60-day, and 90-day anomaly prediction with production-ready performance (F1=0.2789).

---

## 🎯 Project Goal

Predict HVAC equipment anomalies **30, 60, and 90 days in advance** using:
- Historical time-series data (90-day lookback)
- Statistical features (28 dimensions)
- Temporal attention for pattern recognition
- Multi-task learning for efficient shared representation

---

## 🏆 Production Model: v2.2

**Status**: ✅ **PRODUCTION READY**

### Performance Metrics

| Horizon | F1-Score | ROC-AUC | Precision | Recall | Accuracy |
|---------|----------|---------|-----------|--------|----------|
| **30d** | 0.2771 | 0.7230 | 0.2404 | 0.3257 | 0.8318 |
| **60d** | 0.2756 | 0.6981 | 0.2435 | 0.3173 | 0.8349 |
| **90d** | 0.2840 | 0.6958 | 0.2634 | 0.3101 | 0.8486 |
| **Average** | **0.2789** | **0.7056** | 0.2491 | 0.3177 | 0.8384 |

**Model Size**: 195,204 parameters (31.64% trainable)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Layer                             │
│  Time-Series [90 days] + Statistical Features [28 dims]     │
└─────────────────────┬───────────────────────────────────────┘
                      │
              ┌───────┴────────┐
              │                │
    ┌─────────▼──────┐   ┌────▼──────────┐
    │  TinyTimeMixer │   │  Statistical   │
    │   Encoder      │   │   Features     │
    │  [90,1]→[64]   │   │     [28]       │
    │   (Frozen)     │   │                │
    └─────────┬──────┘   └────┬───────────┘
              │               │
              │               │
    ┌─────────▼──────────┐   │
    │ Temporal Attention │   │
    │   (4 heads)        │   │
    │    [64]→[64]       │   │
    └─────────┬──────────┘   │
              │               │
              └───────┬───────┘
                      │
            ┌─────────▼──────────┐
            │  Simple Concat     │
            │   Fusion Layer     │
            │   [64+28]→[92]     │
            └─────────┬──────────┘
                      │
            ┌─────────▼──────────┐
            │  Shared Hidden     │
            │   [92]→[128]       │
            └─────────┬──────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
  ┌─────▼─────┐ ┌────▼─────┐ ┌────▼─────┐
  │  30d Head │ │ 60d Head │ │ 90d Head │
  │ [128]→[1] │ │[128]→[1] │ │[128]→[1] │
  └───────────┘ └──────────┘ └──────────┘
```

### Key Components

1. **TinyTimeMixer Encoder** (Frozen)
   - IBM Granite Time-Series Foundation Model
   - Pre-trained on large-scale time-series data
   - Extracts 64-dimensional embeddings

2. **Temporal Attention** (Trainable)
   - Multi-head self-attention (4 heads)
   - Captures important temporal patterns
   - Enhances interpretability

3. **Statistical Features** (28 dimensions)
   - Time-series statistics (mean, std, trend, etc.)
   - Recent behavior indicators
   - Seasonal & autocorrelation features
   - Distribution features (skewness, kurtosis, entropy)

4. **Simple Concat Fusion**
   - Efficient feature integration
   - Proven superior to complex cross-attention (v2.2.2 experiment)

5. **Multi-Task Learning**
   - Simultaneous 30d/60d/90d prediction
   - Shared representation learning
   - Focal Loss for class imbalance

---

## 📊 Experimental Results

### Model Comparison

| Model | Architecture | Avg F1 | ROC-AUC | Parameters | Status |
|-------|--------------|--------|---------|------------|--------|
| **v2.2** | Simple Concat Fusion | **0.2789** | **0.7056** | 195,204 | ✅ **PRODUCTION** |
| v2.2.2 | Cross-Attention Fusion | 0.2689 | 0.6942 | 218,952 | Experimental |
| v2.3 | No Statistical Features | 0.1973 | 0.5874 | 191,620 | Experimental |

### Key Findings

1. ✅ **Statistical Features are Essential** - Removing them caused -29.2% F1 drop (v2.3)
2. ✅ **Simple Fusion > Complex Fusion** - Cross-Attention underperformed (-3.6% F1) despite +12% params (v2.2.2)
3. ✅ **Feature Engineering > Model Complexity** - Good features matter more than architecture
4. ✅ **Optimal Parameter Count** - v2.2's 195K params is the sweet spot

See [Hybrid_v2.2_v2.3_Lesson.md](Hybrid_v2.2_v2.3_Lesson.md) for detailed analysis.

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
PyTorch 2.6.0+
CUDA 12.4+ (for GPU training)
NVIDIA GPU with 16GB+ VRAM (RTX 4060 Ti or better)
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/tsfm_attension_multitask.git
cd tsfm_attension_multitask

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train v2.2 (Production Model)
python train_hybrid_model_v2_2.py

# Training takes ~25 epochs on RTX 4060 Ti (16GB)
# Model saved to: models/hybrid_model_v2.2/pytorch_model_multitask.pt
```

### Inference

```python
import torch
from pathlib import Path
from train_hybrid_model_v2_2 import MultiTaskHybridModel
from granite_ts_model import GraniteTimeSeriesClassifier

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = Path("models/hybrid_model_v2.2/pytorch_model_multitask.pt")

# ... (see models/README_PRODUCTION.md for complete code)
```

### Visualization

```bash
# Visualize v2.2 predictions
python visualize_forecast_v2_2.py

# Output: results/forecast_comparison_v2.2_[timestamp].png
```

---

## 📁 Project Structure

```
tsfm_attension_multitask/
├── train_hybrid_model_v2_2.py     # Production model training ✅
├── train_hybrid_model_v2_2_2.py   # Cross-Attention experiment
├── train_hybrid_model_v2_3.py     # No-features experiment
├── visualize_forecast_v2_2.py     # v2.2 visualization
├── visualize_forecast_v2_3.py     # v2.3 visualization
├── granite_ts_model.py            # TinyTimeMixer wrapper
├── config.py                      # Configuration
├── requirements.txt               # Dependencies
├── Hybrid_v2.2_v2.3_Lesson.md    # Detailed analysis & lessons
├── models/
│   ├── README_PRODUCTION.md       # Production model guide
│   ├── hybrid_model_v2.2/         # v2.2 model ✅
│   ├── hybrid_model_v2.2.2/       # v2.2.2 model
│   └── hybrid_model_v2.3/         # v2.3 model
├── results/
│   ├── training_history_v2.2.json
│   ├── training_history_v2.2.2.json
│   └── training_history_v2.3.json
└── data/
    └── processed/
        ├── training_samples_enriched.csv
        └── test_samples_enriched.csv
```

---

## 🔬 Experiments & Lessons Learned

### v2.2 (PRODUCTION) ✅
- **Simple Concat Fusion**: Embeddings [64] + Features [28] → [92]
- **Result**: Best performance (F1=0.2789)
- **Lesson**: Simplicity wins when features are well-engineered

### v2.2.2 (Failed Experiment)
- **Cross-Attention Fusion**: Bidirectional attention between embeddings and features
- **Result**: Worse performance (F1=0.2689, -3.6%)
- **Lesson**: Complex attention overkill for 28-dimensional features

### v2.3 (Failed Experiment)
- **No Statistical Features**: TinyTimeMixer embeddings only
- **Result**: Severe degradation (F1=0.1973, -29.2%)
- **Lesson**: Domain-specific statistical features are essential

See [Hybrid_v2.2_v2.3_Lesson.md](Hybrid_v2.2_v2.3_Lesson.md) for comprehensive analysis.

---

## 💡 Key Insights

### 1. Feature Engineering is Critical
Statistical features (mean, std, trend, seasonality) provide **explicit domain knowledge** that deep learning alone cannot capture.

### 2. Architecture Should Match Problem Scale
- 28-dimensional features → Simple Concat Fusion ✅
- 100+ dimensional features → Cross-Attention might help
- Complexity ≠ Better Performance

### 3. Transfer Learning + Domain Knowledge
- Pre-trained TinyTimeMixer (frozen) provides strong time-series representations
- Statistical features add HVAC-specific anomaly detection logic
- Combination is more powerful than either alone

### 4. Multi-Task Learning Benefits
- Shared encoder learns common patterns across horizons
- Parameter efficiency (1 model vs 3 separate models)
- Improved generalization through implicit regularization

---

## 🛠️ Technologies

- **Foundation Model**: IBM Granite Time-Series (TinyTimeMixer)
- **Framework**: PyTorch 2.6.0
- **Hardware**: NVIDIA RTX 4060 Ti (16GB VRAM)
- **Loss Function**: Focal Loss (γ=3.0) for class imbalance
- **Optimizer**: AdamW with Cosine Annealing
- **Attention**: Multi-Head Self-Attention (4 heads)

---

## 📈 Training Details

### Hyperparameters (v2.2)

```python
{
    "epochs": 25,
    "batch_size": 128,
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "focal_gamma": 3.0,
    "embed_dim": 64,
    "hidden_dim": 128,
    "num_attention_heads": 4,
    "dropout": 0.3,
    "lookback_days": 90
}
```

### Training Time
- ~25 epochs on NVIDIA RTX 4060 Ti (16GB)
- Convergence: Epoch 25 (Best F1: 0.2789)
- Total training time: ~2-3 hours

### Dataset
- Training samples: 58,300
- Test samples: 8,745
- Time-series length: 90 days
- Statistical features: 28 dimensions
- Anomaly rate: ~9% (class imbalance handled by Focal Loss)

---

## 📚 Documentation

- **Production Guide**: [models/README_PRODUCTION.md](models/README_PRODUCTION.md)
- **Lessons Learned**: [Hybrid_v2.2_v2.3_Lesson.md](Hybrid_v2.2_v2.3_Lesson.md)
- **Training Logs**: `results/training_history_v2.2.json`

---

## 🔮 Future Work

1. **Feature Selection**
   - Reduce 28d → 15-20d using SHAP analysis
   - Maintain performance while improving efficiency

2. **Model Optimization**
   - ONNX export for production deployment
   - Quantization for faster inference
   - REST API wrapper

3. **Extended Horizons**
   - 120-day, 180-day forecasts
   - Hierarchical multi-task learning

4. **Transfer to Other Domains**
   - Apply to other equipment types
   - Generalize to industrial IoT anomaly detection

---

## 📊 Results Visualization

![Multi-Task Forecast Comparison](results/forecast_comparison_v2.2_example.png)

*(Run `python visualize_forecast_v2_2.py` to generate)*

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **IBM Granite Time-Series Team** - TinyTimeMixer foundation model
- **HuggingFace** - Transformers library
- **PyTorch Team** - Deep learning framework

---

## 📞 Contact

For questions or collaboration:
- Open an issue on GitHub
- Check documentation in [models/README_PRODUCTION.md](models/README_PRODUCTION.md)

---

**Date**: 2026-02-15  
**Status**: ✅ Production Ready  
**Version**: v2.2 (Confirmed)

---

*"The best model is not the most complex, nor the simplest, but the one that captures the right information with the right architecture."*
