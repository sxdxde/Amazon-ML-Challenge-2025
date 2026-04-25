
# 🏆 Amazon ML Challenge 2025: Smart Product Pricing

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

**Team RandomForest** | *Achieving 43.60897526% SMAPE through Advanced Multimodal Fusion (as per Challenge Evaluation)*

<img width="1600" height="1034" alt="amazon" src="https://github.com/user-attachments/assets/68872fa0-a709-4af4-8230-cded0fbfa382" />

[Features](#-key-features) • [Architecture](#-model-architecture) • [Results](#-performance) • [Installation](#-installation) • [Usage](#-usage)

</div>

---

## 📊 Challenge Overview

The Amazon ML Challenge 2025 focuses on predicting  Amazon product prices using multimodal data - combining text descriptions and product images. Our solution leverages state-of-the-art deep learning techniques to achieve competitive performance.

### 🎯 Competition Metrics
- **Primary Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)
- **Our Score:** 46.73% (7-Fold CV) (as per the algorithm)
- **Approach:** Enhanced Multimodal Fusion with Cross-Attention

---

## ✨ Key Features

- 🔄 **Bidirectional Cross-Attention** - Text-to-image and image-to-text attention mechanisms
- 🚪 **Gated Fusion Network** - Learnable gates for optimal information flow
- 🎯 **SMAPE-Optimized Loss** - Direct optimization of evaluation metric
- 🛠️ **Advanced Feature Engineering** - 15+ handcrafted features from text and metadata
- 📈 **Robust Preprocessing** - QuantileTransformer + RobustScaler for outlier handling
- 🔁 **7-Fold Cross-Validation** - Comprehensive generalization testing

---

## 🏗️ Model Architecture

Our solution employs an **Enhanced Multimodal Fusion MLP** that intelligently combines three data sources:

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                              │
├──────────────┬──────────────────┬─────────────────────────┤
│ Text Embed   │  Image Embed     │  Engineered Features    │
│  (384-dim)   │   (512-dim)      │      (165-dim)          │
└──────┬───────┴────────┬─────────┴───────────┬─────────────┘
       │                │                     │
       ▼                ▼                     ▼
┌──────────────┐ ┌──────────────┐  ┌──────────────────┐
│Text Encoder  │ │Image Encoder │  │ Other Encoder    │
│  2-Layer MLP │ │  2-Layer MLP │  │   2-Layer MLP    │
│  (1024-dim)  │ │  (1024-dim)  │  │   (192-dim)      │
└──────┬───────┘ └──────┬───────┘  └────────┬─────────┘
       │                │                    │
       │    ┌───────────┴───────────┐       │
       │    │ Cross-Attention Layer │       │
       │    │  - Text → Image       │       │
       │    │  - Image → Text       │       │
       │    │  (8 attention heads)  │       │
       │    └───────────┬───────────┘       │
       │                │                    │
       └────────┬───────┴────────────────────┘
                │
                ▼
         ┌─────────────┐
         │Gated Fusion │
         │  + Residual │
         └──────┬──────┘
                │
                ▼
    ┌───────────────────────┐
    │   Fusion Network      │
    │  - Layer 1: 2048-dim  │
    │  - Layer 2: 1024-dim  │
    │  - Layer 3: 512-dim   │
    └───────────┬───────────┘
                │
                ▼
         ┌─────────────┐
         │Output Layer │
         │ Price (log) │
         └─────────────┘
```

### 🧩 Component Details

| Component | Input Dim | Output Dim | Parameters |
|-----------|-----------|------------|------------|
| Text Encoder | 384 | 1024 | ~1.18M |
| Image Encoder | 512 | 1024 | ~1.31M |
| Other Encoder | 165 | 192 | ~0.04M |
| Cross-Attention | 1024 | 1024 | ~2.1M |
| Fusion Network | 2240 | 1 | ~6.3M |
| **Total** | - | - | **~10.93M** |

---

## 📈 Performance

### Cross-Validation Results

| Fold | SMAPE (%) | Status |
|------|-----------|--------|
| Fold 1 | 47.11 | ✅ |
| Fold 2 | 46.64 | ✅ |
| Fold 3 | 46.48 | ✅ |
| Fold 4 | 46.73 | ✅ |
| Fold 5 | 46.65 | ✅ |
| Fold 6 | 47.32 | ✅ |
| Fold 7 | 46.16 | ✅ |
| **Mean** | **46.73** | **🎯** |
| **Std Dev** | **0.36** | **📊** |

### Test Set Predictions

```
Min Price:    $0.30
Max Price:    $375.96
Mean Price:   $20.56
Median Price: $13.39
```

### Performance Evolution

```
Initial Baseline (Simple MLP):           52.5% SMAPE
+ Cross-Attention:                       49.8% SMAPE  (-2.7%)
+ Feature Engineering:                   48.2% SMAPE  (-1.6%)
+ SMAPE Loss Optimization:               47.1% SMAPE  (-1.1%)
+ Robust Scaling & 7-Fold CV:           46.7% SMAPE  (-0.4%)
```

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
CUDA 11.0+ (optional, for GPU acceleration)
```

### Setup

```bash
# Clone the repository
git clone https://github.com/sxdxde/Amazon-ML-Challenge-2025.git
cd Amazon-ML-Challenge-2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

---

## 💻 Usage

### Quick Start

```python
# Run the enhanced MLP fusion model
python enhanced_mlp_fusion.py
```

### Fast Training Mode (< 1 hour)

```python
# Run optimized fast version
python fast_mlp_fusion.py

# Expected runtime: ~45-60 minutes on M1 Pro
# Expected SMAPE: ~42-45%
```

### Training Configuration

```python
# Customize hyperparameters
EPOCHS = 250
BATCH_SIZE = 192
LEARNING_RATE = 1.5e-4
N_FOLDS = 7
HIDDEN_DIM = 1024
DROPOUT = 0.3
```

### Expected Output

```
======================================================================
🚀 ENHANCED MLP FUSION: Maximum Feature Extraction
======================================================================

[1/5] Loading data and embeddings...
✓ Loaded embeddings

[2/5] Engineering advanced features...
✓ Enhanced features: 165 dimensions

[3/5] Applying robust scaling...
✓ Scaling complete

[4/5] Training with K-Fold CV...
──────────────────────────────────────────────────────────────────
📊 FOLD 1/7
──────────────────────────────────────────────────────────────────
  Using device: mps
    Epoch 10: train_loss=0.30413, val_loss=0.36193
    ...
  📈 Fold 1 SMAPE: 47.1083%

[5/5] Final evaluation and submission...
🎯 FINAL OOF SMAPE: 46.7276%

✅ Submission saved: enhanced_mlp_fusion_submission.csv
```

---

## 🗂️ Project Structure

```
Amazon-ML-Challenge-2025/
│
├── data/
│   ├── train.csv                              # Training data
│   ├── test.csv                               # Test data
│   ├── final_X_train_medium_with_brand.npy    # Training embeddings
│   └── final_X_test_medium_with_brand.npy     # Test embeddings
│
├── models/
│   ├── enhanced_mlp_fusion.py                 # Main model (150min)
│   └── fast_mlp_fusion.py                     # Fast model (60min)
│
├── submissions/
│   └── enhanced_mlp_fusion_submission.csv     # Final predictions
│
├── notebooks/
│   ├── EDA.ipynb                              # Exploratory analysis
│   └── feature_engineering.ipynb              # Feature extraction
│
├── requirements.txt                           # Dependencies
├── README.md                                  # This file
└── LICENSE                                    # MIT License
```

---

## 🧪 Technical Deep Dive

### 1. Feature Engineering

Our feature engineering pipeline extracts 15+ features:

**Text Statistics:**
- Title length
- Word count
- Average word length
- Uppercase ratio
- Number of digits

**Brand Features:**
- Brand frequency encoding
- Brand mean price
- Brand price std deviation

**Semantic Indicators:**
- Premium keywords (luxury, pro, ultra, etc.)
- Budget keywords (basic, lite, eco, etc.)
- Special character presence

**Quantity Features:**
- Raw quantity
- Log-transformed quantity
- Squared quantity

### 2. Loss Function

Our custom combined loss directly optimizes SMAPE:

```python
def combined_loss(pred, target, alpha=0.6):
    """
    60% SMAPE loss (direct metric optimization)
    40% Huber loss (training stability)
    """
    smape = smape_loss(pred, target)
    huber = nn.SmoothL1Loss()(pred, target)
    return alpha * smape + (1 - alpha) * huber
```

### 3. Cross-Attention Mechanism

Bidirectional attention captures complementary information:

```python
# Text attends to image
text_att = MultiheadAttention(text_enc, image_enc, image_enc)

# Image attends to text
image_att = MultiheadAttention(image_enc, text_enc, text_enc)

# Combine with residuals
text_combined = text_enc + text_att
image_combined = image_enc + image_att
```

### 4. Optimization Strategy
Learning rate of 1.5e-4 "sweet spot" for transformers; stable for our MLP as well.

**Training Pipeline:**
- Optimizer: AdamW (lr=1.5e-4, wd=5e-5) 
- Scheduler: CosineAnnealingWarmRestarts
- Early stopping: 25 epochs patience
- Gradient clipping: 1.0
- Batch size: 192

---

## 🎓 Key Insights

### What Worked

✅ **Cross-modal attention** - Captures text-image relationships  
✅ **SMAPE-optimized loss** - Direct metric optimization,symmetric penalization beats proxy losses  
✅ **Gated fusion** - Learnable gates improve feature selection  
✅ **QuantileTransformer** - Better outlier handling than StandardScaler  
✅ **7-Fold CV** - Lower variance than 5-fold,CV in general reduces risk of overfitting 

### What Didn't Work

❌ Simple concatenation fusion  
❌ MSE loss without SMAPE component  
❌ StandardScaler for embeddings  
❌ Single attention direction  
❌ Fewer than 5 folds  

---

## 🔮 Future Improvements

1. **Pre-trained Models**
   - Use CLIP for image-text alignment
   - Fine-tune BERT for product descriptions

2. **Ensemble Methods**
   - Combine Transformer + MLP + Gradient Boosting
   - Stack multiple architectures

3. **External Data**
   - Market trends and seasonality
   - Competitor pricing data
   - Product category hierarchies

4. **Advanced Techniques**
   - Self-attention within modalities
   - Graph Neural Networks for product relationships
   - Reinforcement learning for price optimization

5. **Optimization**
   - Neural Architecture Search (NAS)
   - AutoML with Optuna
   - Knowledge distillation

---

## 👥 Team

**Team RandomForest**

| Member | Role | Contribution |
|--------|------|--------------|
| Sudarshan Sudhakar | Team Leader | Feature engineering, Model Training, Fine Tuning |


---
## 📝 Citation

If you find this work useful, please cite:

```bibtex
@misc{randomforest2025amazon,
  title        = {Enhanced Multimodal Fusion for Amazon Product Pricing},
  author       = {Team RandomForest},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/sxdxde/Amazon-ML-Challenge-2025}
}
```

---



## 🙏 Acknowledgments

- Amazon ML Challenge 2025 organizers
- Dr.Sivaselvan for providing us with workstations for the challenge
- PyTorch team for excellent deep learning framework
- scikit-learn contributors for preprocessing tools
- Open source community for inspiration and support

---

## 📧 Contact

For questions or collaboration:

- GitHub: [@sxdxde](https://github.com/sxdxde)
- Email: cs23b2007@iiitdm.ac.in

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

Made with ❤️ by Team RandomForest

</div>
