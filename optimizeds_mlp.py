# ===================================================================
# ENHANCED MLP FUSION WITH ADVANCED FEATURE ENGINEERING
# Target: SMAPE < 45% with Maximum Data Extraction
# ===================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from tqdm import tqdm
import gc
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 ENHANCED MLP FUSION: Maximum Feature Extraction")
print("="*70)

# ===================================================================
# ADVANCED FEATURE ENGINEERING
# ===================================================================
def extract_advanced_features(df, is_train=True):
    """Extract rich features from catalog_content and other columns"""
    print(f"  Extracting advanced features...")
    
    features = {}
    
    # Text length features
    features['title_len'] = df['catalog_content'].str.len()
    features['word_count'] = df['catalog_content'].str.split().str.len()
    features['avg_word_len'] = features['title_len'] / (features['word_count'] + 1)
    
    # Brand features (if present)
    if 'brand' in df.columns:
        # Brand frequency encoding
        brand_freq = df['brand'].value_counts()
        features['brand_freq'] = df['brand'].map(brand_freq).fillna(0)
        
        # Brand mean price (only for train)
        if is_train and 'price' in df.columns:
            brand_mean = df.groupby('brand')['price'].mean()
            features['brand_mean_price'] = df['brand'].map(brand_mean).fillna(df['price'].median())
    
    # Quantity features
    if 'quantity' in df.columns:
        features['quantity'] = df['quantity'].fillna(1)
        features['log_quantity'] = np.log1p(features['quantity'])
        features['quantity_squared'] = features['quantity'] ** 2
    
    # Text pattern features
    features['has_digits'] = df['catalog_content'].str.contains(r'\d', regex=True).astype(int)
    features['has_special_chars'] = df['catalog_content'].str.contains(r'[^a-zA-Z0-9\s]', regex=True).astype(int)
    features['uppercase_ratio'] = df['catalog_content'].str.count(r'[A-Z]') / (features['title_len'] + 1)
    
    # Extract numeric values from text
    features['num_numbers'] = df['catalog_content'].str.findall(r'\d+').str.len().fillna(0)
    
    # Check for common price indicators
    price_keywords = ['premium', 'luxury', 'pro', 'plus', 'max', 'ultra', 'deluxe']
    budget_keywords = ['basic', 'mini', 'lite', 'eco', 'value']
    
    features['has_premium_word'] = df['catalog_content'].str.lower().str.contains('|'.join(price_keywords)).astype(int)
    features['has_budget_word'] = df['catalog_content'].str.lower().str.contains('|'.join(budget_keywords)).astype(int)
    
    return pd.DataFrame(features)

# ===================================================================
# ENHANCED MLP ARCHITECTURE WITH RESIDUAL CONNECTIONS
# ===================================================================
class EnhancedMultimodalFusionMLP(nn.Module):
    """
    Enhanced fusion with:
    - Residual connections
    - Better attention mechanism
    - Gated fusion
    - More sophisticated architecture
    """
    def __init__(self, text_dim, image_dim, other_dim, hidden_dim=1024, dropout=0.3):
        super().__init__()
        
        # Individual encoders with residual capability
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.7)
        )
        
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.7)
        )
        
        self.other_encoder = nn.Sequential(
            nn.Linear(other_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.GELU()
        )
        
        # Cross-attention between modalities
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Gated fusion mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 192, hidden_dim * 2),  # Changed from 128 to 192
            nn.Sigmoid()
        )
        
        # Main fusion network with residual connections
        fusion_input_dim = hidden_dim * 2 + 192  # Changed from 128 to 192
        self.fusion1 = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.7)
        )
        
        self.fusion3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.output = nn.Linear(hidden_dim // 2, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, text_emb, image_emb, other_emb):
        # Encode each modality
        text_enc = self.text_encoder(text_emb)
        image_enc = self.image_encoder(image_emb)
        other_enc = self.other_encoder(other_emb)
        
        # Bidirectional cross-attention
        text_att, _ = self.text_to_image_attn(
            text_enc.unsqueeze(1), 
            image_enc.unsqueeze(1), 
            image_enc.unsqueeze(1)
        )
        text_att = text_att.squeeze(1)
        
        image_att, _ = self.image_to_text_attn(
            image_enc.unsqueeze(1), 
            text_enc.unsqueeze(1), 
            text_enc.unsqueeze(1)
        )
        image_att = image_att.squeeze(1)
        
        # Combine with residuals
        text_combined = text_enc + text_att
        image_combined = image_enc + image_att
        
        # Concatenate all features
        fused = torch.cat([text_combined, image_combined, other_enc], dim=1)
        
        # Gated fusion
        gate_values = self.gate(fused)
        
        # Apply fusion layers with residuals
        x = self.fusion1(fused)
        x = x * gate_values[:, :x.size(1)]  # Apply gating
        
        x = self.fusion2(x)
        x = self.fusion3(x)
        output = self.output(x)
        
        return output

# ===================================================================
# CUSTOM LOSS FUNCTION - SMAPE-inspired
# ===================================================================
def smape_loss(pred, target, epsilon=1e-3):
    """SMAPE-inspired loss that directly optimizes the evaluation metric"""
    # Work in log space to match our target
    pred_exp = torch.exp(pred)
    target_exp = torch.exp(target)
    
    numerator = torch.abs(pred_exp - target_exp)
    denominator = (torch.abs(target_exp) + torch.abs(pred_exp)) / 2.0 + epsilon
    
    return torch.mean(numerator / denominator)

def combined_loss(pred, target, alpha=0.6):
    """Combine SMAPE loss with Huber loss for stability"""
    smape = smape_loss(pred, target)
    huber = nn.SmoothL1Loss()(pred, target)
    return alpha * smape + (1 - alpha) * huber

# ===================================================================
# TRAINING FUNCTION WITH ADVANCED TECHNIQUES
# ===================================================================
def train_enhanced_mlp(X_text_tr, X_image_tr, X_other_tr, y_tr, 
                       X_text_val, X_image_val, X_other_val, y_val,
                       epochs=250, batch_size=192, lr=1.5e-4):
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    text_dim = X_text_tr.shape[1]
    image_dim = X_image_tr.shape[1]
    other_dim = X_other_tr.shape[1]
    
    model = EnhancedMultimodalFusionMLP(text_dim, image_dim, other_dim).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5, betas=(0.9, 0.999))
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_text_tr),
        torch.FloatTensor(X_image_tr),
        torch.FloatTensor(X_other_tr),
        torch.FloatTensor(y_tr).unsqueeze(1)
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_text_val),
        torch.FloatTensor(X_image_val),
        torch.FloatTensor(X_other_val),
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    best_val_loss = float('inf')
    patience = 25  # Increased patience for more training
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for text_b, image_b, other_b, y_b in train_loader:
            text_b = text_b.to(device)
            image_b = image_b.to(device)
            other_b = other_b.to(device)
            y_b = y_b.to(device)
            
            optimizer.zero_grad()
            output = model(text_b, image_b, other_b)
            loss = combined_loss(output, y_b)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for text_b, image_b, other_b, y_b in val_loader:
                text_b = text_b.to(device)
                image_b = image_b.to(device)
                other_b = other_b.to(device)
                y_b = y_b.to(device)
                
                output = model(text_b, image_b, other_b)
                val_loss += combined_loss(output, y_b).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")
    
    model.load_state_dict(best_model_state)
    return model

# ===================================================================
# PREDICTION FUNCTION
# ===================================================================
def predict_enhanced(model, X_text, X_image, X_other, batch_size=512):
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_text), batch_size), desc="Predicting", leave=False):
            end_idx = min(i + batch_size, len(X_text))
            
            text_b = torch.FloatTensor(X_text[i:end_idx]).to(device)
            image_b = torch.FloatTensor(X_image[i:end_idx]).to(device)
            other_b = torch.FloatTensor(X_other[i:end_idx]).to(device)
            
            output = model(text_b, image_b, other_b)
            predictions.append(output.cpu().numpy())
    
    return np.vstack(predictions).flatten()

# ===================================================================
# MAIN EXECUTION
# ===================================================================
print("\n[1/5] Loading data and embeddings...")
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Load embeddings
X_train_full = np.load("final_X_train_medium_with_brand.npy", allow_pickle=False)
X_test_full = np.load("final_X_test_medium_with_brand.npy", allow_pickle=False)

# Define dimensions
text_dim = 384
image_dim = 512

# Slice embeddings
train_text = X_train_full[:, :text_dim]
train_image = X_train_full[:, text_dim:text_dim+image_dim]
train_other_base = X_train_full[:, text_dim+image_dim:]

test_text = X_test_full[:, :text_dim]
test_image = X_test_full[:, text_dim:text_dim+image_dim]
test_other_base = X_test_full[:, text_dim+image_dim:]

print(f"✓ Loaded embeddings")
del X_train_full, X_test_full
gc.collect()

# ===================================================================
# ADVANCED FEATURE ENGINEERING
# ===================================================================
print("\n[2/5] Engineering advanced features...")
train_extra_features = extract_advanced_features(df_train, is_train=True)
test_extra_features = extract_advanced_features(df_test, is_train=False)

# Combine with existing other features
train_other = np.hstack([train_other_base, train_extra_features.values])
test_other = np.hstack([test_other_base, test_extra_features.values])

print(f"✓ Enhanced features: {train_other.shape[1]} dimensions")
del train_other_base, test_other_base
gc.collect()

# Target transformation
y_train_log = np.log1p(df_train['price'].values)

# ===================================================================
# ADVANCED SCALING
# ===================================================================
print("\n[3/5] Applying robust scaling...")

# Use QuantileTransformer for embeddings (more robust to outliers)
text_scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
image_scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
other_scaler = RobustScaler()

train_text_scaled = text_scaler.fit_transform(train_text)
test_text_scaled = text_scaler.transform(test_text)

train_image_scaled = image_scaler.fit_transform(train_image)
test_image_scaled = image_scaler.transform(test_image)

train_other_scaled = other_scaler.fit_transform(train_other)
test_other_scaled = other_scaler.transform(test_other)

print("✓ Scaling complete")
del train_text, train_image, train_other, test_text, test_image, test_other
gc.collect()

# ===================================================================
# K-FOLD CROSS-VALIDATION WITH STRATIFICATION
# ===================================================================
print("\n[4/5] Training with K-Fold CV...")

N_FOLDS = 7  # Increased from 5 to 7 for better generalization
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_text_scaled))
test_preds = np.zeros(len(test_text_scaled))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_text_scaled), 1):
    print(f"\n{'─'*70}")
    print(f"📊 FOLD {fold}/{N_FOLDS}")
    print(f"{'─'*70}")
    
    model = train_enhanced_mlp(
        train_text_scaled[train_idx], 
        train_image_scaled[train_idx], 
        train_other_scaled[train_idx], 
        y_train_log[train_idx],
        train_text_scaled[val_idx], 
        train_image_scaled[val_idx], 
        train_other_scaled[val_idx], 
        y_train_log[val_idx]
    )
    
    # OOF predictions
    oof_preds[val_idx] = predict_enhanced(
        model, 
        train_text_scaled[val_idx], 
        train_image_scaled[val_idx], 
        train_other_scaled[val_idx]
    )
    
    # Test predictions
    fold_test_preds = predict_enhanced(
        model, 
        test_text_scaled, 
        test_image_scaled, 
        test_other_scaled
    )
    test_preds += fold_test_preds / N_FOLDS
    
    # Calculate fold SMAPE
    val_pred_price = np.expm1(oof_preds[val_idx])
    val_actual_price = np.expm1(y_train_log[val_idx])
    
    fold_smape = np.mean(
        2 * np.abs(val_pred_price - val_actual_price) / 
        (np.abs(val_actual_price) + np.abs(val_pred_price) + 1e-8)
    ) * 100
    
    fold_scores.append(fold_smape)
    print(f"  📈 Fold {fold} SMAPE: {fold_smape:.4f}%")
    
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ===================================================================
# FINAL EVALUATION
# ===================================================================
print("\n[5/5] Final evaluation and submission...")

oof_prices = np.expm1(oof_preds)
actual_prices = df_train['price'].values

overall_smape = np.mean(
    2 * np.abs(oof_prices - actual_prices) / 
    (np.abs(actual_prices) + np.abs(oof_prices) + 1e-8)
) * 100

print("\n" + "="*70)
print(f"📊 CROSS-VALIDATION RESULTS")
print("="*70)
for i, score in enumerate(fold_scores, 1):
    print(f"  Fold {i}: {score:.4f}%")
print(f"\n  Mean: {np.mean(fold_scores):.4f}%")
print(f"  Std:  {np.std(fold_scores):.4f}%")
print("\n" + "="*70)
print(f"🎯 FINAL OOF SMAPE: {overall_smape:.4f}%")
print("="*70)

# ===================================================================
# CREATE SUBMISSION
# ===================================================================
final_predictions = np.expm1(test_preds)
final_predictions = np.clip(final_predictions, 0.01, None)

submission = pd.DataFrame({
    'sample_id': df_test['sample_id'],
    'price': final_predictions
})

submission.to_csv('enhanced_mlp_fusion_submission.csv', index=False)

print("\n✅ Submission saved: enhanced_mlp_fusion_submission.csv")
print("\n📋 Prediction statistics:")
print(f"  Min:    ${final_predictions.min():.2f}")
print(f"  Max:    ${final_predictions.max():.2f}")
print(f"  Mean:   ${final_predictions.mean():.2f}")
print(f"  Median: ${np.median(final_predictions):.2f}")
print("\n" + "="*70)