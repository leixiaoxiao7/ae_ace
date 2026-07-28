import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('./funcs')
sys.path.append('./data')

from funcs.tools import *

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic data (replace with your actual data)
def generate_synthetic_data():
    """
    Generate synthetic data for demonstration
    Replace this with your actual data loading
    """
    n_subjects = 360
    n_regions = 165
    n_features = 5
    
    # Create synthetic data with some pattern
    X = np.random.randn(n_subjects, n_regions, n_features)
    
    # Create labels (binary classification)
    y = np.random.randint(0, 2, n_subjects)
    
    # Add some pattern to make classification possible
    X[y == 1] += np.random.randn(np.sum(y == 1), n_regions, n_features) * 0.3
    
    return X, y

# Load and preprocess data
def prepare_data(X, y, test_size=0.2):
    """
    Prepare data for training
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Normalize features per feature dimension
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    X_train = X_train_scaled.reshape(X_train.shape)
    X_test = X_test_scaled.reshape(X_test.shape)
    
    return X_train, X_test, y_train, y_test

# Model 1: LSTM-based model
def create_lstm_model(input_shape):
    """
    LSTM model with bidirectional layers and attention
    """
    inputs = layers.Input(shape=input_shape)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Attention mechanism
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='LSTM_Model')
    return model

# Model 2: 1D CNN model
def create_cnn_model(input_shape):
    """
    1D CNN model for sequence classification
    """
    inputs = layers.Input(shape=input_shape)
    
    # Conv1D blocks
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_Model')
    return model

# Model 3: Transformer-based model
def create_transformer_model(input_shape):
    """
    Transformer model with positional encoding
    """
    inputs = layers.Input(shape=input_shape)
    
    # Positional encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embedding = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
    
    # Project input features
    x = layers.Dense(64)(inputs)
    
    # Add positional encoding
    x = x + position_embedding
    
    # Transformer blocks
    for _ in range(3):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=64, dropout=0.1
        )(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed forward network
        ffn_output = layers.Dense(128, activation='relu')(x)
        ffn_output = layers.Dropout(0.2)(ffn_output)
        ffn_output = layers.Dense(64)(ffn_output)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Transformer_Model')
    return model

# Model 4: CNN-LSTM hybrid
def create_cnn_lstm_model(input_shape):
    """
    Hybrid model combining CNN and LSTM
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN feature extraction
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # LSTM sequence modeling
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Attention
    attention = layers.Attention()([x, x])
    x = layers.Add()([x, attention])
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Model')
    return model

# Model 5: GRU with Attention
def create_gru_attention_model(input_shape):
    """
    GRU model with custom attention mechanism
    """
    inputs = layers.Input(shape=input_shape)
    
    # GRU layers
    x = layers.GRU(128, return_sequences=True)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    
    # Custom attention layer
    attention_weights = layers.Dense(1, activation='tanh')(x)
    attention_weights = layers.Flatten()(attention_weights)
    attention_weights = layers.Activation('softmax')(attention_weights)
    attention_weights = layers.RepeatVector(64)(attention_weights)
    attention_weights = layers.Permute([2, 1])(attention_weights)
    
    # Apply attention
    x = layers.Multiply()([x, attention_weights])
    x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='GRU_Attention_Model')
    return model

# Model 6: Ensemble model
def create_ensemble_model(input_shape):
    """
    Ensemble of multiple architectures
    """
    inputs = layers.Input(shape=input_shape)
    
    # Branch 1: CNN
    cnn_branch = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    cnn_branch = layers.BatchNormalization()(cnn_branch)
    cnn_branch = layers.GlobalMaxPooling1D()(cnn_branch)
    
    # Branch 2: LSTM
    lstm_branch = layers.LSTM(64, return_sequences=False)(inputs)
    lstm_branch = layers.BatchNormalization()(lstm_branch)
    
    # Branch 3: GRU
    gru_branch = layers.GRU(64, return_sequences=False)(inputs)
    gru_branch = layers.BatchNormalization()(gru_branch)
    
    # Combine branches
    combined = layers.Concatenate()([cnn_branch, lstm_branch, gru_branch])
    
    # Final layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Ensemble_Model')
    return model

# Training function
def train_model(model, X_train, y_train, X_test, y_test, model_name, epochs=100):
    """
    Train and evaluate a model
    """
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    checkpoint = ModelCheckpoint(
        f'{model_name}_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n{model_name} Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    return history, test_accuracy

# Visualization function
def plot_training_history(histories, model_names):
    """
    Plot training history for all models
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for history, name in zip(histories, model_names):
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label=f'{name} Train')
        axes[0].plot(history.history['val_accuracy'], label=f'{name} Val', linestyle='--')
    
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    for history, name in zip(histories, model_names):
        # Loss plot
        axes[1].plot(history.history['loss'], label=f'{name} Train')
        axes[1].plot(history.history['val_loss'], label=f'{name} Val', linestyle='--')
    
    axes[1].set_title('Model Loss Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Main execution
def main(X,y):
    # Load data (replace with your actual data)
    print("Loading data...")
    #X, y = generate_synthetic_data()  # Replace with your actual data loading
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X, y, test_size=0.2)
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])  # (165, 5)
    
    # Define models
    models_dict = {
        'LSTM': create_lstm_model,
        'CNN': create_cnn_model,
        'Transformer': create_transformer_model,
        'CNN_LSTM': create_cnn_lstm_model,
        'GRU_Attention': create_gru_attention_model,
        'Ensemble': create_ensemble_model
    }
    
    histories = []
    results = {}
    
    # Train each model
    for name, model_func in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Training {name} Model")
        print('='*50)
        
        # Create model
        model = model_func(input_shape)
        print(f"Model parameters: {model.count_params():,}")
        
        # Train model
        history, test_acc = train_model(
            model, X_train, y_train, X_test, y_test, name, epochs=100
        )
        
        histories.append(history)
        results[name] = test_acc
        
        # Clear session to free memory
        keras.backend.clear_session()
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        status = "✓" if acc >= 0.8 else "✗"
        print(f"{name:20s}: {acc:.4f} {status}")
    
    # Plot comparison
    plot_training_history(histories, list(models_dict.keys()))
    
    # Find best model
    best_model = max(results, key=results.get)
    print(f"\nBest performing model: {best_model} with accuracy: {results[best_model]:.4f}")
    
    return results

# Advanced techniques for improving accuracy
def advanced_training_techniques(X_train, y_train, X_test, y_test):
    """
    Additional techniques to achieve >80% accuracy
    """
    print("\n" + "="*50)
    print("ADVANCED TECHNIQUES")
    print("="*50)
    
    # 1. Data Augmentation
    def augment_data(X, y, noise_level=0.01):
        """Add Gaussian noise for augmentation"""
        X_aug = X + np.random.normal(0, noise_level, X.shape)
        return np.vstack([X, X_aug]), np.hstack([y, y])
    
    # 2. Class Weight Balancing
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # 3. K-Fold Cross Validation
    from sklearn.model_selection import StratifiedKFold
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create best model with enhanced architecture
    def create_enhanced_model(input_shape):
        inputs = layers.Input(shape=input_shape)
        
        # Initial projection
        x = layers.Dense(128)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)
        
        # Parallel processing paths
        # Path 1: Temporal features (LSTM)
        temporal = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        temporal = layers.BatchNormalization()(temporal)
        temporal = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(temporal)
        
        # Path 2: Local features (CNN)
        local = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
        local = layers.BatchNormalization()(local)
        local = layers.Conv1D(64, 5, padding='same', activation='relu')(local)
        local = layers.BatchNormalization()(local)
        
        # Combine paths
        combined = layers.Concatenate()([temporal, local])
        
        # Multi-head attention
        attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)(combined, combined)
        combined = layers.Add()([combined, attention])
        combined = layers.LayerNormalization()(combined)
        
        # Global feature aggregation
        max_pool = layers.GlobalMaxPooling1D()(combined)
        avg_pool = layers.GlobalAveragePooling1D()(combined)
        pooled = layers.Concatenate()([max_pool, avg_pool])
        
        # Deep classification head
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(pooled)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='Enhanced_Model')
        return model
    
    # Train enhanced model
    model = create_enhanced_model(input_shape)
    
    # Custom learning rate schedule
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.001,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.0001
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    # Augment training data
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
        ModelCheckpoint('enhanced_best.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    ]
    
    # Train with class weights
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_split=0.2,
        epochs=150,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nEnhanced Model Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    return model, history, test_accuracy

if __name__ == "__main__":

    outSavePkl='./data/gmwm3D'
   # load 3D gmwm data --------------------------------------------------------------
    data = loadPkl_file(outSavePkl)

    gmwm3D = data['gmwm3D']
    gmwm3D_extend=data['gmwm3D_extend']
    gmwmDF = data['df']
    gmwmDim = data['gmwmDimensions']
    uniqueHeaders = data['uniqueHeaders']
    otherFeaturesH = data['otherFeatureHeaders']
    siteList = data['siteList']

    # specify X and y
    X=gmwm3D
    y=gmwmDF['ace']

    # Run main training
    results = main(X,y)
    
    # If no model achieves >80%, run advanced techniques
    if max(results.values()) < 0.8:
        print("\nRunning advanced techniques to achieve >80% accuracy...")
        #X, y = generate_synthetic_data()
        X_train, X_test, y_train, y_test = prepare_data(X, y, test_size=0.2)
        model, history, accuracy = advanced_training_techniques(X_train, y_train, X_test, y_test)