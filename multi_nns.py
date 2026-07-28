import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import sys
sys.path.append('./funcs')
sys.path.append('./data')

from funcs.tools import *

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Assume X shape: (360, 165, 5), y shape: (360,)
# X = your data, y = your labels

def create_cnn_lstm_attention_model(input_shape=(165, 5)):
    """
    Hybrid model: CNN for local feature extraction + LSTM for sequential patterns + Attention
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers for local feature extraction
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.1)(x)
    
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv1D(256, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Bidirectional LSTM for capturing temporal/sequential dependencies
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    #x = layers.Dropout(0.1)(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.1)(x)
    
    # Self-attention mechanism
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    attention = layers.Dropout(0.3)(attention)
    x = layers.Add()([x, attention])  # Residual connection
    x = layers.LayerNormalization()(x)
    
    # Global pooling
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Attention')
    model.summary()
    return model



def create_multiscale_cnn_model(input_shape=(165, 5)):
    """
    Multi-scale CNN with parallel convolution branches
    """
    inputs = layers.Input(shape=input_shape)
    
    # Branch 1: Small kernel size
    branch1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    branch1 = layers.BatchNormalization()(branch1)
    branch1 = layers.MaxPooling1D(pool_size=2)(branch1)
    
    # Branch 2: Medium kernel size
    branch2 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.MaxPooling1D(pool_size=2)(branch2)
    
    # Branch 3: Large kernel size
    branch3 = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(inputs)
    branch3 = layers.BatchNormalization()(branch3)
    branch3 = layers.MaxPooling1D(pool_size=2)(branch3)
    
    # Concatenate branches
    x = layers.Concatenate()([branch1, branch2, branch3])
    x = layers.Dropout(0.1)(x)
    
    # Additional CNN layers
    x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Attention
    attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='MultiScale_CNN')
    model.summary()
    return model


def create_transformer_model(input_shape=(165, 5)):
    """
    Transformer encoder for brain feature classification
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial projection
    x = layers.Dense(256)(inputs)
    
    # Positional encoding (simplified)
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embedding = layers.Embedding(input_dim=input_shape[0], output_dim=256)(positions)
    x = x + position_embedding
    
    # Transformer blocks
    for _ in range(4):
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=256, dropout=0.1
        )(x, x)
        attention_output = layers.Dropout(0.1)(attention_output)
        x1 = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward network
        ffn = layers.Dense(512, activation='relu')(x1)
        #ffn = layers.Dropout(0.1)(ffn)
        ffn = layers.Dense(256)(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(x1 + ffn)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='Transformer')
    model.summary()
    return model


def create_resnet_cnn_model(input_shape=(165, 5)):
    """
    ResNet-style CNN with residual connections
    """
    inputs = layers.Input(shape=input_shape)
    
    def residual_block(x, filters, kernel_size=3):
        # Save input for residual connection
        shortcut = x
        
        # First conv layer
        x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Second conv layer
        x = layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut dimensions if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
        
        # Add residual
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    # Initial convolution
    x = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 128)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = residual_block(x, 512)
    x = residual_block(x, 512)
    
    # Attention
    attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='ResNet_CNN')
    model.summary()
    return model


def train_and_evaluate(X, y, model_fn, model_name):
    """
    Complete training and evaluation pipeline
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train.astype(int))}, Test: {np.bincount(y_test.astype(int))}")
    
    # Create model
    model = model_fn(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    # Class weights for imbalanced data
    class_weights = {
        0: len(y_train) / (2 * np.sum(y_train == 0)),
        1: len(y_train) / (2 * np.sum(y_train == 1))
    }
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=16,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"\n{model_name} Test Results:")
    print(f"  Loss: {test_results[0]:.4f}")
    print(f"  Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
    print(f"  AUC: {test_results[2]:.4f}")
    print(f"  Precision: {test_results[3]:.4f}")
    print(f"  Recall: {test_results[4]:.4f}")
    
    # Predictions
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Healthy', 'Autism'])}")
    
    return model, history, test_results


def plot_training_history(results):
    """Plot training history for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, (model_name, result) in enumerate(results.items()):
        history = result['history']
        
        ax1 = axes[idx // 2, idx % 2]
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title(f'{model_name}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":

    
    # Those models are obviously over-complicated.
    
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

    # ------ formal processing ------------------------------------------------------
    # Train all models
    models_to_train = [
        (create_cnn_lstm_attention_model, "CNN-LSTM-Attention"),
        (create_multiscale_cnn_model, "MultiScale-CNN"),
        #(create_transformer_model, "Transformer"),
        (create_resnet_cnn_model, "ResNet-CNN")
    ]

    results = {}
    for model_fn, model_name in models_to_train:
        model, history, test_results = train_and_evaluate(X, y, model_fn, model_name)
        results[model_name] = {
            'model': model,
            'history': history,
            'test_accuracy': test_results[1],
            'test_auc': test_results[2]
        }

    # visualize & analysis:
    plot_training_history(results)

    # Compare results
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name:25s} - Accuracy: {result['test_accuracy']:.4f}, AUC: {result['test_auc']:.4f}")