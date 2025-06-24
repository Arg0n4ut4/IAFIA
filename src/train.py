import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.models.rna import build_model
import os

def train_from_csv(train_csv, val_split=0.2, output_model='data/models/rna.h5'):
    # 1. Carrega dados
    df = pd.read_csv(train_csv)
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # 2. Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Divide em treino/validação
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=val_split, stratify=y, random_state=42
    )

    # 4. Cria o modelo
    model = build_model(input_dim=X.shape[1])

    # 5. Callbacks
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(output_model, save_best_only=True)
    ]

    # 6. Treinamento
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Salvamento
    scaler_path = os.path.join(os.path.dirname(output_model), 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"[OK] Modelo salvo em {output_model}")
    print(f"[OK] Scaler salvo em {scaler_path}")
    return model, scaler

if __name__ == "__main__":
    train_from_csv("data/features/train_reduced.csv")