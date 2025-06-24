import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def evaluate_model(test_csv: str,
                   model_path: str = 'data/models/rna.h5',
                   scaler_path: str = 'data/models/scaler.pkl',
                   output_dir: str = 'data/models/evaluation'):
    # 1. Carregar dados de teste
    df = pd.read_csv(test_csv)
    if 'label' not in df.columns:
        raise ValueError("CSV de teste deve ter coluna 'label'.")
    X = df.drop('label', axis=1).values
    y_true = df['label'].values

    # 2. Carregar scaler e transformar X
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler não encontrado em {scaler_path}.")
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    # 3. Carregar modelo Keras
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}.")
    model = load_model(model_path)

    # 4. Previsões
    y_prob = model.predict(X_scaled).ravel()  # valores entre 0 e 1
    # Definimos limiar 0.5 para classificar
    y_pred = (y_prob >= 0.5).astype(int)

    # 5. Métricas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # 6. Exibir no console
    print("=== Avaliação no conjunto de teste ===")
    print(f"Acurácia : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:\n", report)
    print("Matriz de Confusão:\n", cm)
    print(f"ROC AUC  : {roc_auc:.4f}")

    # 7. Salvar relatórios e curva ROC
    os.makedirs(output_dir, exist_ok=True)
    # 7a. Matriz de confusão como CSV
    cm_df = pd.DataFrame(cm, index=['true_0','true_1'], columns=['pred_0','pred_1'])
    cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'), index=True)
    # 7b. Classification report em txt
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    # 7c. Curva ROC em arquivo PNG
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0,1], [0,1], 'k--')  # diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    print(f"[OK] Resultados salvos em {output_dir}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--test_csv', default='data/features/test.csv',
                   help="Caminho para test.csv")
    p.add_argument('--model_path', default='data/models/rna.h5',
                   help="Caminho do modelo salvo")
    p.add_argument('--scaler_path', default='data/models/scaler.pkl',
                   help="Caminho do scaler salvo")
    p.add_argument('--output_dir', default='data/models/evaluation',
                   help="Pasta para salvar métricas e plots")
    args = p.parse_args()

    evaluate_model(
        test_csv=args.test_csv,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        output_dir=args.output_dir
    )
