import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

def select_top_k(train_csv, test_csv,
                 k=10,
                 out_train='data/features/train_reduced.csv',
                 out_test='data/features/test_reduced.csv'):
    # 1. Carrega train.csv
    df_train = pd.read_csv(train_csv)
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    # 2. Padroniza para o modelo de importância (opcional, mas ajuda se usar métodos sensíveis a escala)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # 3. Treina RandomForest para obter importância
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    imp = pd.Series(rf.feature_importances_, index=X_train.columns)
    imp_sorted = imp.sort_values(ascending=False)
    # 4. Seleciona top k
    topk = imp_sorted.head(k).index.tolist()
    print(f"Top {k} features:", topk)
    # 5. Salva CSVs reduzidos
    df_train_reduced = df_train[topk + ['label']]
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    df_train_reduced.to_csv(out_train, index=False)
    df_test = pd.read_csv(test_csv)
    df_test_reduced = df_test[topk + ['label']]
    df_test_reduced.to_csv(out_test, index=False)
    print(f"[OK] Salvos:\n  {out_train} ({df_train_reduced.shape})\n  {out_test} ({df_test_reduced.shape})")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', default='data/features/train.csv')
    p.add_argument('--test_csv',  default='data/features/test.csv')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--out_train', default='data/features/train_reduced.csv')
    p.add_argument('--out_test',  default='data/features/test_reduced.csv')
    args = p.parse_args()
    select_top_k(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        k=args.k,
        out_train=args.out_train,
        out_test=args.out_test
    )
