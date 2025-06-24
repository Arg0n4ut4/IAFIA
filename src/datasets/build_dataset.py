import os
import pandas as pd
from sklearn.model_selection import train_test_split

def build_full_dataset(shape_path, color_path, texture_path, out_train, out_test):
    # 1. Carrega os CSVs
    df_shape   = pd.read_csv(shape_path)
    df_color   = pd.read_csv(color_path)
    df_texture = pd.read_csv(texture_path)

    # 2. Garante que os arquivos tenham as mesmas ordens
    assert len(df_shape) == len(df_color) == len(df_texture), "[ERRO] Quantidades diferentes de linhas"

    # 3. Remove coluna 'label' dos intermediários (preservar apenas uma no final)
    y = df_shape["label"].copy()
    df_shape.drop("label", axis=1, inplace=True)
    df_color.drop("label", axis=1, inplace=True)
    df_texture.drop("label", axis=1, inplace=True)

    # 4. Concatena tudo
    X = pd.concat([df_shape, df_color, df_texture], axis=1)
    X["label"] = y

    # 5. Embaralha e divide
    df_train, df_test = train_test_split(
        X,
        test_size=0.3,
        stratify=y,
        shuffle=True,
        random_state=42
    )

    # 6. Salva arquivos
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    df_train.to_csv(out_train, index=False)
    df_test.to_csv(out_test, index=False)
    print(f"[OK] Dataset salvo:")
    print(f"  Treino: {out_train} ({len(df_train)} linhas)")
    print(f"  Teste : {out_test} ({len(df_test)} linhas)")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--shape', default='data/features/shape.csv')
    p.add_argument('--color', default='data/features/color.csv')
    p.add_argument('--texture', default='data/features/texture.csv')
    p.add_argument('--train_out', default='data/features/train.csv')
    p.add_argument('--test_out',  default='data/features/test.csv')
    args = p.parse_args()

    build_full_dataset(
        shape_path=args.shape,
        color_path=args.color,
        texture_path=args.texture,
        out_train=args.train_out,
        out_test=args.test_out
    )
