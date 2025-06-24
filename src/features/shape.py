import os
import cv2
import numpy as np
import pandas as pd

def extract_shape_features(image_path: str):
    """
    Lê uma imagem crop (RGB), gera máscara e retorna lista de shape features:
    [area, perimeter, aspect_ratio, circularity, extent, solidity, hu_1…hu_7]
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # main contour
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    x,y,w,h = cv2.boundingRect(cnt)
    aspect = float(w)/h
    circularity = 4*np.pi*area/(perim**2 + 1e-6)
    extent = area/(w*h + 1e-6)
    hull = cv2.convexHull(cnt)
    solidity = area/(cv2.contourArea(hull) + 1e-6)

    # Hu moments (log-transformed)
    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-6)

    return [area, perim, aspect, circularity, extent, solidity, *hu]


def build_shape_dataframe(processed_dir: str):
    """
    Percorre data/processed/{bons,ruins}, extrai shape features e retorna DataFrame:
    feat_1…feat_13 + label
    """
    rows = []
    for cls, label in [('bons', 1), ('ruins', 0)]:
        folder = os.path.join(processed_dir, cls)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            feats = extract_shape_features(path)
            if feats is None:
                print(f"[WARN] sem contorno em {path}")
                continue
            rows.append(feats + [label])

    # colunas
    cols = ['area','perimeter','aspect','circularity',
            'extent','solidity'] + [f'hu_{i+1}' for i in range(7)] + ['label']
    return pd.DataFrame(rows, columns=cols)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('processed_dir', help="data/processed")
    p.add_argument('out_csv', help="data/features/shape.csv")
    args = p.parse_args()

    df = build_shape_dataframe(args.processed_dir)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Shape CSV gerado em {args.out_csv} com {len(df)} linhas")
