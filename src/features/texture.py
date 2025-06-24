import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

def extract_texture_features(image_path: str):
    """
    Lê crop de feijão e retorna vetor com:
      - GLCM: contrast, homogeneity, energy, correlation
      - LBP: histograma uniform (bins=10)
      - HOG: feature_vector
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) GLCM (quantiza em 8 níveis)
    gray_q = (gray / 32).astype(np.uint8)
    glcm = graycomatrix(
        gray_q, distances=[1], angles=[0], levels=8,
        symmetric=True, normed=True
    )
    contrast     = graycoprops(glcm, 'contrast')[0,0]
    homogeneity  = graycoprops(glcm, 'homogeneity')[0,0]
    energy       = graycoprops(glcm, 'energy')[0,0]
    correlation  = graycoprops(glcm, 'correlation')[0,0]

    # 2) LBP (P=8, R=1, método uniform)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    # histograma uniform zones (0…9)
    (lbp_hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, 11),
                                 density=True)

    # 3) HOG
    hog_fd = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    # concatena tudo num vetor
    feats = [contrast, homogeneity, energy, correlation]
    feats += lbp_hist.tolist()
    feats += hog_fd.tolist()
    return feats


def build_texture_dataframe(processed_dir: str):
    """
    Percorre data/processed/{bons,ruins}, extrai texture features e retorna DataFrame
    """
    rows = []
    for cls, label in [('bons', 1), ('ruins', 0)]:
        folder = os.path.join(processed_dir, cls)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            feats = extract_texture_features(path)
            rows.append(feats + [label])

    # colunas
    cols = ['glcm_contrast','glcm_homogeneity','glcm_energy','glcm_correlation']
    cols += [f'lbp_{i}' for i in range(len(extract_texture_features(path))-4-len(hog(np.zeros((32,32)), orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)))]  # avoid dynamic, just demo
    # better: define exact number: LBP bins=10, HOG length = len(hog(gray…))
    # mas para simplificar:
    lbp_bins = 10
    hog_len = len(extract_texture_features(path)) - 4 - lbp_bins
    cols = ['glcm_contrast','glcm_homogeneity','glcm_energy','glcm_correlation'] \
           + [f'lbp_{i}' for i in range(lbp_bins)] \
           + [f'hog_{i}' for i in range(hog_len)] \
           + ['label']

    return pd.DataFrame(rows, columns=cols)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('processed_dir', help="data/processed")
    p.add_argument('out_csv', help="data/features/texture.csv")
    args = p.parse_args()

    df = build_texture_dataframe(args.processed_dir)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Texture CSV gerado em {args.out_csv} com {len(df)} linhas")
