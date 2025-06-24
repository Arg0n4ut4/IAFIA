import os
import cv2
import numpy as np
import pandas as pd

def extract_color_features(image_path: str, thresh_rgb: tuple=(30,30,30)):
    """
    Lê uma imagem crop (RGB) e retorna:
      - prop_black: proporção de pixels RGB ≤ thresh_rgb
      - mean_rgb: média dos canais R, G, B
      - std_rgb: desvio-padrão dos canais R, G, B
      - prop_dark_otsu: proporção de pixels considerados “escuros” no canal V (HSV + Otsu)
    """
    img = cv2.imread(image_path)
    # proporção de pixels “quase pretos” em RGB
    lower = np.array([0,0,0], dtype=np.uint8)
    upper = np.array(thresh_rgb, dtype=np.uint8)
    mask_black = cv2.inRange(img, lower, upper)
    prop_black = cv2.countNonZero(mask_black) / (img.shape[0]*img.shape[1])

    # estatísticas de cores
    chans = cv2.split(img)
    mean_rgb = [float(np.mean(c)) for c in chans]
    std_rgb  = [float(np.std(c))  for c in chans]

    # HSV + Otsu no canal V
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    _, mask_dark = cv2.threshold(
        v, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    prop_dark_otsu = cv2.countNonZero(mask_dark) / (img.shape[0]*img.shape[1])

    return [prop_black, *mean_rgb, *std_rgb, prop_dark_otsu]


def build_color_dataframe(processed_dir: str):
    """
    Percorre data/processed/{bons,ruins}, extrai color features e retorna DataFrame:
    feat_1…feat_8 + label
    """
    rows = []
    for cls, label in [('bons', 1), ('ruins', 0)]:
        folder = os.path.join(processed_dir, cls)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            feats = extract_color_features(path)
            rows.append(feats + [label])

    cols = [
      'prop_black',
      'mean_b', 'mean_g', 'mean_r',
      'std_b',  'std_g',  'std_r',
      'prop_dark_otsu',
      'label'
    ]
    return pd.DataFrame(rows, columns=cols)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('processed_dir', help="data/processed")
    p.add_argument('out_csv', help="data/features/color.csv")
    args = p.parse_args()

    df = build_color_dataframe(args.processed_dir)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Color CSV gerado em {args.out_csv} com {len(df)} linhas")
