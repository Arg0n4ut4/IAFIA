import os
import cv2
import argparse

def segment_and_crop(input_dir: str, output_dir: str, size: int = 128):
    # Garante as pastas de saída
    for cls in ['bons', 'ruins']:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    # Para cada imagem em bons/ruins
    for cls in ['bons', 'ruins']:
        src_folder = os.path.join(input_dir, cls)
        dst_folder = os.path.join(output_dir, cls)

        for filename in os.listdir(src_folder):
            if not filename.lower().endswith(('.jpg', '.png')):
                continue

            img_path = os.path.join(src_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] {img_path} não pôde ser lido")
                continue

            # 1) Cinza + Otsu
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # 2) Limpeza (morph open)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 3) Contorno principal
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                print(f"[WARN] Sem contorno em {img_path}")
                continue
            cnt = max(contours, key=cv2.contourArea)

            # 4) Crop pelo bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            crop = img[y:y+h, x:x+w]

            # 5) Redimensiona
            crop_resized = cv2.resize(crop, (size, size),
                                      interpolation=cv2.INTER_AREA)

            # 6) Salva
            out_path = os.path.join(dst_folder, filename)
            cv2.imwrite(out_path, crop_resized)
            print(f"[OK] {filename} → {os.path.relpath(out_path)}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Segmenta e recorta feijões (bons/ruins)"
    )
    p.add_argument('input_dir', help="data/raw")
    p.add_argument('output_dir', help="data/processed")
    p.add_argument('--size', type=int, default=128,
                   help="tamanho do crop (padrão 128)")
    args = p.parse_args()
    segment_and_crop(args.input_dir, args.output_dir, args.size)
