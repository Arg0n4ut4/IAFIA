# FeijõesAI

> **Classificação automatizada de feijões bons e defeituosos**
> Pipeline em Python + TensorFlow/Keras para segmentação, extração de atributos e rede neural densa.

---

## 📋 Participantes

- **João Pedro Rocha Senna**
- **Thiago Tanaka Peczek**

---

## 🤖 Nome da IA

**FeijõesAI - FIA** – Rede neural densa para classificação binária (`ruim` vs. `bom`).

---

## 🚀 Passo a passo de uso

1. **Clone o repositório**

   ```bash
   git clone https://github.com/Arg0n4ut4/IAFIA.git
   cd IAFIA
   ```

2. **Crie e ative um ambiente virtual**

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows
   ```

3. **Instale as dependências**

   ```bash
   pip install -r requirements.txt
   ```

4. **Pré-processamento**
   Segmenta e recorta cada grão de `data/raw/bons` e `data/raw/ruins` → `data/processed/`

   ```bash
   python src/preprocessing/segment.py data/raw data/processed --size 128
   ```

5. **Extração de features**

   - Forma:

     ```bash
     python src/features/shape.py data/processed data/features/shape.csv
     ```

   - Cor:

     ```bash
     python src/features/color.py data/processed data/features/color.csv
     ```

   - Textura:

     ```bash
     python src/features/texture.py data/processed data/features/texture.csv
     ```

6. **Construção do dataset**
   Junta shape, color e texture → divide 70% treino / 30% teste

   ```bash
   python src/datasets/build_dataset.py
   ```

   Gera:

   - `data/features/train.csv`
   - `data/features/test.csv`

7. **Treinamento da RNA**
   Treina com `train.csv` (70% dos bons + 70% dos ruins) e valida internamente (20% de validação).

   ```bash
   python src/train.py
   ```

   Modelo salvo em `data/models/rna.h5`.

8. **Avaliação**
   (Opcional) Rode `evaluate.py` para métricas finais no `test.csv`.

   ```bash
   python src/evaluate.py data/features/test.csv data/models/rna.h5
   ```

---

## 📊 Estrutura dos CSVs

### `train.csv` / `test.csv`

| Coluna                     | Descrição                                           |
| -------------------------- | --------------------------------------------------- |
| `area`                     | Área do contorno do grão                            |
| `perimeter`                | Perímetro do contorno                               |
| `aspect`                   | Razão largura/altura da bounding box                |
| `circularity`              | 4π·area / perímetro²                                |
| `extent`                   | Área ⁄ (largura·altura) da bounding box             |
| `solidity`                 | Área ⁄ (área do convex hull)                        |
| `hu_1` … `hu_7`            | 7 momentos de Hu invariantes (log-transformados)    |
| `prop_black`               | Proporção de pixels quase-pretos (RGB ≤ (30,30,30)) |
| `mean_b`,`mean_g`,`mean_r` | Média dos canais B, G, R                            |
| `std_b`,`std_g`,`std_r`    | Desvio-padrão dos canais B, G, R                    |
| `prop_dark_otsu`           | Proporção de pixels escuros no canal V (HSV + Otsu) |
| `glcm_contrast`            | Contraste da matriz GLCM                            |
| `glcm_homogeneity`         | Homogeneidade da matriz GLCM                        |
| `glcm_energy`              | Energia da matriz GLCM                              |
| `glcm_correlation`         | Correlação da matriz GLCM                           |
| `lbp_0` … `lbp_9`          | Histograma LBP (10 bins, método “uniform”)          |
| `hog_0` … `hog_N`          | Vetor HOG (orientações de gradiente)                |
| `label`                    | Classe: `0` = ruim, `1` = normal                    |

> **Observação:**
>
> - `train_reduced.csv` e `test_reduced.csv` contêm apenas as **10 features** mais importantes (calculadas via `RandomForestClassifier.feature_importances_`) + `label`.

---

## 🧪 Metodologias

1. **Pré-processamento**

   - Conversão para escala de cinza + Otsu
   - Operações morfológicas (open)
   - Detecção de contorno principal + crop + resize (128×128)

2. **Extração de atributos**

   - **Forma**: área, perímetro, razão, circularidade, extent, solidity, momentos de Hu
   - **Cor**: proporção de pixels escuros, estatísticas RGB, proporção escura em V (HSV+Otsu)
   - **Textura**: GLCM (contrast, homog., energy, corr.), LBP (10 bins), HOG

3. **Seleção de features**

   - Treinamento de `RandomForestClassifier(n_estimators=100)` em `train.csv`
   - Ordenação por importância e retenção das **10 top features**
   - Geração de `train_reduced.csv` e `test_reduced.csv`

4. **Rede Neural Densa**

   - Camadas fully-connected: 128 → 64 → 32
   - BatchNormalization + Dropout(0.3)
   - Saída Sigmoid (binary_crossentropy)
   - Métricas: `accuracy`, `AUC`
   - Callback: EarlyStopping (patience=10), ModelCheckpoint

---

## 📈 Resultados e Métricas

| Métrica                | Modelo Completo | Modelo Reduzido |
| ---------------------- | --------------- | --------------- |
| **Acurácia**           | 90,20 %         | 89,71 %         |
| **F1-score (médio)**   | 0,9091          | 0,9041          |
| **ROC-AUC**            | 0,9628          | 0,9511          |
| **Precision (ruim)**   | 0,88            | 0,87            |
| **Recall (ruim)**      | 0,91            | 0,91            |
| **Precision (normal)** | 0,93            | 0,93            |
| **Recall (normal)**    | 0,89            | 0,88            |

---

## 🔗 Repositório e Contato

- GitHub: [github.com/Arg0n4ut4/IAFIA](https://github.com/Arg0n4ut4/IAFIA)
