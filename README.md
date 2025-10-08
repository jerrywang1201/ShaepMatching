# ShapeMatching
# TrashNet — Traditional Shape Classifier (Hu + Geometry)

> minimal, reproducible, CPU-only pipeline

---

## Quickstart
```bash
# 0) python 3.10+ recommended
python3 -m venv trashnet_env
source trashnet_env/bin/activate

# 1) deps
pip install --upgrade pip
pip install datasets huggingface_hub pillow opencv-python numpy pandas scikit-learn

# 2) run experiment (5-fold kNN; saves results.txt)
python trashnet_hu_otsu.py --k 5

python trashnet_hu_HSV_Otsu.py --k 5 --fd_k 20


What this repo does (in code terms)
load HF dataset → garythung/trashnet
preprocess → grayscale/HSV + Otsu + morph close + largest CC
features → Hu(7) + geometry [ratio, circularity, solidity, eccentricity, norm_perimeter]
scaler → StandardScaler()
classifier → KNN(k, weights='distance') with {euclidean, mahalanobis}
eval → 5-fold CV → accuracy / classification_report / confusion matrix
output → prints to console and writes results.txt

Script
trashnet_hu_plus_save.py
entrypoint: python trashnet_hu_plus_save.py --k 5
args:
--k (int, default=5) # k for kNN
side-effects: creates results.txt

Typical Output
========== FINAL (EUCLIDEAN) ==========
overall acc: ~0.39
...
========== FINAL (MAHALANOBIS) ==========
overall acc: ~0.40
...
💾 Results saved to results.txt

Repo Tree
.
├── trashnet_hu_plus_save.py   # main pipeline
├── results.txt                # saved report (created after run)
└── README.md                  # this file
