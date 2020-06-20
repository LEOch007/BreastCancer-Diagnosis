### Breast Cancer Diagnosis

---

A typical classification task to determine whether the cancer is benign or malignant.

The dataset is from UCI-ML: WDBC (Wisconsin Diagnostic Breast Cancer).

There are some features in dataset, i.e.:

- radius (mean of distances from center to points on the perimeter)
- texture (standard deviation of gray-scale values)
- perimeter
- area
- smoothness (local variation in radius lengths)
- compactness (perimeter^2 / area - 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry 
- fractal dimension ("coastline approximation" - 1)



The project includes: 

Data Exploration ( data preprocessing, feature selection, dimension reduction, etc...)

Model Construction ( sampling methods to solve imbalance, tree-based methods, etc...)



which can be divided into two stages: 

[ Stage 1]
File: BreastCancer-Diagnosis.ipynb 
Determine whether a new record data is related with benign breast cancer or malignant breast cancer using machine learning method.
The dataset is Wisconsin Diagnostic Breast Cancer from UCI Machine Learning Repository (UCI-MLR) can also be found in Kaggle

1. Data exploration
2. Data preprocessing
3. Feature engineering
4. Classifier Construction
5. Result Analysis



[ Stage 2 ]
Fold: FeatureExtraction&CellDetection:
We extract features that we use in Stage1 from breast diagnosis cells:

1. Detect cells from image using models
2. Calculate features following the original paper
   Details are stated in README.md of FeatureExtraction&CellDetection Fold.