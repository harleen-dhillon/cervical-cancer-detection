# Cervical Cancer Detection 

This project predicts cervical cancer using machine learning models trained on clinical data.
It focuses on handling imbalanced data and comparing multiple models.

 Technologies Used
* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib, Seaborn

 Models Used
* Logistic Regression
* SVM
* Decision Tree
* Random Forest
* KNN
* Naive Bayes
* Gradient Boosting
* AdaBoost

 Results
* **Best Model:** Gradient Boosting
* **Best F1 Score:** 66.82%
* **Accuracy:** ~54.70%

The accuracy appears low because:

The dataset is imbalanced, with fewer cancer cases
The model prioritizes detecting cancer (Recall) over overall accuracy
Using SMOTE and class balancing shifts focus from majority-class accuracy to better minority detection
Medical datasets are complex, and patterns are not easily captured by basic ML models.

 Files
* `cervicalcancer.py`
* Dataset file
* Output graphs (confusion matrix, ROC, F1 comparison, heatmap)

## 2. Deep Learning — Cervical Cell Classification

This component explores deep learning-based classification of cervical cell images using the SIPaKMeD dataset. It focuses on image preprocessing, data augmentation, feature extraction, model evaluation, and visual interpretation of predictions.

### Dataset

The project uses the SIPaKMeD dataset, containing five cervical cell categories:

- Dyskeratotic
- Koilocytotic
- Metaplastic
- Parabasal
- Superficial-Intermediate

The dataset is not stored in this repository. It must be downloaded separately and the local dataset path configured before running the notebooks.

### Methodology

**1. Dataset Auditing and Splitting**
- Inspect image counts, class distribution, and image readability.
- Use group-aware train, validation, and test splits to reduce the risk of related images appearing across multiple subsets.

**2. ResNet18 Classification**
- Train a ResNet18-based image classifier.
- Compare strong and mild data augmentation strategies.
- Track training loss, validation loss, and Macro-F1 across epochs.

**3. Deep Feature Classification and Optimization**
- Explore SVM classification using features extracted by ResNet18.
- Evaluate tuned SVM and PSO-SVM approaches using saved validation and test comparisons.

**4. Model Interpretability**
- Generate Grad-CAM visualizations to examine image regions influencing predictions.
- Inspect examples of correctly classified and misclassified images.

### Results and Visualizations

The repository includes the following experimental artifacts:

- Training and validation loss curves
- Training and validation Macro-F1 curves
- Strong versus mild augmentation comparison
- Validation and test confusion matrices
- Validation/test comparison CSV files
- Grad-CAM visualizations for correct and incorrect predictions

The saved comparison results report approximately **96.2% test accuracy** for the ResNet18 + PSO-SVM experiment. These are recorded experimental results, not independently verified clinical performance. The test set was previously inspected during development, so the result should not be interpreted as an unbiased estimate of generalization.

### Notebooks and Files

- [`Dataset Audit`](DL/notebooks/01_dataset_audit.ipynb)
- [`Augmentation and Baseline`](DL/notebooks/02_augmentation_and_baseline.ipynb)
- [`DL Results`](DL/results/)
- [`Detailed DL Documentation`](DL/README.md)
