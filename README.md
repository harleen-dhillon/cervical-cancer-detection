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

 Future Work
* Implement **Deep Learning models (ANN / CNN)**
* Improve feature engineering
* Use larger datasets
* Optimize model performance

 Files
* `cervicalcancer.py`
* Dataset file
* Output graphs (confusion matrix, ROC, F1 comparison, heatmap)

