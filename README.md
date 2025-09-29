# Task 5: Decision Trees and Random Forests Classification 🌳

## 🎯 Objective
The goal of this task was to implement and evaluate **tree-based ensemble models** for classification, specifically focusing on the **Decision Tree Classifier** and the **Random Forest Classifier**. This exercise emphasizes understanding model complexity, preventing overfitting, and interpreting results through feature importance.

***

## 📊 Dataset
* **Dataset Used:** Heart Disease Dataset (e.g., UCI Cleveland).
* **Task:** Binary Classification—predicting the presence (1) or absence (0) of heart disease.
* **Features:** Includes clinical parameters such as Age, Sex, Chest Pain Type (CP), Resting Blood Pressure (Trestbps), Cholesterol (Chol), and Maximum Heart Rate Achieved (Thalach).

***

## 🛠️ Tools and Libraries
| Tool/Library | Purpose |
| :--- | :--- |
| **`scikit-learn`** | Core library for model building (`DecisionTreeClassifier`, `RandomForestClassifier`), data splitting, and evaluation metrics. |
| **`Pandas` & `NumPy`** | Data manipulation and numerical operations. |
| **`Matplotlib`** | Visualization of feature importances. |
| **`Graphviz`** | Used in conjunction with `sklearn.tree.export_graphviz` to visualize the Decision Tree structure. |

***

## ✨ Implementation Summary (Mini Guide)

1.  **Decision Tree (DT) Classifier:**
    * An initial, **unconstrained** DT was trained. This demonstrated how a deep tree can achieve near-perfect **training accuracy** but suffer from **overfitting** on the test data.
    * The tree structure was visualized to understand the feature-splitting logic (e.g., "If *thalach* $\le$ 145.5, go left").
2.  **Overfitting Control (Pruning):**
    * A second DT was trained with a constrained depth (`max_depth=4`) to demonstrate **pruning**. This typically leads to a slight decrease in training accuracy but an increase in **test accuracy**, showing improved generalization.
3.  **Random Forest (RF) Classifier:**
    * The RF model (an **ensemble** of 100+ trees using **bagging**) was trained. This model averages the predictions of individual, uncorrelated trees to significantly reduce the **variance** and minimize overfitting.
    * The RF model generally yielded the highest and most stable **test accuracy**.
4.  **Feature Importance:**
    * The `feature_importances_` attribute was extracted from the Random Forest model. This revealed the most influential features (e.g., CP, Thalach, Oldpeak) in predicting heart disease, enhancing model **interpretability**. 
5.  **Cross-Validation (CV):**
    * The Random Forest model was evaluated using **5-fold Cross-Validation** to ensure the model's performance was robust and not dependent on a single train-test split.

***

## 💡 Key Takeaways
* **Decision Trees** are highly interpretable but prone to **high variance/overfitting**.
* **Random Forests** overcome the limitations of a single tree by using **Bagging** (Bootstrap Aggregating) to build multiple trees and average their predictions, resulting in a more robust and accurate model.
* **Feature Importance** is a valuable output of ensemble tree models, providing clear, actionable insights into which input variables drive the prediction.
