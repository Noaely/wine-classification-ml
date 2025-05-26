# 🍷 Wine Classification – Machine Learning Project

This project was completed as part of an academic Machine Learning course.  
The objective is to classify wine types from the UCI Wine dataset using various supervised learning techniques.

##  Dataset
The dataset includes 178 samples with 13 numerical chemical features.  
Each sample belongs to one of three wine cultivars.

##  Project Workflow

1. **Data Loading & EDA**  
   Loaded the train/test datasets and explored feature distributions, class balance, and correlations.

2. **Feature Engineering**  
   Applied preprocessing techniques including StandardScaler, PCA, SelectKBest, and SMOTE for class imbalance.

3. **Model Training & Experiments**  
   Trained models using GridSearchCV with 5-fold cross-validation.  
   Models used: Random Forest, Logistic Regression, KNN.  
   Evaluated with **macro F1-score**.

4. **Best Model Selection**  
   Selected the top-performing combination of feature pipeline, model, and hyperparameters based on validation scores.

5. **Final Training & Test Evaluation**  
   Retrained the selected model on the full training set and evaluated it on the test set.

##  Results
- Best model: `RandomForestClassifier` with SelectKBest
- Macro F1-score used to ensure balanced performance across all classes
- Final performance reported using accuracy, F1, precision, recall

## 🛠️ Technologies
- Python, scikit-learn, pandas, matplotlib, seaborn, imblearn
- Jupyter Notebook

##  Files
- `Wine_Classification.ipynb` – full notebook with code and analysis
- `/charts` – visualizations generated during EDA and evaluation
- `README.md` – project documentation

##  Author
Noa Elyashar
