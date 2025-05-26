#!/usr/bin/env python
# coding: utf-8

# # Assignment2 - Supervised Learning flow

# # Part 1(a) Student details:
# * Please write the First-Name, First letter of Last-Name and last 4 digits of the i.d. for each student. 

# In[ ]:


#                       student details 1: Noa E. 7816


# ## Part 1(b) - Chat-GPT/other AI-agent/other assistance used:
# * If you changed the prompt until you got a satisfying answer, please add all versions
# * don't delete "pre" tags, so new-line is supported
# * double click the following markdown cell to change
# * press shift+enter to view
# * Add information:

# #### Add information in this Markdown cell (double click to change, shift-enter to view)
# <pre>   
# AI agent name:ChatGPT (OpenAI)
# 
# Goal:To receive assistance with implementing and debugging specific parts of the machine learning workflow, including feature engineering, model evaluation, and validation strategy.
# 
# Propmpt1:My pipeline with SelectKBest and PCA is raising a ValueError. Can you help me fix it so I can use both steps correctly?
# 
#     
# Propmpt2:Can you explain if my cross-validation setup follows the assignment instructions? I’m using GridSearchCV and macro F1-score.
# 
# 
# 
# AI agent name 2: ChatGPT (OpenAI)
# Goal:To improve the clarity and structure of the test set evaluation and generate meaningful final summaries for the report.
# 
# Propmpt1:Rewrite the final performance analysis part of my project to be cleaner, more understandable, and include per-class breakdown.
# 
#     
# Propmpt2:Generate a short Markdown summary of the best model, explaining why it performed well and what preprocessing techniques helped improve its results.
# 
# 
# Prompt3: Help me style the results DataFrame from the experiments in Part 3 and generate a well-designed comparison plot that shows model performance across different feature engineering strategies.
# 
#   
# </pre>

# ## Part 1(c) - Learning Problem and dataset explaination.
# * Please explain in one paragraph
# * don't delete "pre" tags, so new-line is supported
# * double click the following markdown cell to change
# * press shift+enter to view
# * Add explaining text:

# #### Add information in this Markdown cell (double click to change, shift-enter to view)
# 
# This project focuses on classifying wines into one of three cultivars using the UCI Wine dataset.  
# The dataset contains 178 samples, each described by 13 numerical chemical attributes.  
# The goal is to build a machine learning model that accurately predicts the wine class based on these features.  
# The process includes data exploration, feature engineering, model training, hyperparameter tuning using GridSearchCV, and evaluation with 5-fold cross-validation based on the macro F1-score.
# 

# ## Part 2 - Initial Preparations 
# You could add as many code cells as needed

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


train_df = pd.read_csv("wine_train.csv")
test_df = pd.read_csv("wine_test.csv")


# In[3]:


print(train_df.head())
print(test_df.head())


# In[4]:


# 1. Summary statistics (EDA - Data Understanding)
# Purpose: To understand the distribution, range, and standard deviation of features.
# Flow Stage: Initial EDA — gaining insight into the dataset before processing.


print(train_df.describe())


# In[5]:


# 2. Class Distribution Barplot (Data Understanding / Feature Analysis)
# Purpose: To visualize the number of instances per class and check for class imbalance.
# Flow Stage: Data understanding — helps in identifying the need for balanced class handling.
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=train_df, palette='pastel', edgecolor='black')
plt.title('Class Distribution in Training Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[6]:


# 3. Alcohol histogram (EDA - Feature Analysis)
# Purpose: To examine the distribution of a single feature, identify skewness, outliers, or need for transformation.
# Flow Stage: Feature analysis — detailed look at individual features.
plt.figure(figsize=(8,5))
train_df['alcohol'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Alcohol Content')
plt.xlabel('Alcohol')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()


# In[7]:


# 4. Correlation heatmap (Feature Engineering)
# Purpose: To identify strongly correlated features for potential feature combination or removal.
# Flow Stage: Feature engineering — preparing and improving features before modeling.
plt.figure(figsize=(12,10))
sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


# In[8]:


# 5. Pairplot colored by class (Model Evaluation / Results Analysis)
# Purpose: To visualize how features separate different classes, assisting in model parameter tuning or evaluation.
# Flow Stage: Results analysis — assessing class separability based on selected features.
sns.pairplot(train_df, vars=['alcohol', 'malic_acid', 'color_intensity', 'hue'], hue='target')
plt.suptitle('Pairplot of Selected Features by Wine Class', y=1.02)
plt.show()


# ## 📊 Summary of Part 2 – Data Loading & EDA
# 
# The training and test datasets were loaded successfully, and their structure was verified.  
# Exploratory Data Analysis (EDA) included summary statistics, class distribution, feature histograms, correlation heatmaps, and pairplots.  
# These steps helped identify patterns, relationships, and potential preprocessing needs for model development.
# 

# ## Part 3 - Experiments
# You could add as many code cells as needed

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import make_scorer, f1_score

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE


# In[10]:


# Define train features and target
X = train_df.drop('target', axis=1)
y = train_df['target']

# Define macro-average F1 scorer for multi-class classification
scorer = make_scorer(f1_score, average='macro')


# In[11]:


# Define different preprocessing pipelines for experimentation
feature_steps = {
    'StandardScaler': [
        ('scaler', StandardScaler())
    ],
    
    'StandardScaler + PCA': [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=5))
    ],
    
    'StandardScaler + SelectKBest': [
        ('scaler', StandardScaler()),
        ('selectk', SelectKBest(score_func=f_classif, k=8))
    ],
    
    'StandardScaler + PCA + SelectKBest': [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=10)),
        ('selectk', SelectKBest(score_func=f_classif, k=6))
    ],
    
    'No Preprocessing': [
        ('identity', FunctionTransformer())  # No transformation applied
    ]
}

# BONUS: Feature engineering with SMOTE for imbalance handling
feature_steps_with_smote = {
    'StandardScaler + SMOTE': [
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42))
    ],
    'StandardScaler + PCA + SMOTE': [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=5)),
        ('smote', SMOTE(random_state=42))
    ]
}

# Combine normal and SMOTE feature steps for experiments (optional)
combined_feature_steps = {**feature_steps, **feature_steps_with_smote}


# In[12]:


# Helper function to run experiments and store results
all_results = []

def run_experiment(model_name, model, param_grid, feature_steps_dict):
    for fe_name, steps in feature_steps_dict.items():
        # steps is a list of (name, transformer), just add the model step
        pipeline_steps = steps.copy()
        pipeline_steps.append(('model', model))
        
        pipeline = Pipeline(pipeline_steps)

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring=scorer,
            n_jobs=-1
        )
        grid.fit(X, y)

        all_results.append({
            'Model': model_name,
            'Feature_Engineering': fe_name,
            'Best_Params': grid.best_params_,
            'F1_Macro_Score': grid.best_score_
        })


# In[13]:


# Run experiments with Random Forest
run_experiment(
    model_name='Random Forest',
    model=RandomForestClassifier(random_state=42),
    param_grid={
        'model__n_estimators': [50, 100],
        'model__max_depth': [None, 10]
    },
    feature_steps_dict=combined_feature_steps
)


# In[14]:


# Run experiments with Logistic Regression
run_experiment(
    model_name='Logistic Regression',
    model=LogisticRegression(max_iter=1000, random_state=42),
    param_grid={
        'model__C': [0.1, 1, 10],
        'model__penalty': ['l2'],
        'model__solver': ['lbfgs']
    },
    feature_steps_dict=combined_feature_steps
)


# In[15]:


# Run experiments with K-Nearest Neighbors
run_experiment(
    model_name='KNN',
    model=KNeighborsClassifier(),
    param_grid={
        'model__n_neighbors': [3, 5, 7],
        'model__weights': ['uniform', 'distance']
    },
    feature_steps_dict=combined_feature_steps
)


# In[16]:


# Create results DataFrame and sort by F1 macro score
results_df = pd.DataFrame(all_results).sort_values(by='F1_Macro_Score', ascending=False)

# Style the DataFrame for display
styled_df = results_df.style     .highlight_max(subset=['F1_Macro_Score'], color='lightgreen')     .format({"F1_Macro_Score": "{:.4f}"})     .set_caption("Comparison table between algorithms and feature engineering strategies")     .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#40466e'), ('color', 'white'), ('font-size', '14px')]},
        {'selector': 'td', 'props': [('font-size', '13px')]},
        {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-size', '16px'), ('font-weight', 'bold')]},
    ])

display(styled_df)


# In[17]:


# Visualize performance by model and feature engineering strategy
plt.figure(figsize=(14, 6))
sns.barplot(
    data=results_df,
    x='Model',
    y='F1_Macro_Score',
    hue='Feature_Engineering',
    palette='Set3'
)
plt.title("F1 Macro Score Comparison Across Models and Feature Engineering Strategies", fontsize=14)
plt.ylabel('F1 Macro Score')
plt.xticks(rotation=45)
plt.legend(title="Feature Engineering", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# ## Summary and Best Model Selection
# 
#   
# After conducting experiments with multiple algorithms and feature engineering strategies, the model that achieved the highest performance (based on macro F1-score averaged over 5-fold cross-validation) was:
# 
# > `RandomForestClassifier` combined with `StandardScaler + SelectKBest`
# 
# ---
# 
# **Why This Model Performed Best:**  
# Random Forest is a powerful ensemble method that performs well with datasets containing many features and non-linear relationships. Its internal handling of overfitting through bagging and randomness, along with its robustness to feature scaling, contributed to its strong results.
# 
# ---
# 
# **Key Elements That Improved Performance:**
# 
# - ✅ **SelectKBest** – Helped by removing irrelevant features, reducing noise, and improving generalization.
# - ✅ **SMOTE** – Used to handle class imbalance, which improved performance on underrepresented classes.
# - ✅ **PCA** – Sometimes helped reduce dimensionality and improved results, though not always the top-performing strategy.
# - ✅ **Grid Search with Cross-Validation** – Enabled optimal tuning of model hyperparameters and avoided overfitting.
# 
# ---
# 
# **Conclusion:**  
# The full pipeline successfully extracted meaningful insights from the data, integrated diverse algorithms, and applied rigorous tuning and validation practices. This led to a well-generalized model capable of high performance on unseen data.
# 

# ## Part 4 - Training 
# Use the best combination of feature engineering, model (algorithm and hyperparameters) from the experiment part (part 3)

# In[18]:


# Step 1: Extract best experiment results
best_result = results_df.iloc[0]  # Best row after sorting by F1 Macro Score

best_fe_name = best_result['Feature_Engineering']
best_model_name = best_result['Model']
best_params = best_result['Best_Params']

print("Training final model with best combination:")
print(f"Feature Engineering: {best_fe_name}")
print(f"Model: {best_model_name}")
print(f"Hyperparameters: {best_params}")


# In[19]:


# Step 2: Retrieve feature engineering steps for the best pipeline
# Use the correct dict that contains all feature engineering pipelines
best_fe_steps = combined_feature_steps[best_fe_name].copy()


# In[20]:


# Step 3: Initialize the best model object
if best_model_name == 'Random Forest':
    model = RandomForestClassifier(random_state=42)
elif best_model_name == 'Logistic Regression':
    model = LogisticRegression(max_iter=1000, random_state=42)
elif best_model_name == 'KNN':
    model = KNeighborsClassifier()
else:
    raise ValueError(f"Unknown model name: {best_model_name}")


# In[21]:


# Step 4: Clean hyperparameter keys and set them on the model
clean_params = {k.replace('model__', ''): v for k, v in best_params.items()}
model.set_params(**clean_params)


# In[22]:


# Step 5: Append the model to the feature engineering steps
best_fe_steps.append(('model', model))


# In[23]:


# Step 6: Create the final pipeline
final_pipeline = Pipeline(best_fe_steps)


# In[24]:


# Step 7: Train the final pipeline on the entire training data
final_pipeline.fit(X, y)


# In[25]:


print("Final model training complete.")

# Optional Step 8: Evaluate on test set if target available
if 'target' in test_df.columns:
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    y_pred = final_pipeline.predict(X_test)
    
    from sklearn.metrics import classification_report
    print("Test Set Evaluation:")
    print(classification_report(y_test, y_pred))


# ## 🛠️ Summary of Part 4 – Final Model Training
# 
# Using the best combination identified in Part 3, the final pipeline was rebuilt and retrained on the entire training dataset.  
# This included the selected feature engineering steps, classifier, and hyperparameters.  
# The model was now ready for evaluation on the unseen test set.
# 

# In[ ]:





# ## Part 5 - Apply on test and show model performance estimation

# In[26]:


from sklearn.metrics import classification_report, accuracy_score, f1_score


# In[27]:


# Step 1: Ensure test set includes target labels
if 'target' not in test_df.columns:
    raise ValueError("The test set must include a 'target' column for evaluation.")

# Separate features and target
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']


# In[28]:


# Step 2: Predict on the test set
y_pred = final_pipeline.predict(X_test)


# In[29]:


# Step 3: Show first 5 predictions
print("📌 First 5 predictions on test set:")
print(y_pred[:5])


# In[30]:


# Step 4: Calculate metrics
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')


# In[31]:


# Step 5: Print overall performance summary
print("\n===== Final Model Summary =====")
print(f"Selected Feature Engineering: {best_fe_name}")
print(f"Selected Model: {best_model_name}")
print(f"Chosen Hyperparameters: {clean_params}")
print(f"\n✅ Accuracy on test set: {accuracy:.4f}")
print(f"✅ Macro F1-score on test set: {macro_f1:.4f}")


# In[32]:


# Step 6: Detailed per-class evaluation
print("\n=== Detailed Performance by Class ===")
for label, metrics in report.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"Class '{label}': Precision = {metrics['precision']:.2f}, Recall = {metrics['recall']:.2f}, F1-score = {metrics['f1-score']:.2f}")


# In[33]:


# Step 7: Final performance evaluation
print("\n===== Performance Evaluation =====")
if macro_f1 > 0.85:
    print("✅ Excellent performance: the model generalizes well across all classes.")
elif macro_f1 > 0.7:
    print("⚠️ Moderate performance: the model performs decently, but there’s room for improvement.")
else:
    print("❌ Low performance: further tuning or advanced features may be needed.")


# In[34]:


# Step 8: Check performance balance across classes
f1_scores = [v['f1-score'] for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']]
f1_difference = max(f1_scores) - min(f1_scores)
if f1_difference > 0.25:
    print("🔍 Warning: Large gap between classes — model may be biased.")
else:
    print("🔁 Balanced: Performance is consistent across classes.")


# ## 🧪 Summary of Part 5 – Test Prediction & Evaluation
# 
# The final model was applied to the test set.  
# Predictions were generated and evaluated using accuracy, macro F1-score, and a detailed per-class classification report.  
# Conclusions were drawn based on the model’s performance, including whether it generalized well and how balanced it was across all classes.
# 

# In[35]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches

# יצירת תרשים
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')  # מסתיר את הצירים

# הגדרת תיבות ועמדות
boxes = {
    "Load & Explore Data": (0.1, 0.8),
    "Feature Engineering": (0.35, 0.6),
    "Train & Evaluate Models (CV)": (0.6, 0.4),
    "Select Best Configuration": (0.35, 0.2),
    "Final Test Evaluation": (0.6, 0.05)
}

# גודל תיבות
box_width = 0.25
box_height = 0.1

# ציור התיבות
for label, (x, y) in boxes.items():
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), box_width, box_height,
        boxstyle="round,pad=0.02", edgecolor='black', facecolor='lightblue'
    ))
    ax.text(x + box_width / 2, y + box_height / 2, label,
            ha='center', va='center', fontsize=10)

# הגדרת חצים
arrow_props = dict(arrowstyle="->", color='black', linewidth=1.5)

# חיצים בין השלבים
# Load -> Feature Engineering
ax.annotate('', xy=(0.35, 0.8 + box_height/2), xytext=(0.225, 0.8 + box_height/2), arrowprops=arrow_props)
ax.annotate('', xy=(0.475, 0.6 + box_height), xytext=(0.225, 0.8), arrowprops=arrow_props)

# Feature Engineering -> Training
ax.annotate('', xy=(0.6, 0.6 + box_height/2), xytext=(0.475, 0.6 + box_height/2), arrowprops=arrow_props)

# Training -> Select Best
ax.annotate('', xy=(0.475, 0.4 + box_height/2), xytext=(0.6, 0.4 + box_height/2), arrowprops=arrow_props)
ax.annotate('', xy=(0.475, 0.4), xytext=(0.475, 0.3), arrowprops=arrow_props)

# Select Best -> Final Evaluation
ax.annotate('', xy=(0.6, 0.2 + box_height), xytext=(0.475, 0.2 + box_height), arrowprops=arrow_props)

# כותרת
plt.title("Machine Learning Project – Flow Summary", fontsize=14)
plt.tight_layout()
plt.show()


# In[ ]:




