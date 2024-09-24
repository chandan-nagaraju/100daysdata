# 100daysdata

## Syllabus for 1st 10 days(24.09.224 - 04.09.2024) 

1. EDA, Covariance, Correlation, Pearson, Spearman Rank, Multi-dimensional, Feature Engineering, Column normalization, Standardization, Covariance matrix, Missing Values, Outlier treatment
    Vector space modeling, Cosine Similarity, Euclidean Distance
2. Linear Regression, Gradient Descent, Multicollinearity, VIF, R-square, Heteroscedasticity, Sklearn, Polynomial Regression, Bias-Variance trade-off, Regularization
 Logistic Regression, Squashing function, AUC. ROC, Precision-Recall Curve, Confusion matrix, Specificity, KNN, Naive Bayes, Support Vector Machine, Bayesian Machine Learning, Decision Trees, Ensemble learning, Bagging, Boosting

# Chatgpt derived schedule 
## Day-by-Day Schedule with Theory + Practice Focus

# Day 1-2: Exploratory Data Analysis (EDA) & Feature Engineering
 ## Theory:
    Understand types of correlations (Pearson, Spearman) and covariances.
    Learn the principles of multi-dimensional data and feature engineering techniques.
## Practical:
    Load datasets (e.g., Titanic, Iris).
    Perform univariate, bivariate analysis (using Pandas, Seaborn, Matplotlib).
    Create new features and analyze feature importance.
# Day 3: Normalization, Standardization, Missing Values, Outlier Treatment
  ## Theory:
    Learn why scaling (normalization vs standardization) is essential.
    Understand different missing value imputation techniques.
    Study various outlier detection and treatment methods.
  ## Practical:
    Normalize and standardize features using Sklearn.
    Handle missing values (mean, median, KNN imputation) in datasets.
    Use IQR, z-score to identify and treat outliers.
# Day 4: Distance Metrics & Vector Space Modeling
    ## Theory:
        Learn how cosine similarity and Euclidean distance work.
        Study vector space modeling (term-document matrix, TF-IDF).
    ## Practical:
        Apply vector space modeling on text data.
        Calculate similarity between text documents using cosine similarity and Euclidean distance in Python (sklearn, scipy).
# Day 5-6: Linear Regression, Gradient Descent, Regularization
    ## Theory:
        Study the mathematics behind Linear Regression, cost functions, and gradient descent.
        Understand multicollinearity, VIF, R-squared, and heteroscedasticity.
        Learn Lasso and Ridge regularization to avoid overfitting.
    ## Practical:
        Implement Linear Regression using sklearn.
        Visualize residuals, check assumptions of regression models (homoscedasticity).
        Apply regularization and visualize how coefficients shrink with Lasso and Ridge.
# Day 7: Logistic Regression, Performance Metrics
    ## Theory:
        Learn the workings of logistic regression, its decision boundary, and squashing functions.
        Understand evaluation metrics: AUC-ROC, F1 score, precision-recall, confusion matrix.
        Discuss bias-variance trade-off and its impact on model performance.
    ## Practical:
        Build logistic regression models and plot ROC curves.
        Practice using confusion matrices to calculate precision, recall, and F1 scores.
        Compare model performance using AUC-ROC and F1.
# Day 8: KNN, Naive Bayes, SVM, Decision Trees
    ## Theory:
        Understand the theory behind KNN, Naive Bayes, SVM, and Decision Trees.
        Learn how these algorithms work, their pros/cons, and appropriate use cases.
        Study tie-breaking in KNN and kernel functions in SVM.
    ## Practical:
        Implement KNN with sklearn, handle ties using weighted neighbors.
        Build Naive Bayes, Decision Tree models and compare performances.
        Train a Support Vector Machine on a classification task using different kernels.
# Day 9: Ensemble Learning, Bagging, Boosting
    ## Theory:
        Learn the difference between Bagging and Boosting.
        Study how GBDT works in multiclass classification.
    ## Practical:
        Implement Random Forest (Bagging) and GBDT for multiclass classification.
        Visualize how boosting reduces error in GBDT using Python (xgboost or lightgbm).
# Day 10: Theory Revision + Hands-on Projects
    ## Theory:
    Quickly review the core concepts: from EDA to advanced algorithms like GBDT.
    
    ## Practical:
        Work on a mini-project that combines classification, regression, and performance evaluation.
        Apply all learned techniques to clean data, create features, and build models.
        Evaluate the model using proper metrics (AUC-ROC, F1, etc.).

### Additional Tips:
1. Use Python notebooks: Write code alongside theory to consolidate learning.
2. Kaggle or UCI datasets: Apply concepts to datasets from Kaggle or UCI to solidify practical knowledge.
3. Document your learning: Maintain notes for each concept, explaining it in your own words with code snippets.
4. Test with questions: Ask yourself theoretical questions (e.g., when to use AUC over F1?) and solve coding problems after learning each concept.
