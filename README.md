# Titanic-Survival-Prediction

## Description
This notebook demonstrates a complete machine learning workflow using the Titanic dataset.
It includes data cleaning, feature engineering, model training, evaluation, and visualization.  
The goal is to predict passenger survival and compare model performance using different machine learning algorithms.

## Steps
1. **Data Exploration:**
   - Checking data structure
   - Handling missing values
   - Basic visualization: bar charts, pie charts, histograms, pairplots

2. **Feature Engineering:**
   - Encoding categorical variables (`Sex`, `Title`, `Embarked`)
   - Creating new features: `FamilySize`, `IsAlone`, `Has_Cabin`

3. **Model Training:**
   - **Logistic Regression:** baseline model (with StandardScaler)
   - **Random Forest Classifier:** ensemble model to reduce variance
   - **Gradient Boosting Classifier:** ensemble model to improve accuracy

4. **Evaluation:**
   - Confusion Matrix
   - Classification Report (precision, recall, F1-score)
   - Accuracy Score
   - Feature Importance
   > Note: Precision, Recall, and F1-score are especially important due to class imbalance in the dataset.

5. **Optional Improvements (Level Up):**
   - Hyperparameter tuning (e.g., `n_estimators`, `max_depth`, `min_samples_leaf`, `learning_rate`, `subsample`)
   - Cross-validation
   - Advanced feature engineering
   - Additional visualizations

## Libraries Used
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`

## Usage
1. Load the dataset (`Titanic_Dataset.csv`)
2. Ensure Python 3.x and required libraries are installed
3. Run the notebook cells sequentially
4. Explore the outputs: metrics, plots, and model performance
