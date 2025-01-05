# BAN6420_M5_1
# BAN6420: Programming in R & Python
# Milestone Assignment 2: Principal Component Analysis

# Name: Babalola Taiwo
# Learner IS: 162894
# Breast Cancer Classification Using PCA and Logistic Regression

1. Introduction
	This project demonstrates how to perform dimensionality reduction using Principal Component Analysis (PCA) and classify breast cancer data using Logistic Regression.
	The dataset used is the `sklearn.datasets.load_breast_cancer`, which contains 	features of malignant and benign tumors 
		1. Analyze the breast cancer dataset.
		2. Perform dimensionality reduction using PCA.
		3. Optimize logistic regression with hyperparameter tuning.
		4. Compare classification performance with and without PCA.
		5. Visualize results including PCA components, ROC curve, and confusion matrix.

2. Requirements
	- Python 3.7+
	- Libraries:
 		- numpy
  		- pandas
  		- matplotlib
  		- seaborn
  		- scikit-learn

	Install the required libraries using:
		```bash
	pip install numpy pandas matplotlib seaborn scikit-learn
		```

3. Code Overview

	3.1 Load and Explore the Dataset
	- Load the `breast_cancer` dataset using `sklearn.datasets`.
	- Analyze basic statistics and visualize feature correlations.

	3.2 Data Preprocessing
	- Standardize features using `StandardScaler`.
	- Visualize pairwise feature correlations with a heatmap.

	3.3 Principal Component Analysis (PCA)
	- Use PCA to reduce dimensions while retaining 95% variance.
	- Visualize cumulative explained variance.
	- Scatter plot the first two PCA components to show class separation.

	3.4 Logistic Regression
	- Perform hyperparameter tuning using `GridSearchCV`.
	- Train the optimized logistic regression model on PCA-reduced data.
	- Evaluate model performance with metrics: accuracy, precision, recall, F1-score, and ROC AUC.
	- Compare results with models trained on original features.

	3.5 Visualizations
	- Explained variance plot for PCA.
	- Feature contributions to principal components.
	- Confusion matrix.
	- ROC curve.

4. Results
	Key results obtained:
	- Optimal PCA Components: Determined dynamically based on 95% variance threshold.
	- Accuracy on PCA-Reduced Features: ~97%
	- Accuracy on Original Features: ~98%
	- Cross-Validation Precision, Recall, and F1 Scores: High scores demonstrating robust model performance.
	- ROC AUC: Area under the ROC curve ~0.99.

5. How to Run
	1. Clone or download the script via: https://github.com/AmazingTaiwo/BAN6420_M5_1.git
	2. Run the script in a Python environment on CMD via below command 
   		python main.py
  	3. Review outputs and visualizations in the terminal and pop-up generated plots.

6. Visualizations
    - 6.1 PCA Components Scatter Plot
    	- Illustrates the separation between malignant and benign classes using the first two principal components.
    - 6.2 Cumulative Explained Variance
    	- Helps identify the optimal number of PCA components.
    - 6.3 Confusion Matrix
    	- A heatmap showing true vs. predicted classifications.
    - 6.4 ROC Curve
    	- Visualizes the model’s performance across various classification thresholds.

