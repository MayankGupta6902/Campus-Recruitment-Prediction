# Campus-Recruitment-Prediction
It is a Machine Learning model (gradient boosting) which will predict the recuitments on the basis of dataset provided

# Objective of the project

**Problem Statement:**

The Placement of students is one of the most important objective of an educational
institution. Reputation and yearly admissions of an institution invariably depend on the
placements it provides it students with. That is why all the institutions, arduously, strive
to strengthen their placement department so as to improve their institution on a whole.
Any assistance in this particular area will have a positive impact on an institution’s ability
to place its students. This will always be helpful to both the students, as well as the
institution.

The main goal is to predict whether the student will be recruited in campus placements
or not based on the available factors in the dataset.

# Important terms to know beforehand:

**Libraries:**

1. **NumPy**: NumPy is a fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.

2. **Pandas**: Pandas is a powerful data manipulation library built on top of NumPy, offering data structures like DataFrame and Series that simplify working with structured data and time series data, along with tools for reading and writing data between in-memory data structures and various file formats.

3. **Matplotlib**: Matplotlib is a versatile plotting library for creating static, animated, and interactive visualizations in Python. It offers a wide range of plotting functions to generate plots, histograms, power spectra, bar charts, error charts, and scatterplots, among others, with full control over every aspect of the figure.

4. **Seaborn**: Seaborn is a statistical data visualization library based on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics. It complements Matplotlib and Pandas by providing additional plotting functions and themes to enhance the visual appeal of plots.

5. **Scikit-learn**: Scikit-learn is a machine learning library in Python that provides simple and efficient tools for data mining and data analysis. It features various algorithms for classification, regression, clustering, dimensionality reduction, and model selection, along with tools for evaluating model performance and data preprocessing.

6. **sklearn.model_selection**: The `model_selection` module in scikit-learn provides functions for splitting datasets into train and test sets, cross-validation, and parameter tuning, enabling users to evaluate the performance of machine learning models and select the best parameters for their models.

7. **train_test_split**: `train_test_split` is a function in scikit-learn's `model_selection` module used to split datasets into training and testing sets, allowing users to train a model on one subset of the data and evaluate it on another subset to assess its performance.

8. **sklearn.ensemble**: The `ensemble` module in scikit-learn provides a set of ensemble learning methods, such as Random Forest and Gradient Boosting, which combine multiple base estimators to improve the overall performance of the model.

9. **GradientBoostingClassifier**: `GradientBoostingClassifier` is an implementation of gradient boosting for classification in scikit-learn, which builds an ensemble of decision trees sequentially, each correcting the errors of its predecessor, to improve the model's predictive accuracy.

10. **sklearn.metrics**: The `metrics` module in scikit-learn provides functions for evaluating the performance of machine learning models, including metrics such as accuracy, precision, recall, F1-score, ROC AUC, and mean squared error, among others.

11. **accuracy_score**: `accuracy_score` is a function in scikit-learn's `metrics` module used to calculate the accuracy of a classification model, which is the proportion of correct predictions to the total number of predictions made.

12. **pickle**: Pickle is a Python module used for serializing and deserializing Python objects. It allows objects to be saved to a file and later restored, enabling users to save the state of their program or share data between Python processes.

13. **random**: The random module in Python provides functions for generating random numbers and selecting random elements from a sequence. It is commonly used for tasks like shuffling data, generating random samples, and implementing random algorithms.


**For Visualization:**

**Histogram**: A histogram is a graphical representation of the distribution of numerical data. It consists of bars that show the frequency of data within equal intervals or bins. Histograms are useful for visualizing the shape, central tendency, and spread of data, making it easy to identify patterns, outliers, and underlying distributions.

**Heatmap**: A heatmap is a graphical representation of data where the individual values contained in a matrix are represented as colors. It is useful for visualizing the magnitude of values in a matrix and is often used to plot correlations, distributions, or densities of data. Heatmaps are particularly effective for highlighting patterns and trends in large datasets.


# About gradient boosting model and it's advantages

Gradient boosting is a powerful ensemble learning technique used for regression and classification tasks. It builds a strong predictive model by combining the predictions of multiple individual models, typically decision trees, in a sequential manner. The key idea behind gradient boosting is to fit a new model to the residual errors (the difference between the actual and predicted values) of the existing model, thereby reducing the errors at each step. This process is repeated iteratively, with each new model focusing on the mistakes of its predecessors, gradually improving the overall model's performance.


**Algorithm**:

1. **Initialize the model**: The algorithm starts by initializing the model with a constant value, usually the mean of the target variable for regression tasks, or a class with the highest frequency for classification tasks.

2. **Compute the residuals**: For each iteration, the algorithm computes the negative gradient of the loss function with respect to the current model's predictions. This gives the residual errors, which are the differences between the actual and predicted values.

3. **Fit a base learner**: A base learner, often a decision tree, is fitted to the residuals. The tree is typically shallow to avoid overfitting and is trained to predict the residuals of the previous model.

4. **Update the model**: The predictions of the new model are added to the previous model's predictions, gradually improving the overall model's performance.

5. **Repeat**: Steps 2-4 are repeated until a predefined number of iterations is reached, or until the model's performance no longer improves.

6. **Final model**: The final model is the sum of all the base learners' predictions, weighted by a learning rate, which controls the contribution of each model to the final prediction.


**Advantages**:

1. **High Predictive Accuracy**: Gradient boosting often achieves higher predictive accuracy compared to other machine learning algorithms due to its ability to capture complex relationships in the data.

2. **Handles Mixed Data Types**: Gradient boosting can handle a mixture of feature types (e.g., numerical, categorical) and automatically handles missing data, reducing the need for extensive data preprocessing.

3. **Feature Importance**: The algorithm provides a measure of feature importance, which can help identify the most relevant features for prediction and improve interpretability.

4. **Robust to Overfitting**: Gradient boosting is less prone to overfitting compared to other ensemble methods like bagging, especially when using shallow trees and regularization techniques.

5. **Flexibility**: It can be applied to various types of problems, including regression, classification, and ranking tasks, making it a versatile algorithm for different use cases.


