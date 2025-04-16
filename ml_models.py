import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from feature_engineering import distances as features
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log the start of the process
logging.info('Starting ML model training and evaluation')

# Generate synthetic labels for demonstration purposes
def generate_labels(num_samples):
    return np.random.randint(0, 2, num_samples)

# Use the pre-calculated distances as features
labels = generate_labels(len(features))

# Reshape features to be 2D
features = np.array(features).reshape(-1, 1)

# Log the reshaped features
logging.info(f'Reshaped features to 2D array with shape: {features.shape}')

# Log the number of features and labels
logging.info(f'Number of features: {len(features)}')
logging.info(f'Number of labels: {len(labels)}')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Log the data split
logging.info('Data split into training and testing sets')

# Train and evaluate a Random Forest model
logging.info('Training Random Forest model')
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
logging.info(f'Random Forest Accuracy: {rf_accuracy:.2f}')

# Train and evaluate a Support Vector Machine model
logging.info('Training SVM model')
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
logging.info(f'SVM Accuracy: {svm_accuracy:.2f}')

# Log the end of the process
logging.info('ML model training and evaluation completed') 