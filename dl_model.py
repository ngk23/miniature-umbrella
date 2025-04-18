import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from feature_engineering import distances as features


def generate_labels(num_samples):
    return np.random.randint(0, 2, num_samples)
labels = generate_labels(len(features))
features = np.array(features).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

def build_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

model = build_model()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1, batch_size=128, validation_split=0.4)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Deep Learning Model Accuracy: {accuracy:.2f}') 