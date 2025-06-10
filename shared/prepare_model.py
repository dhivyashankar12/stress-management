# shared/prepare_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import os

# Load and preprocess data
df = pd.read_csv('SaYoPillow.csv')
X = df.drop('sl', axis=1)
y = df['sl']
X = minmax_scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create directory to save models
os.makedirs('../server/model', exist_ok=True)

### 1. Neural Network ###
model = Sequential([
    Dense(64, input_shape=(8,), activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', restore_best_weights=True)
model.fit(X_train, y_train, epochs=25, batch_size=16, verbose=0)
nn_loss, nn_acc = model.evaluate(X_test, y_test, verbose=0)
model.save('../server/model/model.h5')
np.save('../server/model/model_weights.npy', np.array(model.get_weights(), dtype=object), allow_pickle=True)

### 2. Logistic Regression ###
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
logreg_acc = accuracy_score(y_test, logreg.predict(X_test))
joblib.dump(logreg, '../server/model/logreg_model.pkl')

### 3. Random Forest ###
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
joblib.dump(rf, '../server/model/rf_model.pkl')

### 4. Support Vector Machine ###
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test))
joblib.dump(svm, '../server/model/svm_model.pkl')

### 5. XGBoost ###
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
joblib.dump(xgb, '../server/model/xgb_model.pkl')

# Save performance
with open('../server/model/centralized_performance.txt', 'w') as f:
    f.write(f"Neural Network Accuracy: {nn_acc:.4f}\n")
    f.write(f"Logistic Regression Accuracy: {logreg_acc:.4f}\n")
    f.write(f"Random Forest Accuracy: {rf_acc:.4f}\n")
    f.write(f"SVM Accuracy: {svm_acc:.4f}\n")
    f.write(f"XGBoost Accuracy: {xgb_acc:.4f}\n")
