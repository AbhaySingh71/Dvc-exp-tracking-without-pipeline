import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from dvclive import Live 

# Load data
df = pd.read_csv('./data/student_performance.csv')

X = df.iloc[:, :-1]
y = df['Placed']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Define hyperparameters
n_estimators = 100
max_depth = 10
min_samples_split = 4
min_samples_leaf = 2
max_features = 'sqrt'
bootstrap = True
random_state = 42

# Model
rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    bootstrap=bootstrap,
    random_state=random_state
)

# Train
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Log metrics and parameters
with Live(save_dvc_exp=True) as live:
    live.log_metric('Accuracy', accuracy_score(y_test, y_pred))
    live.log_metric('Precision', precision_score(y_test, y_pred))
    live.log_metric('Recall', recall_score(y_test, y_pred))
    live.log_metric('F1 score', f1_score(y_test, y_pred))

    live.log_param('n_estimators', n_estimators)
    live.log_param('max_depth', max_depth)
    live.log_param('min_samples_split', min_samples_split)
    live.log_param('min_samples_leaf', min_samples_leaf)
    live.log_param('max_features', max_features)
    live.log_param('bootstrap', bootstrap)



