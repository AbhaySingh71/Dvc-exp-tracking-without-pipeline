import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, accuracy_score, recall_score , precision_score

from dvclive import Live 

df = pd.read_csv('./data/student_performance.csv')

X = df.iloc[:, :-1]
y = df['Placed']

X_train, X_test ,y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

n_estimators = 100
max_depth = 10

rf = RandomForestClassifier(n_estimators=n_estimators , max_depth=max_depth)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


with Live(save_dvc_exp=True) as live:
    live.log_metric('Accurcy', accuracy_score(y_test,y_pred))
    live.log_metric('Precision', precision_score(y_test,y_pred))
    live.log_metric('Recall', recall_score(y_test,y_pred))
    live.log_metric('F1 score', f1_score(y_test,y_pred))

    live.log_params('n_estimators' , n_estimators)
    live.log_params('max_depth' , max_depth)

    

