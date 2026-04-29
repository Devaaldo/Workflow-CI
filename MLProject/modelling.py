import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

TRAIN_PATH = 'diabetesdataset_preprocessing/diabetes_train.csv'
TEST_PATH  = 'diabetesdataset_preprocessing/diabetes_test.csv'
TARGET_COL = 'Outcome'

mlflow.set_experiment('diabetes-classification')

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train = train_df.drop(TARGET_COL, axis=1)
y_train = train_df[TARGET_COL]
X_test  = test_df.drop(TARGET_COL, axis=1)
y_test  = test_df[TARGET_COL]

with mlflow.start_run(run_name='RandomForest-CI'):
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
