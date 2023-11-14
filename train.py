import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    df = pd.read_csv("train.csv")
    df.drop('sl_no',axis=1,inplace=True)
    df['salary'] = df['salary'].fillna(0)
    df['ssc_b_Central'] = df['ssc_b'].map({'Central':1,'Others':0})
    df['hsc_b_Central'] = df['hsc_b'].map({'Central':1,'Others':0})
    df['workex'] = df['workex'].map({'No':0,'Yes':1})
    df['status'] = df['status'].map({'Placed':1,'Not Placed':0})
    df['specialisation_fin'] = df['specialisation'].map({'Mkt&HR':0,'Mkt&Fin':1})
    df.drop(['ssc_b','hsc_b','specialisation'],axis=1,inplace=True)
    ohe = pd.get_dummies(df[['hsc_s','degree_t']],drop_first=True).astype(int)
    df1 = pd.concat([ohe,df.drop(['hsc_s','degree_t'],axis=1)],axis=1)

    X = df1.drop('salary',axis=1)
    y = df1['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train_scaled, y_train)

        y_pred = lr.predict(X_test_scaled)

        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = lr.predict(X_train_scaled)
        signature = infer_signature(X_train_scaled, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetCampusModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)