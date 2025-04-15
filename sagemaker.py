import joblib
joblib.dump(best_model, 'isolation_forest_tuned.joblib')

import boto3
s3 = boto3.client('s3')
bucket_name = 'your-sagemaker-bucket'
s3.upload_file('isolation_forest_tuned.joblib', bucket_name, 'isolation_forest_tuned.joblib')

import joblib
import numpy as np

from preprocessing import load_and_preprocess

# Load preprocessed data
file_path = 'creditcard.csv'
X_train_sm, y_train_sm, X_test, y_test = load_and_preprocess(file_path)

def model_fn(model_dir):
    model = joblib.load(f"{model_dir}/isolation_forest_tuned.joblib")
    return model

def input_fn(request_body, content_type):
    if content_type == "application/json":
        import json
        return np.array(json.loads(request_body))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return [0 if x == 1 else 1 for x in prediction]

import sagemaker
from sagemaker.sklearn import SKLearnModel

sagemaker_session = sagemaker.Session()
role = "your-aws-iam-role"

# Define model
model = SKLearnModel(
    model_data=f"s3://{bucket_name}/isolation_forest_tuned.joblib",
    role=role,
    entry_point="inference.py"
)

# Deploy
predictor = model.deploy(instance_type="ml.m5.large", initial_instance_count=1)

# Test prediction
result = predictor.predict(X_test[:5].tolist())
print(result)
