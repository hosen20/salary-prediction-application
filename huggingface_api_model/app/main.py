# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os, pandas as pd

app = FastAPI(title="Minimal Salary Predictor")

MODEL_PATH = os.path.join("artifacts", "pipeline.joblib")
pipeline = joblib.load(MODEL_PATH)

class Job(BaseModel):
     experience_level: str,
     employment_type: str,
     job_title: str,
     company_location: str,
     company_size: str,
     remote_ratio: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(job: Job):
    """Predict the salary.
    
       Args:
           job: The information of a job.
           
       Returns:
           A dictionary of salary and model version.

       Raises:
           InternalServerError: If the dataframe object cannot be created.
    """
    try:
        row = pd.DataFrame([{
            "experience_level": job["experience_level"],
            "employment_type": job["employment_type"],
            "job_title": job["job_title"],
            "company_location": job["company_location"],
            "company_size": job["company_size"],
            "remote_ratio": job["remote_ratio"]
        }])
        pred = pipeline.predict(row)[0]
        return {"predicted_salary": float(pred), "model_version": "v1"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
