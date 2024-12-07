from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from .schemas import LoanApplication, LoanResponse
import pandas as pd

app = FastAPI(title="Loan Approval API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = joblib.load('app/models/loan_type_models.joblib')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Loan Approval API",
            "documentation": "/docs",
            "health": "OK"}

@app.post("/predict", response_model=LoanResponse)
async def predict_loan(application: LoanApplication):
    try:
        # Convert application to correct format
        loan_type = application.loan_type.lower()
        if loan_type not in ['student', 'business', 'agricultural']:
            raise HTTPException(
                status_code=400, 
                detail="Loan type must be 'student', 'business', or 'agricultural'"
            )
            
        # Prepare application data
        application_data = application.dict(exclude={'loan_type'})
        
        # Make prediction
        result = model.predict_loan_approval(application_data, loan_type)
        
        return LoanResponse(
            approved=result['approved'],
            confidence=result['confidence'],
            feedback=result['feedback']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}