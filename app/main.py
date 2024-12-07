from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from .schemas import LoanApplication, LoanResponse
import pandas as pd
import numpy as np
from typing import List, Dict

app = FastAPI(title="Loan Approval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    model_data = joblib.load('app/models/loan_type_models.joblib')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_data = {}

class LoanPredictor:
    def __init__(self, model_data):
        self.models = model_data
        
    def create_interaction_features(self, X: pd.DataFrame, loan_type: str) -> pd.DataFrame:
        if loan_type == 'student':
            X['score_income_ratio'] = X['credit_score'] * X['annual_income']
            X['age_education_factor'] = X['person_age'] * X['education']
            X['debt_income_ratio'] = X['debt_to_income'] / (X['income_to_loan'] + 1)
            X['credit_term_factor'] = X['credit_score'] * X['term']
        elif loan_type == 'agricultural':
            X['credit_mortgage_ratio'] = X['credit_score'] * X['mortgage']
            X['income_emp_factor'] = X['annual_income'] * X['emp_length']
            X['debt_asset_ratio'] = X['debt_to_income'] / (X['person_home_ownership'] + 1)
            X['loan_term_factor'] = X['loan_amount'] * X['term']
        elif loan_type == 'business':
            X['credit_card_income'] = X['credit_card_usage'] * X['annual_income']
            X['business_exp_factor'] = X['emp_length'] * X['annual_income']
            X['debt_credit_ratio'] = X['debt_to_income'] / (X['credit_score'] + 1)
            X['card_util_ratio'] = X['credit_card'] * X['credit_card_usage']
        return X
    
    def predict_loan_approval(self, data: Dict, loan_type: str) -> Dict:
        try:
            if loan_type not in self.models:
                raise ValueError(f"No model found for loan type: {loan_type}")
            
            # Create DataFrame
            X = pd.DataFrame([data])
            
            # Create interaction features
            X = self.create_interaction_features(X, loan_type)
            
            # Get model and features
            model_info = self.models[loan_type]
            model = model_info['model']
            features = model_info['features']
            
            # Select required features
            X = X[features]
            
            # Get prediction probability
            prob_approval = model.predict_proba(X)[0][1]
            confidence = round(prob_approval * 100, 2)
            
            # Thresholds
            thresholds = {
                'student': 0.65,
                'agricultural': 0.70,
                'business': 0.75
            }
            
            approved = prob_approval >= thresholds[loan_type]
            
            # Generate feedback
            feedback = self.generate_feedback(data, loan_type, approved, confidence)
            
            return {
                'approved': approved,
                'confidence': confidence,
                'feedback': feedback
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def generate_feedback(self, data: Dict, loan_type: str, approved: bool, confidence: float) -> List[str]:
        feedback = []
        strengths = []
        improvements = []
        
        # Add feedback logic here (same as before)
        # [Previous feedback generation code]
        
        return feedback

# Initialize predictor
predictor = LoanPredictor(model_data)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Loan Approval API"}

@app.post("/predict", response_model=LoanResponse)
async def predict_loan(application: LoanApplication):
    try:
        # Convert application to dict and predict
        application_dict = application.dict(exclude={'loan_type'})
        result = predictor.predict_loan_approval(application_dict, application.loan_type.lower())
        
        return LoanResponse(
            approved=result['approved'],
            confidence=result['confidence'],
            feedback=result['feedback']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))