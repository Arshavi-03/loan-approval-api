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
        """Create interaction features matching the trained model"""
        if loan_type == 'student':
            X['score_income_ratio'] = X['credit_score'] * X['annual_income']
            X['age_education_factor'] = X['person_age'] * X['Education']  # Matches training capitalization
            X['debt_income_ratio'] = X['debt_to_income'] / (X['income_to_loan'] + 1)
            X['credit_term_factor'] = X['credit_score'] * X['term']
        
        elif loan_type == 'agricultural':
            X['credit_mortgage_ratio'] = X['credit_score'] * X['Mortgage']  # Matches training capitalization
            X['income_emp_factor'] = X['annual_income'] * X['emp_length']
            X['debt_asset_ratio'] = X['debt_to_income'] / (X['person_home_ownership'] + 1)
            X['loan_term_factor'] = X['loan_amount'] * X['term']
        
        elif loan_type == 'business':
            X['credit_card_income'] = X['credit_card_usage'] * X['annual_income']
            X['business_exp_factor'] = X['emp_length'] * X['annual_income']
            X['debt_credit_ratio'] = X['debt_to_income'] / (X['credit_score'] + 1)
            X['card_util_ratio'] = X['CreditCard'] * X['credit_card_usage']  # Matches training capitalization
        
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
    
        # Analyze credit score
        if data.get('credit_score', 0) > 0.5:
            strengths.append("Strong credit score")
        elif data.get('credit_score', 0) < -0.3:
            improvements.append("Consider improving your credit score")
        
        # Income analysis
        if data.get('income_to_loan', 0) > 0.5:
            strengths.append("Good income to loan ratio")
        elif data.get('income_to_loan', 0) < 0:
            improvements.append("Income might be low relative to requested loan amount")
        
        # Debt analysis
        if data.get('debt_to_income', 0) < -0.3:
            trengths.append("Low debt-to-income ratio")
        elif data.get('debt_to_income', 0) > 0.3:
            improvements.append("Consider reducing your debt burden")
        
        # Loan type specific feedback
        if loan_type == 'student':
            age_value = data.get('person_age', 0)
            # Convert standardized age to approximate real age
            approx_age = int(22 + (age_value * 5))  # Rough approximation
        
            if data.get('education', 0) > 0.5:
                strengths.append("Strong educational background")
            if approx_age < 22:
                improvements.append(f"Age ({approx_age} years) is on the lower side for student loans")
            
        elif loan_type == 'agricultural':
            if data.get('emp_length', 0) > 0.5:
                strengths.append("Sufficient agricultural experience")
            elif data.get('emp_length', 0) < 0:
                improvements.append("More agricultural experience would strengthen your application")
            if data.get('person_home_ownership', 0) > 0:
                strengths.append("Property ownership is a positive factor")
            if data.get('mortgage', 0) < -0.1:
                strengths.append("Good mortgage history")
            
        elif loan_type == 'business':
            if data.get('annual_income', 0) > 0.5:
                strengths.append("Strong business income")
            if data.get('emp_length', 0) > 0.5:
                strengths.append("Good business experience")
            elif data.get('emp_length', 0) < 0:
                improvements.append("More business experience would strengthen your application")
            if data.get('credit_card_usage', 0) < -0.3:
                strengths.append("Responsible credit card usage")
            elif data.get('credit_card_usage', 0) > 0.3:
                improvements.append("Consider reducing credit card usage")
    
        # Analyze loan amount
        if data.get('loan_amount', 0) < -0.3:
            strengths.append("Conservative loan request")
        elif data.get('loan_amount', 0) > 0.3:
            improvements.append("Consider requesting a lower loan amount")
    
        # Compile final feedback
        if approved:
            feedback.append(f"Congratulations! Your {loan_type} loan application is approved")
            feedback.append(f"Approval confidence: {confidence}%")
            if strengths:
                feedback.append("\nKey strengths in your application:")
                feedback.extend([f"- {s}" for s in strengths])
            if improvements:
                feedback.append("\nAreas for future improvement:")
                feedback.extend([f"- {s}" for s in improvements])
        else:
            feedback.append(f"Your {loan_type} loan application was not approved")
            if improvements:
                feedback.append("\nAreas needing improvement:")
                feedback.extend([f"- {s}" for s in improvements])
            if strengths:
                feedback.append("\nPositive aspects of your application:")
            feedback.extend([f"- {s}" for s in strengths])
            feedback.append("\nRecommendations:")
            feedback.append("- Consider applying for a lower loan amount")
            feedback.append("- Work on improving the highlighted areas")
            feedback.append("- You may reapply after addressing these factors")
        
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