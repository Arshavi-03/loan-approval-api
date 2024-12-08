from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class LoanType(str, Enum):
    STUDENT = "student"
    AGRICULTURAL = "agricultural"
    BUSINESS = "business"

class LoanApplication(BaseModel):
    loan_type: LoanType
    loan_amount: float = Field(..., description="Loan amount (standardized)")
    term: float = Field(..., description="Loan term (standardized)")
    int_rate: float = Field(..., description="Interest rate (standardized)")
    annual_income: float = Field(..., description="Annual income (standardized)")
    debt_to_income: float = Field(..., description="Debt to income ratio (standardized)")
    credit_score: float = Field(..., description="Credit score (standardized)")
    person_age: float = Field(..., description="Age (standardized)")
    income_to_loan: float = Field(..., description="Income to loan ratio (standardized)")
    
    # Optional fields with correct capitalization
    Education: float | None = Field(None, description="Education level for student loans")
    credit_risk: float | None = Field(None, description="Credit risk for student loans")
    Mortgage: float | None = Field(None, description="Mortgage for agricultural loans")
    person_home_ownership: float | None = Field(None, description="Home ownership for agricultural loans")
    emp_length: float | None = Field(None, description="Employment length")
    credit_card_usage: float | None = Field(None, description="Credit card usage for business loans")
    CreditCard: float | None = Field(None, description="Credit card info for business loans")

class LoanResponse(BaseModel):
    approved: bool
    confidence: float
    feedback: List[str]