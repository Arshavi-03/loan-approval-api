from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class LoanType(str, Enum):
    STUDENT = "student"
    AGRICULTURAL = "agricultural"
    BUSINESS = "business"

class LoanApplication(BaseModel):
    loan_type: LoanType
    # Common fields for all loan types
    loan_amount: float = Field(..., description="Loan amount (standardized)")
    term: float = Field(..., description="Loan term (standardized)")
    int_rate: float = Field(..., description="Interest rate (standardized)")
    annual_income: float = Field(..., description="Annual income (standardized)")
    debt_to_income: float = Field(..., description="Debt to income ratio (standardized)")
    credit_score: float = Field(..., description="Credit score (standardized)")
    person_age: float = Field(..., description="Age (standardized)")
    income_to_loan: float = Field(..., description="Income to loan ratio (standardized)")
    
    # Student loan specific fields
    Education: float | None = Field(None, description="Education level (standardized)")
    credit_risk: float | None = Field(None, description="Credit risk (standardized)")
    
    # Agricultural loan specific fields
    emp_length: float | None = Field(None, description="Employment length (standardized)")
    person_home_ownership: float | None = Field(None, description="Home ownership status (standardized)")
    Mortgage: float | None = Field(None, description="Mortgage status (standardized)")
    
    # Business loan specific fields
    credit_card_usage: float | None = Field(None, description="Credit card usage (standardized)")
    CreditCard: float | None = Field(None, description="Credit card status (standardized)")

class LoanResponse(BaseModel):
    approved: bool
    confidence: float
    feedback: List[str]