from pydantic import BaseModel
from typing import Optional

class LoanApplication(BaseModel):
    loan_type: str
    loan_amount: float
    term: float
    int_rate: float
    annual_income: float
    debt_to_income: float
    credit_score: float
    person_age: float
    education: Optional[float] = None
    income_to_loan: float
    credit_risk: Optional[float] = None
    emp_length: Optional[float] = None
    person_home_ownership: Optional[float] = None
    mortgage: Optional[float] = None
    credit_card_usage: Optional[float] = None
    credit_card: Optional[float] = None

class LoanResponse(BaseModel):
    approved: bool
    confidence: float
    feedback: list[str]