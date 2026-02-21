from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRow(BaseModel):
    Store: int = Field(..., ge=1)
    Date: str  # "YYYY-MM-DD"
    Promo: Optional[int] = 0
    StateHoliday: Optional[str] = "0"
    SchoolHoliday: Optional[int] = 0

class PredictRequest(BaseModel):
    rows: List[PredictRow]

class PredictResponse(BaseModel):
    predictions: List[float]