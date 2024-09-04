from pydantic import BaseModel, Field, field_validator
from typing import Optional


class FeedbackModel(BaseModel):
    query_id: int = Field(..., ge=1)
    is_correct: bool 
    corrected_intent: Optional[str] = Field(None, max_length=100)
      
    @field_validator('corrected_intent', mode='before')
    def check_corrected_intent(cls, v, info):
        if info.data['is_correct'] is False and not v:
            raise ValueError('corrected_intent is required \
                              when is_correct is False')
        return v
      

class InferenceResponseModel(BaseModel):
    predicted_intent: str
    confidence_score: float
    query_id: int