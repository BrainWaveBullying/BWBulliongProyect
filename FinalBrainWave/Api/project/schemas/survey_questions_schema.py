# Python
from datetime import datetime

from typing import Dict
# Pydantic

from pydantic import BaseModel
from pydantic import Field


class SurveyQuestions(BaseModel):  
      
    student_id : str = Field(
        ...
    )
   
#############################################################


class SurveyAnswers(BaseModel):
    student_id: str
    answers: Dict[str, float]