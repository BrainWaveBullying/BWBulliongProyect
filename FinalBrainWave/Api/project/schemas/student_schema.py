# Python
from datetime import datetime

# Pydantic
from pydantic import BaseModel
from pydantic import Field

class StudentBase(BaseModel):  
      
    school_id : int = Field(
        ...
    )
    comments: str = Field(
        max_length=200,
        example = 'cualquier comentario que pueda ser util'
    )
   
class StudentCreate(StudentBase):
    student_id: int = Field(...)

