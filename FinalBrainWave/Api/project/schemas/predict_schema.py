from typing import Optional
from pydantic import BaseModel, Field,validator
from datetime import datetime


class PredictSchema(BaseModel):

    order_by: str = Field(...)
    date: str = Field(...)
    