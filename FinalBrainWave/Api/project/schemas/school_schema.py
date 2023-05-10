from typing import Optional
from pydantic import BaseModel, Field,validator
from datetime import datetime

from project.models.satellite_tables_model import Countries, School_types

class SchoolBase(BaseModel):
    desc_school: str = Field(
        ..., #los tres puntos indican que el campo será obligatorio
        example= 'Name School'
    )
    cif: str = Field(
        ..., #los tres puntos indican que el campo será obligatorio
        example= 'B0000000',
        max_length=12,
    )
    phone: str = Field(
        ...,
        max_length=15,
        example = '999999999'
    )
    zip_code: str = Field(
        ...,
        max_length=6,
        example = '15011'
    )
    email: str = Field(
        ...,
        example="info@school.com"
    )
    

class School(SchoolBase):
    school_id: int = Field(
        ...,        
    )
    country_id: int = Field(
        ..., #los tres puntos indican que el campo será obligatorio
        example=1 # Ejemplo de ID de país
    )
    type_id: int = Field(
        ..., #los tres puntos indican que el campo será obligatorio
        example=1 # Ejemplo de ID de tipo de colegio
    )
    

class SchoolRegister(SchoolBase):
    password: str = Field(
        ...,
        min_length=8,
        max_length=64,
        example="strongpass"
    )
    country_id: str = Field(
        ..., #los tres puntos indican que el campo será obligatorio
        example="España" # Ejemplo de país
    )
    type_id: str = Field(
        ..., #los tres puntos indican que el campo será obligatorio
        example="Concertado" # Ejemplo de tipo de colegio
    )
    city: str = Field(
        ...,
        example = 'La Coruña'
    )
        
    
    @validator('country_id')
    def validate_country_id(cls, v):
        country = Countries.get(Countries.desc_country == v)
        return country.country_id

    @validator('type_id')
    def validate_schooltype_id(cls, v):
        schooltype = School_types.get(School_types.desc_type == v)
        return schooltype.type_id
    
class SchoolRegisterResponse(SchoolRegister):    
    credits: int = Field(
        0,
        example = '0'
    )
    dt_insert: datetime = datetime.now()

class schoolUpdate(SchoolBase):    
    dt_update: Optional[datetime] = None

class SchoolPayments(SchoolBase):
    credits : int = Field(
        ...
    )


