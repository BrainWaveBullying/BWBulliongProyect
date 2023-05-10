from fastapi import HTTPException, status


from project.models.school_model import School as SchoolModel
from project.schemas import school_schema
from project.controllers.auth_school import get_password_hash

def create_school(school: school_schema.SchoolRegisterResponse):

    get_school = SchoolModel.filter(SchoolModel.cif == school.cif)
    
    if get_school:
        msg = "El CIF ya está registrado"
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )
    #Creamos el objeto de la bbdd
    db_school = SchoolModel(
        desc_school = school.desc_school,
        cif = school.cif,
        phone = school.phone,
        zip_code = school.zip_code,
        email = school.email,
        country_id = school.country_id,
        city = school.city,
        password=get_password_hash(school.password),
        credits = 0,
        type_id = school.type_id,
                 )
    #lo guardamos
    db_school.save()

    # Retornamos el objeto creado
    return school_schema.School(
        school_id = db_school.school_id,
        desc_school = db_school.desc_school,
        cif = db_school.cif,
        phone = db_school.phone,
        zip_code = db_school.zip_code,
        email = db_school.email,
        country_id = str(db_school.country_id),
        city = db_school.city,
        type_id = str(db_school.type_id),
        
    )