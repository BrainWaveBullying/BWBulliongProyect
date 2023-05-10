from fastapi import APIRouter,Depends,status,Body,Query
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from typing import List, Dict

from project.schemas import school_schema
from project.schemas.predict_schema import PredictSchema
from project.schemas.token_schema import Token
from project.controllers.school_controllers import create_students_with_owner, read_predict,survey_date_query
from project.controllers.register_school import create_school
from project.controllers.auth_school import get_current_user,generate_token
from project.schemas.applicate_id_schema import NivelEducativoData
#from utils import create_students_with_owner

from project.models.school_model import School as SchoolModel

from project.db import get_db, db 

oauth2_scheme = OAuth2PasswordBearer('/token')

router = APIRouter(    
    tags = ['schools']
    )

@router.post (
    "/register",     
    status_code= status.HTTP_201_CREATED,
    response_model=school_schema.SchoolBase,
    dependencies=[Depends(get_db)],
    summary="Creación de nuevo colegio en base de datos"
)
def register_school(school: school_schema.SchoolRegister = Body(...)):
    """Crea un nuevo usuario en la app. recibirá los campos en un json
    desc_school 
    cif 
    phone
    cod_post
    email 
    country
    city 
    password 
    type
    date_insert 
    date_update 
    comments 
    
    return : info del colegio registrado """
    print(school)
    return create_school(school)

@router.get('/school/me',
        summary="comprueba que el token facilitado este verificado" )
def school_user(school: SchoolModel = Depends(get_current_user)):
    return  {'status' : 'OK' , 'user' : school}

@router.post(
    "/token",
    tags=["schools"],
    response_model=Token,
    summary="Obtener token de acceso una vez que se inicia la sesion para poder acceder a los endpoints autenticados",
    description="Obtener un token de acceso.",
)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    ## Iniciar sesión para el token de acceso

    ### Args
    La aplicación puede recibir los siguientes campos por datos de formulario       
    - CIF: Your CIF
    - password: Your password

    ### Returns
    - access token and token type
    """
    access_token = generate_token(form_data.username, form_data.password)
    return Token(access_token=access_token, token_type="bearer")

@router.post("/payment",
             summary= 'EN CONSTRUCCION')
async def process_payment(amount: int,  token: str = Depends(oauth2_scheme)):
    pass


@router.post(
    "/survey/applicate_id",
    status_code=status.HTTP_201_CREATED,
    response_model=Dict,
    summary="Crea los id de estudiantes que indiquen en el formulario por nivel, clase y aula",
    dependencies=[Depends(get_db), Depends(get_current_user), Depends(oauth2_scheme)])
@db.atomic()
def create_student_id(application = Body(...), current_user: SchoolModel = Depends(get_current_user)):    
    return create_students_with_owner(
        db = db, 
        application= application, 
        current_user= current_user        
        )

@router.get(
    "/profile",
    summary="Realiza consulta en base de datos de las fechas en las que el colegio realizó solicitudes de test'",
    dependencies=[Depends(get_db), Depends(get_current_user), Depends(oauth2_scheme)]
)
def date_query(current_user: SchoolModel = Depends(get_current_user)
    ):
    dates = survey_date_query(school_id=current_user.school_id)
    print(dates)
    formatted_dates = [d.dt_insert.strftime("%B - %Y") for d in dates]
    return {'dates': formatted_dates}

@router.post(
    "/query",
    summary="Realiza consulta los id de alumnos con sus predicciones para una fecha dada y los ordena por un campo dado ",
    dependencies=[Depends(get_db), Depends(get_current_user), Depends(oauth2_scheme)]
)
def query_predict(search_features : PredictSchema = Body(),
    current_user: SchoolModel = Depends(get_current_user)
    ):
    
    return read_predict(
        school_id = current_user.school_id,
        order_by = search_features.order_by,
        date = search_features.date,        
    )

"""
@router.post("/", response_model=SchoolRead)
    def create_school(school_data: SchoolCreate, controller: SchoolController = Depends()):
        return controller.create(school_data)

    @router.get("/{school_id}", response_model=SchoolRead)
    def get_school(school_id: int, controller: SchoolController = Depends()):
        return controller.get(school_id)

    @router.get("/", response_model=List[SchoolRead])
    def get_all_schools(controller: SchoolController = Depends()):
        return controller.get_all()

    @router.put("/{school_id}", response_model= dict)
    def update_school(school_id: int, school_data: SchoolUpdate, controller: SchoolController = Depends()):
        return controller.update(school_id, school_data)

    @router.delete("/{school_id}")
    def delete_school(school_id: int, controller: SchoolController = Depends()):
        controller.delete_school(school_id)
        return {"detail": "School deleted successfully"}"""


