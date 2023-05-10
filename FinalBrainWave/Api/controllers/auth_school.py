from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from jose import JWTError, jwt
from passlib.context import CryptContext

from project.models.school_model import School as SchoolModel
from project.schemas.token_schema import TokenData
from project.config import Config

settings = Config()

#definimos las variables del token a traves de la instancia a la clase Config
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = int(settings.ACCESS_TOKEN_EXPIRE_MINUTES)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")  #Nos servirá para verificar la validez del password y para generar un hash a partir del password
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
#almacenamos en una variable las excepciones
credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

def verify_password(plane_password, password):
    """verifica que las dos password pasadas por parametro sean iguales"""
    return pwd_context.verify(plane_password, password)


def get_password_hash(password):
    """genera el hash del password que se le pasa por parámetro"""
    return pwd_context.hash(password)


def get_school(username: str):
    """pasamos por parámetro un mail y retorna true si existe en la base de datos"""
    if SchoolModel.get(SchoolModel.cif == username) :
        return SchoolModel.get(SchoolModel.cif == username)
    return None 


def authenticate_school(username: str, password: str):
    """esta funcion llama a la funcion get_user y verify_password para comprobar si existe el mail en bbdd y verifica la contraseña"""
    school = get_school(username)
    if not school:
        return credentials_exception
    if not verify_password(password, school.password):
        return credentials_exception
    return school


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """recibe un diccionario con la información que queremos guardar en el token y el tiempo de expiración de este y después lo genera 
    con la función jwt.encode que recibe por parámetro la información a guardar, nuestra clave secreta y el algoritmo que utilizaremos"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def generate_token(cif, password):
    """llama a la funcion authenticate_user para validar los datos. si no lo están lanzará una HTTPException, si todo va bien
    llamara a la función create_access_token y retornará el token"""
    school = authenticate_school(cif, password)
    
    if not school:
        raise credentials_exception
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return create_access_token(
        data={"sub": school.cif}, expires_delta=access_token_expires
    )

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Decodifica el token utilizando una clave secreta y un algoritmo de cifrado especificado.
    Luego, la función extrae el nombre de usuario del token decodificado y lo utiliza para buscar al usuario correspondiente en la base de datos.
    Si no se encuentra ningún usuario, se lanza una excepción HTTP 401. Si se encuentra el usuario, se devuelve el objeto de usuario. 
    Esta función se puede utilizar como decorador para proteger las rutas que requieren autenticación en una API. """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        cif: str = payload.get("sub")
        if cif is None:
            raise credentials_exception
        token_data = TokenData(cif=cif)
    except JWTError:
        raise credentials_exception

    school = get_school(username=token_data.cif)
    if school is None:
        raise credentials_exception
    if get_user_disabled_current(school):
        return school

def get_user_disabled_current(school: SchoolModel = Depends(get_current_user)):
    if school.disable:
        raise HTTPException(status_code=400, detail='Inactive School')
    return school