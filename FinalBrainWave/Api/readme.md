# API BrainWave-BullyingProject

## Tabla de contenidos
1. [Información general del proyecto](#general-info)
2. [Tecnologías y dependencias](#technologies)
3. [Instalación](#installation)

<a name="general-info"></a>

## 1. Información general del proyecto

### 1.1: Introducción
El objetivo principal de esta API es proporcionar a los colegios una herramienta para prevenir y detectar el bullying en sus alumnos. La API permitirá a los colegios registrarse en la plataforma web y solicitar identificadores únicos para sus alumnos, lo que les permitirá responder a una encuesta. La API procesará los resultados de la encuesta a través de un modelo de Machine Learning en GCP para determinar si el alumno sufre o no de bullying. Toda el proyecto será desplegado en GCP.

### 1.2: Estructura del carpetas
El proyecto tiene la siguiente estructura de ficheros:
```
        fastapi/
        ├── __init__.py
        ├── main.py
        ├── .env
        ├── requirements.txt
        └── Project/
            ├── config.py
            ├── db.py
            ├── gcs.py
            ├── utils.py
            ├── models/
            │   ├── __init__.py
            │   ├── school_model.py
            │   ├── student_model.py
            │   ├── survey_model.py
            │   ├── answer_model.py
            │   ├── master_model.py
            │   └── satellite_tables_model.py
            ├── controllers/
            │   ├── __init__.py
            │   ├── auth_school.py
            │   ├── register_school.py
            │   ├── school_controllers.py
            |   ├── student_controllers.py
            │   └── survey_controllers.py
            ├── schemas/
            │   ├── __init__.py
            │   ├── school_schema.py
            |   ├── predict_schema.py
            │   ├── survey_schema.py
            |   ├── survey_question_schema.py
            │   ├── application_id_schema.py
            │   └── token_schema.py
            ├── endpoints/
            │   ├── __init__.py
            │   ├── school_endpoints.py
            │   └── surveys_endpoints.py
            └── service/
                ├── __init__.py
                └── predict.py
```
- __init__.py: es un archivo que Python utiliza para indicar que la carpeta es un paquete.
- config.py: archivo que contiene la configuración de la aplicación, como la cadena de conexión a la base de datos.
- main.py: archivo principal que ejecuta la aplicación.
- db.py: archivo que define y establece la conexión con la base de datos.
- .env: archivo que contiene variables de entorno que se cargan en la aplicación en tiempo de ejecución.
- utils.py: archivo que contiene funciones de utilidad genéricas.
- requirements.txt: archivo que contiene las dependencias del proyecto.
- models/: carpeta que contiene los modelos de la base de datos.
- controllers/: carpeta que contiene los controladores de la aplicación.
- schemas/: carpeta que contiene los esquemas de validación de datos.
- endpoints/: carpeta que contiene las rutas y controladores para los endpoints de la aplicación.
- machine_learning/: carpeta que contiene los archivos relacionados con el modelo de aprendizaje automático.

### 1.4: Modelos de datos:
 
Para realizar el modelado de los datos usaremos la libreria peewe. Peewee es un ORM (Object-Relational Mapping) para Python que se utiliza para trabajar con bases de datos relacionales. Un ORM es una técnica de programación que permite representar objetos de una aplicación en una base de datos, y viceversa, sin tener que escribir código SQL directamente. En lugar de eso, el ORM se encarga de traducir las operaciones de bases de datos en código Python.

Peewee es una biblioteca ligera y fácil de usar que admite múltiples bases de datos relacionales, como SQLite, MySQL y PostgreSQL, entre otras. Se integra bien con diferentes marcos web.

Entre las características notables de Peewee se incluyen:

- Soporte para múltiples bases de datos y tipos de campos, incluidos campos personalizados.

- Fácil creación y modificación de esquemas de bases de datos.

- Selección, actualización, inserción y eliminación de datos.

- Soporte para transacciones y bloqueo de bases de datos.

- Una sintaxis clara y concisa para consultas, y una API simple y fácil de usar.

Hemos definido los siguientes modelos:

####   School
- school_id: campo autoincremental que actúa como clave primaria de la tabla.
- desc_school: campo de tipo CharField que almacena la descripción de la escuela.
- cif: campo de tipo CharField que almacena el código de identificación fiscal de la escuela.
- phone: campo de tipo CharField que almacena el número de teléfono de la escuela.
- zip_code: campo de tipo CharField que almacena el código postal de la escuela.
- email: campo de tipo CharField que almacena la dirección de correo electrónico de la escuela.
- country_id: campo de tipo ForeignKeyField que hace referencia a la tabla de países y almacena el ID del país donde se encuentra la escuela.
- city: campo de tipo CharField que almacena la ciudad donde se encuentra la escuela.
- password: campo de tipo CharField que almacena la contraseña de acceso a la plataforma para la escuela.
- credits: campo de tipo IntegerField que almacena el número de créditos disponibles para la escuela.
- type_id: campo de tipo ForeignKeyField que hace referencia a la tabla de tipos de escuela y almacena el ID del tipo de escuela.
- dt_insert: campo de tipo DateTimeField que almacena la fecha y hora de inserción del registro.
- dt_update: campo de tipo DateTimeField que almacena la fecha y hora de la última actualización del registro.
- comments: campo de tipo TextField que almacena comentarios adicionales sobre la escuela.
- disable: campo de tipo BooleanField que indica si la escuela está deshabilitada o no.

#### Master

- entry_id: campo autoincremental que actúa como clave primaria de la tabla. 
- school_id: campo de tipo ForeignKeyField que hace referencia a la tabla School y almacena el ID de la escuela asociada al registro. 
- student_id: campo de tipo ForeignKeyField que hace referencia a la tabla Student y almacena el ID del estudiante asociado al registro.
- model_id: campo de tipo ForeignKeyField que hace referencia a la tabla SurveyModels y almacena el ID del modelo de encuesta utilizado para el registro. 
- prediction: campo de tipo IntegerField que almacena el resultado de la predicción realizada para el registro. 
- prob_prediction: campo de tipo IntegerField que almacena la probabilidad asociada al resultado de la predicción.
- date_insert: campo de tipo DateTimeField que almacena la fecha y hora de inserción del registro.
- date_update: campo de tipo DateTimeField que almacena la fecha y hora de la última actualización del registro.
- comments: campo de tipo TextField que almacena comentarios adicionales sobre el registro.

#### School_types
El modelo "School_types" define una tabla en la base de datos que almacena información sobre los diferentes tipos de escuelas que pueden ser registradas en el sistema. Esta tabla tiene los siguientes campos:

- type_id: campo autoincremental que actúa como clave primaria de la tabla.
- desc_type: campo de tipo CharField que almacena la descripción del tipo de escuela.
- dt_insert: campo de tipo DateTimeField que almacena la fecha y hora de inserción del registro en la tabla. 
- dt_update: campo de tipo DateTimeField que almacena la fecha y hora de la última actualización del registro en la tabla.

#### Countries
El modelo "Countries" define una tabla en la base de datos que almacena información sobre los países donde se encuentran las diferentes escuelas registradas en el sistema. Esta tabla tiene los siguientes campos:
country_id: campo autoincremental que actúa como clave primaria de la tabla. •
desc_country: campo de tipo CharField que almacena la descripción del país. 
dt_insert: campo de tipo DateTimeField que almacena la fecha y hora de inserción del registro en la tabla. 
dt_update: campo de tipo DateTimeField que almacena la fecha y hora de la última actualización del registro en la tabla.


#### Student
El modelo "Student" define una tabla en la base de datos que almacena información sobre los estudiantes registrados en una escuela y sus comentarios y calificaciones sobre las encuestas realizadas. Esta tabla tiene los siguientes campos:
- student_id: campo de tipo TextField que actúa como clave primaria de la tabla y almacena el ID del estudiante. 
- school_id: campo de tipo ForeignKeyField que hace referencia a la tabla School y almacena el ID de la escuela asociada al estudiante.
- comments: campo de tipo TextField que almacena los comentarios del estudiante sobre las encuestas realizadas. 
- dt_insert: campo de tipo DateTimeField que almacena la fecha y hora de creación del registro. 
- dt_update: campo de tipo DateTimeField que almacena la fecha y hora de la última actualización del registro. 
- credits: campo de tipo IntegerField que almacena los créditos del estudiante. 
- times_done: campo de tipo IntegerField que almacena la cantidad de veces que el estudiante ha completado las encuestas.

#### SurveyModels
El modelo "SurveyModels" define una tabla en la base de datos que almacena información relacionada con los modelos de encuestas. Esta tabla tiene los siguientes campos:
- model_id: campo autoincremental que actúa como clave primaria de la tabla. 
- desc_survey: campo de tipo CharField que almacena la descripción del modelo de encuesta. 
- date_insert: campo de tipo DateTimeField que almacena la fecha de inserción del registro. 
- date_update: campo de tipo DateTimeField que almacena la fecha de actualización del registro. 
- comments: campo de tipo TextField que almacena comentarios adicionales sobre el modelo de encuesta.

#### Survey_Questions
El modelo "Survey_Questions" define una tabla en la base de datos que almacena información relacionada con las preguntas de los modelos de encuestas. Esta tabla tiene los siguientes campos:
- model_id: campo de tipo ForeignKeyField que hace referencia a la tabla SurveyModels y almacena el ID del modelo de encuesta asociado a la pregunta. 
- answer_id: campo que hace referencia al tipo de respuestas de la pregunta del test.
- order_num: campo de tipo IntegerField que almacena el orden de la pregunta dentro del modelo de encuesta. 
- quest: campo de tipo TextField que almacena el texto de la pregunta.

####  Response_answers
El modelo  "Response_answers", almacena las posibles respuestas a las preguntas del test, contiene los siguientes campos:
- answer_id: campo de tipo clave primaria
- answer: columna donde se almacena los distintos tipos de respuestas

#### answers
El modelo "answers" almacena las respuestas de los alumnos, contiene los siguientes campos:
- answe_student_id: clave primaria, identificador de las respuestas del alumno
- entry_id: clave foranea de la tabla master
- model_id: clave foranea de la tabla survey_models
- answer_json: columna donde se almacena el json con las respuesta del alumno.

### 1.4: Autentificación 
El endpoint para la autenticación es /token. La autenticación utiliza el esquema de autenticación OAuth2, es decir, permite a las aplicaciones obtener tokens de acceso en nombre de un usuario al enviar las credenciales del usuario directamente al servidor de autorización.
En este flujo, el usuario proporciona sus credenciales de inicio de sesión (cif y contraseña)a la app, que a su vez envía una solicitud de token de acceso al servidor de autorización. El servidor de autorización autentica al usuario y, si las credenciales son correctas, emite un token de acceso al cliente. Este token puede usarse para acceder a los recursos protegidos por el servidor de recursos (API).
Para obtener un token de acceso, se debe enviar una solicitud POST a /token con los siguientes datos de formulario:
 - cif: identificación fiscal del centro.
 - password: Contraseña.
 
Si las credenciales son válidas, el servidor responderá con un token de acceso válido.
Los endpoints que requieren autenticación utilizan el esquema de autenticación Bearer con el token de acceso recibido en el paso anterior.
Los endpoints que requieren autenticación son:
 /payment 
/survey/applicate_id.

### 1.5: Endpoints:

Para la creación de los esquemas de datos de las solicitudes y respuestas, hemos utilizado la librería pydantic. Pydantic es una biblioteca de Python que se utiliza para validar y serializar datos diseñada para trabajar con estructuras de datos que se utilizan comúnmente en aplicaciones web, como JSON y YAML.
Con estos esquemas de datos conseguimos que sea más fácil desarrollar aplicaciones web que sean seguras y estén bien documentadas.

La aplicación consta de los siguiente endpoints:

#### School_endpoints.py
Estos endpoints forman parte de una API de gestión de colegios y autenticación de usuarios mediante tokens:
- Método: POST, Endpoint: /register/
        Descripción: Crea un nuevo colegio en la app.
        Parámetros: Se reciben los campos del colegio en formato JSON.
        Respuesta: Retorna la información del colegio registrado.
        
        
        formato peticion - application/json
        
               {
                "desc_school": "Name School",
                "cif": "B0000000",
                "phone": "999999999",
                "zip_code": "15011",
                "email": "info@school.com",
                "password": "strongpass",
                "country_id": "España",
                "type_id": "Concertado",
                "city": "La Coruña"
                }
                
        formato respuesta - application/json
        
                {
                  "desc_school": "Name School",
                  "cif": "ccccc",
                  "phone": "999999999",
                  "zip_code": "15011",
                  "email": "info@school.com",
                  "school_id": 20,
                  "country_id": 1,
                  "type_id": 2
                }
                
       
- Método: GET, Endpoint: /school/me
        Descripción: Retorna la información del colegio actualmente autenticado.
        Parámetros: Se espera un token de autenticación en la cabecera de la petición.
        Respuesta: Retorna la información del colegio actualmente autenticado.

        formato petición -application/json
         {
          "access_token": "string",
          "token_type": "string"
        }
        
- Método: POST, Endpoint: /token
        Descripción: Inicia sesión y devuelve un token de acceso.
        Parámetros: Se espera que se envíen las credenciales del usuario en formato de formulario.
        Respuesta: Retorna un objeto Token con el token de acceso y el tipo de token.
        formato - application/x-www-form-urlencoded
        username y password
        
        formato respuesta -application/json
         {
          "access_token": "string",
          "token_type": "string"
        }

        
- Método: POST, Endpoint: /payment
        Descripción: Procesa un pago.
        Parámetros: Se espera que se envíe el monto del pago y un token de autenticación en la cabecera de la petición.
        
- Método: POST, Endpoint: /survey/applicate_id
        Descripción: Crea los identificadores para que los estudiantes tengan acceso a la encuesta.
        Parámetros: Se espera un objeto ApplicationIDCreate con la cantidad de estudiantes, el curso y la clase a la que pertenecen, y un token de         autenticación en la cabecera de la petición.
        Respuesta: Retorna un diccionario con los identificadores de los estudiantes creados.
        
        formato peticion -application/json
                {
          'PRIMARIA': {
              'Curso 1': {'Aula 1': 3, 'Aula 2': 2}, 
              'Curso 2': {'Aula 1': 3, 'Aula 2': 5, 'Aula 3': 4}
          }, 
          'SECUNDARIA': {
              'Curso 1': {'Aula 1': 5, 'Aula 2': 5, 'Aula 3': 5}, 
              'Curso 2': {'Aula 1': 6, 'Aula 2': 6, 'Aula 3': 6, 'Aula 4': 6}, 
              'Curso 3': {'Aula 1': 10, 'Aula 2': 10}, 
              'Curso 4': {'Aula 1': 5, 'Aula 2': 5, 'Aula 3': 5}
          }, 
          'BACHILLERATO': {
              'Curso 1': {'Aula 1': 10, 'Aula 2': 10, 'Aula 3': 10}, 
              'Curso 2': {'Aula 1': 9, 'Aula 2': 9, 'Aula 3': 9}, 
              'Curso 3': {'Aula 1': 4, 'Aula 2': 4, 'Aula 3': 4, 'Aula 4': 4, 'Aula 5': 4}
                      }
                  }

        formato respuesta -application/json
        
                  {
        'PRIMARIA': {'Curso 1': {'Aula 1': ['202341PRI1Au1001']}}, 
        'SECUNDARIA': {'Curso 1': {'Aula 1': ['202341SEC1Au1001']}}, 
        'BACHILLERATO': {'Curso 1': {'Aula 1': ['202341BAC1Au1001']}}
                    }
        
- Método: GET, Endpoint: /profile
        Descripción: Realiza consulta a la base de datos de las fechas en las que el colegio realizo los test para mostrarlos   en la web
        Parámetros: Solo recibe por parámetro el token de autentificacion

        Respuesta: Devuelve un json con las fechas de los test solicitados 
        
        formato respuesta -application/json 
         
- Método: POST, Endpoint: /query
        Descripción: Realiza la consulta de los student_id y sus predicciones.
        Parámetros: Recibe una fecha en formato 'April - 2023', y un parámetro llamado order_by que indica a la api como ordenar el resultado de la busqueda, unicamente adminte 'student_id' y 'prediction'
        Respuesta: Devuelve un json con los id de estudiante y sus predicciones

        formato petición -application/json

        ```
        {
          "order_by": "string",
          "date": "string"
        }

        ```

        formato respuesta -application/json

        ```

        ``` 
        
#### Survey_endpoints.py
Los siguientes endpoints están relacionados con las encuestas
- Método: POST, Endpoint: /survey/questions/{student_id}
        Descripción: Permite acceder a la encuesta a través del identificador del alumno.
        Parámetros: student_id: Identificador del alumno que se desea obtener la encuesta.
         
        Respuesta: Retorna un diccionario con las preguntas de la encuesta y las opciones de respuesta para cada una. 

         formato peticion - string       
                
          "student_id": "string",
                
                
         formato respuesta -application/json        
    ```

     {    
    "questions": [
    {
      "quest": "¿Qué edad tienes?",
      "answer": "[]"
    },
    {
      "quest": "¿En qué grado estás?",
      "answer": "[]"
    },
    {
      "quest": "¿Cuánto mides sin zapatos? (Nota: los datos están en metros.)",
      "answer": "[]"
    },
    {
      "quest": "¿Cuánto pesas sin zapatos? (Nota: los datos están en kilogramos.)",
      "answer": "[]"
    },
    {
      "quest": "¿Cuál es tu género?",
      "answer": "[\"Mujer\", \"Hombre\"]"
    },
    {
      "quest": "¿Con qué frecuencia pasaste hambre durante los últimos 30 días porque no había suficiente comida en tu hogar?",
      "answer": "[\"Nunca\", \"Rara vez\", \"Algunas veces\", \"Casi siempre\", \"Siempre\"]"
    },
    {
      "quest": "¿Durante los últimos 7 días, cuántas veces comiste verduras, como lechuga, tomates, zanahorias o calabaza?",
      "answer": "[\"No comí durante los últimos 7 días\", \"1 a 3 veces durante los últimos 7 días\", \"4 a 6 veces durante los últimos 7 días\", \"1 vez al día\", \"2 veces al día\", \"3 veces al día\", \"4 o más veces al día\"]"
    },
    {
      "quest": "¿Durante los últimos 7 días, cuántas veces bebiste una lata, botella o vaso de refresco carbonatado, como Coca-Cola, Fanta, Pepsi, Seven Up o Pritty?",
      "answer": "[\"No comí durante los últimos 7 días\", \"1 a 3 veces durante los últimos 7 días\", \"4 a 6 veces durante los últimos 7 días\", \"1 vez al día\", \"2 veces al día\", \"3 veces al día\", \"4 o más veces al día\"]"
    },
    {
      "quest": "¿Durante los últimos 7 días, cuántas veces comiste fruta, como manzanas, plátanos, naranjas o mandarinas?",
      "answer": "[\"No comí durante los últimos 7 días\", \"1 a 3 veces durante los últimos 7 días\", \"4 a 6 veces durante los últimos 7 días\", \"1 vez al día\", \"2 veces al día\", \"3 veces al día\", \"4 o más veces al día\"]"
    },
    {
      "quest": "¿Con qué frecuencia comiste comida de un restaurante de comida rápida, como McDonalds, Burger King, Mostaza, o un vendedor de hamburguesas o hot dogs, durante los últimos 7 días?",
      "answer": "[\"0 días\", \"1 días\", \"2 días\", \"3 días\", \"4 días\", \"5 días\", \"6 días\", \"7 días\"]"
    },
    {
      "quest": "Durante los últimos 30 días, ¿cuántos días has fumado cigarrillos?",
      "answer": "[\"0 días\", \"1 días\", \"2 días\", \"3 días\", \"4 días\", \"5 días\", \"6 días\", \"7 días\"]"
    },
    {
      "quest": "Durante los últimos 30 días, ¿cuántos días has usado productos de tabaco distintos de cigarrillos, como pipas, narguile o tabaco sin humo?",
      "answer": "[\"0 días\", \"1 días\", \"2 días\", \"3 días\", \"4 días\", \"5 días\", \"6 días\", \"7 días\"]"
    },
    {
      "quest": "Durante los últimos 30 días, ¿cuántos días has consumido al menos una bebida que contenga alcohol?",
      "answer": "[\"0 días\", \"1 días\", \"2 días\", \"3 días\", \"4 días\", \"5 días\", \"6 días\", \"7 días\"]"
    },
    {
      "quest": "Durante los últimos 7 días, ¿cuántos días has estado físicamente activo durante un total de al menos 60 minutos al día?",
      "answer": "[\"0 días\", \"1 días\", \"2 días\", \"3 días\", \"4 días\", \"5 días\", \"6 días\", \"7 días\"]"
    },
    {
      "quest": "Durante los últimos 7 días, ¿cuántos días has caminado o andado en bicicleta hacia o desde la escuela?",
      "answer": "[\"0 días\", \"1 días\", \"2 días\", \"3 días\", \"4 días\", \"5 días\", \"6 días\", \"7 días\"]"
    },
    {
      "quest": "¿Cuántas veces resultaste gravemente herido durante los últimos 12 meses?",
      "answer": "[\"Ninguna\", \"1 vez\", \"2 o 3 veces\", \"4 o 5 veces\", \"6 o 7 veces\", \"8 o 9 veces\", \"10 u 11 veces\", \"12 o mas veces\"]"
    },
    {
      "quest": "¿Cuántas veces intentaste suicidarte realmente durante los últimos 12 meses?",
      "answer": "[\"Ninguna\", \"1 vez\", \"2 o 3 veces\", \"4 o 5 veces\", \"6 o 7 veces\", \"8 o 9 veces\", \"10 u 11 veces\", \"12 o mas veces\"]"
    },
    {
      "quest": "¿Cuántas veces fuiste físicamente atacado durante los últimos 12 meses?",

      "answer": "[\"Nunca usé drogas\", \"7 años o menos\", \"8 o 9 años\", \"10 o 11 años\", \"12 o 13 años\", \"14 o 15 años\", \"16 o 17 años\", \"18 años o más\"]"
    },
    {
      "quest": "Durante tu vida, ¿cuántas veces has consumido marihuana (también conocida como porro, faso o churro)?",
      "answer": "[\"0 veces\", \"1 o 2 veces\", \"3 a 9 veces\", \"10 a 19 veces\", \"20 o más veces\"]"
    },
    {
      "quest": "Durante los últimos 30 días, ¿cuántas veces has consumido marihuana (también conocida como porro, faso o churro)?",
      "answer": "[\"0 veces\", \"1 o 2 veces\", \"3 a 9 veces\", \"10 a 19 veces\", \"20 o más veces\"]"
    },
    {
      "quest": "Durante tu vida, ¿cuántas veces has consumido anfetaminas o metanfetaminas?\"",
      "answer": "[\"0 veces\", \"1 o 2 veces\", \"3 a 9 veces\", \"10 a 19 veces\", \"20 o más veces\"]"
    },
    {
      "quest": "¿A qué edad tuviste relaciones sexuales por primera vez?",
      "answer": "[\"Nunca tuve relaciones sexuales\", \"11 años o menos\", \"12 años\", \"13 años\", \"14 años\", \"15 años\", \"16 o 17 años\", \"18 años o más\"]"
    },
    {
      "quest": "Durante tu vida, ¿con cuántas personas has tenido relaciones sexuales?",
      "answer": "[\"Nunca tuve relaciones sexuales\", \"1 persona\", \"2 personas\", \"3 personas\", \"4 personas\", \"5 personas\", \"6 o más personas\"]"
    },
    {
      "quest": "La última vez que tuviste relaciones sexuales, ¿tú o tu pareja usaron algún otro método anticonceptivo, como el retiro, el ritmo seguro, las píldoras anticonceptivas o cualquier otro método para prevenir el embarazo?",
      "answer": "[\"Nunca tuve relaciones sexuales\", \"Si\", \"No\"]"
    },
    {
      "quest": "La última vez que tuviste relaciones sexuales, ¿tú o tu pareja usaron un condón?",
      "answer": "[\"Nunca tuve relaciones sexuales\", \"Si\", \"No\"]"
    },
    {
      "quest": "Durante este año escolar, ¿cuántos días a la semana has asistido a clases de educación física (EF)?",
      "answer": "[\"0 días\", \"1 días\", \"2 días\", \"3 días\", \"4 días\", \"5 días o más\"]"
    },
    {
      "quest": "¿Durante los últimos 30 días, en cuántos días faltaste a clases o a la escuela sin permiso?",
      "answer": "[\"0 días\", \"1 días\", \"2 días\", \"3 días\", \"4 días\", \"5 días o más\"]"
    },
    {
      "quest": "¿Cuánto tiempo pasas durante un día típico o normal sentado y viendo televisión, jugando videojuegos, hablando con amigos o realizando",
      "answer": "[\"Menos de 1 hora al día\",\"1 a 2 horas al día\", \"3 a 4 horas al día\",\"5 a 6 horas al día\", \"7 a 8 horas al día\",\"Más de 8 horas por día\"]"
    },
    {
      "quest": "¿Con qué frecuencia te preocupaste tanto por algo que no pudiste dormir por la noche durante los últimos 12 meses?",
      "answer": "[\"Nunca\",\"Rara vez\",\"Algunas veces\",\"Casi siempre\",\"Siempre\"]"
    },
    {
      "quest": "¿Durante los últimos 30 días, con qué frecuencia tus padres o tutores sabían realmente lo que estabas haciendo con tu tiempo libre?",
      "answer": "[\"Nunca\",\"Rara vez\",\"Algunas veces\",\"Casi siempre\",\"Siempre\"]"
    },
    {
      "quest": "¿Durante los últimos 30 días, con qué frecuencia la mayoría de los estudiantes en tu escuela fueron amables y serviciales?",
      "answer": "[\"Nunca\",\"Rara vez\",\"Algunas veces\",\"Casi siempre\",\"Siempre\"]"
    },
    {
      "quest": "¿Durante los últimos 30 días, con qué frecuencia tus padres o tutores verificaron si habías hecho tu tarea?",
      "answer": "[\"Nunca\",\"Rara vez\",\"Algunas veces\",\"Casi siempre\",\"Siempre\"]"
    },
    {
      "quest": "¿Durante los últimos 30 días, con qué frecuencia tus padres o tutores entendieron tus problemas y preocupaciones?",
      "answer": "[\"Nunca\",\"Rara vez\",\"Algunas veces\",\"Casi siempre\",\"Siempre\"]"
    },
    {
      "quest": "¿Con qué frecuencia te sentiste solo durante los últimos 12 meses?",
      "answer": "[\"Nunca\",\"Rara vez\",\"Algunas veces\",\"Casi siempre\",\"Siempre\"]"
    },]
    }

    ```
- Método: POST, Endpoint: /survey/submit
        Descripción: Recoje json con respuestas de encuesta del alumno y el student_id. Realiza el registro de la encuesta en la tabla master, almacena en la tabla answers las respuestas y realiza la prediccion y la almacena.
        Parámetros: student_id: Identificador del alumno que se desea obtener la encuesta.
         answers: Objeto SurveyQuestions con las preguntas de la encuesta.
        Respuesta: retorna un mesaje con informando sobre el preceso

        formato peticion - application/json         
                
          {
          "student_id": "string",
          "answers": 
                    {
                      "additionalProp1": 0,
                      "additionalProp2": 0,
                      "additionalProp3": 0
                    }
          }
               
                
         formato respuesta -application/json   

<a name="technologies"></a>

## 2. Tecnologías y dependencias

### 2.1 Tecnologías

#### FastApi

Para la realización de esta API, hemos optado por usar el marco web basado en Python, Fastapi. Su estructura de desarrollo es similar a Flask.
Fastapi es fácil de usar y su [documentación](https://fastapi.tiangolo.com/es/) es clara. Además ofrece un alto rendimiento, y genera la documentación de forma automática con un esfuerzo mínimo por parte del desarrollador. Esta información se puede encontrar en el directorio /docs de la aplicación.  La documentación contiene información detallada sobre puntos finales de API, códigos de retorno, parámetros de respuesta y otros detalles.

#### App Engine de Google Cloud Platform

Para realizar el despliegue del proyecto de la API en cloud hemos optado por [App Engine](https://cloud.google.com/appengine).  Google App Engine es otro de los servicios que conforman la familia de Google Cloud Platform. Este servicio es del tipo Plataforma como Servicio o Platform as a Service (PaaS), nos permite publicar aplicaciones web en línea sin necesidad de preocuparnos por la parte de la infraestructura y con un enfoque 100% en la construcción de nuestra aplicación y en la posibilidad de correrla directamente sobre la infraestructura de Google, es decir, la que Google usa para sus propios productos.

Como cualquier otra Plataforma como Servicio, App Engine nos facilita construir, mantener y escalar nuestra aplicación en la medida que sea necesario. Cuando usamos Google App Engine (GAE) no nos tenemos que preocupar por la escalabilidad de nuestra aplicación ya que cuenta con un balanceador de carga y escalamiento automático.

Así nuestra aplicación solamente será atendida por las máquinas necesarias para tener un perfecto comportamiento y para que la respuesta de nuestra aplicacion sea la más óptima.


### 2.2 Dependencias
Las dependencias utilizadas en el proyecto son:

- fastapi==0.95.0: Es un framework web para construir APIs rápidas y escalables con Python 3.6+ basado en estándares abiertos. Proporciona herramientas para la validación de datos, la documentación de API y la autenticación de usuario.
- uvicorn==0.21.1: Es un servidor web asincrónico basado en ASGI (Asynchronous Server Gateway Interface) que permite servir aplicaciones web construidas con el framework FastAPI.
- google-auth: Es una biblioteca de autenticación para Python que permite autenticar con la API de Google Cloud Platform y otras APIs de Google.
- google-auth-oauthlib: Es una biblioteca de autenticación de OAuth 2.0 para Google APIs.
- google-auth-httplib2: Es una biblioteca de autenticación de HTTP para Google APIs.
- google-cloud-storage: Es una biblioteca que permite interactuar con Google Cloud Storage desde Python.
- pandas: Es una biblioteca de análisis de datos de código abierto para Python que proporciona estructuras de datos y herramientas para el análisis de datos.
- numpy: Es una biblioteca de cálculo numérico para Python que proporciona una gran cantidad de funciones matemáticas y de álgebra lineal.
- scikit-learn: Es una biblioteca de aprendizaje automático de código abierto para Python que proporciona herramientas para la minería de datos y el análisis de datos.
- psycopg2-binary==2.9.5: Es un adaptador de base de datos PostgreSQL para Python que permite interactuar con bases de datos PostgreSQL desde Python.
- dotenv: Es una biblioteca que permite cargar variables de entorno desde un archivo .env en la raíz del proyecto.
- jwt==1.3.1: Es una biblioteca que permite codificar y decodificar tokens de autenticación JSON Web Tokens (JWT) en Python.
- peewee==3.16.0: Es una biblioteca de ORM (Object Relational Mapper) de Python que proporciona una forma sencilla de interactuar con bases de datos relacionales desde Python.
- pydantic==1.10.7: Es una biblioteca que proporciona herramientas para la validación de datos y la serialización de objetos en Python.
- bcrypt==1.7.4: Es una biblioteca de hash de contraseñas en Python que proporciona herramientas para la generación y verificación de contraseñas seguras.
- python-jose==3.3.0: Es una biblioteca de Python para JSON Object Signing and Encryption (JOSE) que proporciona herramientas para codificar y decodificar tokens de autenticación JSON Web Tokens (JWT) y para cifrar y descifrar datos en JSON.
        
<a name="installation"></a>

## 3. Instalación

#### 3.1 Local

Levantar la aplicación en local es una tarea muy sencilla. Lo primero que vamos a hacer es clonar el repositorio de proyecto global en nuestro equipo. Para ello ejecutaremos en consola dentro de la carpeta donde queramos clonar el proyecto, el siguiente comando.

```
$ git clone https://github.com/BrainWaveBullying/BullyingProject.git
```

Una vez tengamos clonado el repositorio navegaremos hasta la carpeta del proyecto de la api con el siguiente comando

```
$ cd API/
```

Una vez dentro de esa carpeta, crearemos el entorno virtual para instalar todas la dependencias necesarias y lo activaremos

```
$ virtualenv env
$ cd env/Scripts/activate
```

y volvemos a la carpeta inicial ejecutando dos veces el comando siguente

```
cd ..
```

Tras estos pasos procederemos a la instalacion de las bibliotecas relacionadas con el proyecto, éstas se encuentran en  requirements.txt . Para instalar todos los módulos de una sola vez, ejecute el siguiente comando.

```
$ pip install -r requirements.txt
```

Con esto tendremos el entorno virtual listo para poder levantar la aplicación. Para ello usaremos el siguiente comando en consola

```
uvicorn main:app --reload
```

Y en este momento tendremos la aplicación levantada en https://localhosts:8000, y lista para funcionar.


#### 3.2 Despliegue en GCP(App Engine)

##### Requisitos previos
- Cuenta de Google Cloud Platform con cuenta de facturación activada.
- Tener creado un proyecto en Google Cloud Platform 

#### Configuración de fichero YAML para implementar FastAPI en Google App Engine

Google Cloud Platform permite que App Engine realice la implementación en función de una configuración definida en un archivo yaml. Para que alojemos FastAPI en Google App Engine, la configuración de yaml debe tener la siguiente configuración.

```
runtime: python37
entrypoint: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```
Nuestro repositorio tiene un archivo  app.yaml  en la raíz del proyecto FastAPI, que tiene la configuración yaml especificada anteriormente para ayudar a implementar FastAPI en App Engine.

#### Clonar el repositorio de git en GCP
 
Primeramente debemos activar el CloudShell y posteriormente escribiremos el siguiente comando

```
git clone https://github.com/BrainWaveBullying/BullyingProject.git

```
Navegaremos al directorio de trabajo mediante el siguiente comando:

```
cd API/
```

#### Creación y activacion del entorno virtual

Crearemos el entorno mediante el siguiente comando

```
virtualenv env
```

Con el siguiente comando lo activaremos

```
source env/bin/activate
```

#### Instalamos las dependencias 

Ejecutamos el siguiente comando

```
pip install requirements.txt
```

#### Vista previa de la aplicación

Tras los pasos anteriores verificaremos si existe algun problema antes de compilar el proyecto en app engine. Para ello comprobaremos si la app se levanta sin problemas mediante el siguiente comando

```
gunicorn -w 4 -k uvicorn.workers.UvicornWorker principal:aplicación
```

Para obtener una vista previa, haga clic en el botón Vista previa en la parte superior derecha de la capa de la nube, como se muestra. Seleccione cambiar puerto y elija 8000 como nuestro puerto en el que se puede obtener una vista previa de nuestra aplicación. Elegimos 8000 porque nuestra aplicación está escuchando en ese puerto.

#### Implementación de la aplicación FastAPI en App Engine

Para implementar la aplicación FastAPI en App Engine y acceder a ella a través de un dominio personalizado o por defecto your-proj-id.appspot.com, necesitamos crear la aplicación gcloud e implementar nuestra aplicación.

Escribiremos en consola el siguiente comando

```
gcloud app create
```

Seleccionaremos la región.

#### Compliaremos la aplicación FastAPI en GCP App Engine

Es en este punto donde usarmos el archivo yaml en donde esta la configuración de implementación para desplegar la app
mediante el siguiente comando

```
gcloud app deployd app.yaml
```

Seguiremos los pasos que nos va indicando la shell y tras unos minutos la aplicación estara funcionando en cloud. Tras este último proceso nos indicará la url donde está desplegada la API



 












