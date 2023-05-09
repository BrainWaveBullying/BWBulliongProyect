# 1: API BrainWave-BullyingProject

## 1.1: Introducción
Esta será la aplicación web que dará forma al proyecto. Su objetivo es proporcionar a las instituciones educativas una herramienta para detectar de forma más rápida si alguno de sus alumnos está sufriendo acoso escolar. Esta aplicación web trabaja en conjunto con la API para que los centros puedan registrarse y tener un entorno privado donde solicitar claves de acceso y visualizar resultados. Y a su vez, dónde cada alumno de forma individual y en cualquier lugar, sin necesidad de estar logueado pueda realizar el test mediante la clave que le proporciona el centro. De esta forma los resultados del test son totalmente privados y solo el centro tiene acceso a ellos, ya que es solo la institución quién sabe a qué alumno entrega cada clave. 

## 1.2: Estructura del proyecto
El proyecto tiene la siguiente estructura:
```
        WEB/
        ├── index.html
        ├── arquitecturaWeb.drawio
        ├── readme.md
        ├── assets/
        |   ├── centro.html
        |   ├── generateKeys.html
        |   ├── home.html
        |   ├── login.html
        |   ├── realizartest.html
        |   ├── Resultados.html
        |   ├── shop.html
        |   ├── successful.html
        |   ├── successKeys.html
        |   └── test.html
        ├── css/
        |   └── images
        |       ├── images/
        |       |    ├── fondodos.webp
        |       |    └── logo.png
        |       ├── centro.css
        |       ├── generateKeys.css
        |       ├── home.css
        |       ├── login.css
        |       ├── realizartest.css
        |       ├── Resultados.css
        |       ├── shop.css
        |       ├── styles.css
        |       ├── successful.css
        |       ├── successKeys.css
        |       └── test.css
        └── js
            ├── index.js
            |-- visualizationData.js     
            ├── apiConnection.js
            └── validations.js
```

**index.html**: es la página principal de la web
**arquitecturaWeb.drawio**: es el esquema de la web
**readme.md**: es la leyenda de toda la aplicación web
**assets/**: aquí se encuentran el resto de págs que componen la web
**css/**: contiene todos los archivos css que dan estilo a la web

## 1.3: Dependencias
**html5**: lenguaje con el que están creadas todas las págs que componen la web
**css3**: lenguaje de diseño gráfico que da estilo a todas las págs html
**javascript-ECMAScript**: lenguaje de programación font-end que proporciona el dinamismo a la web

## 1.4: Arquitectura
Esta sería la arquitectura de la web
```

                                                  ├── shop ──> shop
                                                  |
                                      ├── login ──├── generateKeys ──> successKeys (back centro)
                        ├──> centro ──|           |
                        |             |           ├── Resultados
└── index  ──> home   ──|             |         
                        |             ├── Create Account ──> successfull (back login)
                        |
                        |
                        ├──> alumno ──> realizartest ──> test

```

## 1.5: Funcionamiento
El flujo de control de la web se realiza usando javaScript tanto para interactuar con los elementos del DOM como para hacer las peticiones asíncronas a la API y procesar
la respuesta. 

**apConnections.js**: Contiene clases de Js que se encargan de realizar las diferentes peticiones-respuestas a la API

        - ApiFormAuthRequest: Recibe un diccionario json + tokenAuth + endpointAPI + callback. Envía el json al endpoint correspondiente comprobando que el usuario este
        autenticado. Si la petición se resuleve con éxito se desencadena el callback que ha recibido en el constructor

        - ApiFormRequest: Hace exactamente lo mismo que la anterior pero sin autenticar al usuario. Se usa para peticiones públicas 

        - ApiLoginAndGetTokenRequest: Se encarga de enviar una petición a la API para validar el login. Si todo ha ido bien, la API devuelve un tokenOAuth que se almacena
        en la sesión del navegador y redirige al usuario de centro a la pagina home.html

        - QuizApp: Es una de las clases mas complejas del módulo. Primero envía una petición a la API para comprobar si las credenciales del alumno son correctas. En caso
        afirmativo, la API devuelve un json cuyas 'keys' son las preguntas del test y cuyos 'values' una lista con las posibles respuestas. Dinámicamente se crean los titulos de las preguntas (con las keys del json recibido) y el selector(cuyos options son los diferentes elementos de la lista de values del json). Esto se realiza iterativamente de forma que las preguntas se muestran una a una y al pulsar el boton de siguiente se carga la siguiente pregunta y posibles respuestas. Una vez el indice del json ha llegado al final, se desencadena un evento que envía la información del test completo a la API asociado al id del estudiante. NOTA: Las preguntas 
        cuya respuesta no es un seleccionable modifican el display para mostrar un input type number.

        - GenerateDynamicKeys: Se encarga de ir generando dinámicamente selectores para Curso, Aulas y Alumnos para los 3 ciclos educativos que estamos contemplando: Primaria,
        Secundaria y Bachillerato. Se selecciona un numero de cursos en cada ciclo, en cada curso se selecciona un numero de aulas y en cada aula un numero de alumnos. Una vez
        adecuada la selección a la distribución del centro se genera un json que recoge esa información y la envía a la API para generar y almacenar las claves de cada alumno
        en la base de datos. (Las claves generadas así como los resultados del test si el alumno lo ha realizado pueden visualizarse y descargarse en resultados.html previa autenticación).

        - checkAuthentication: Esta función obtiene el token de autenticación almacenado en la sesión del navegador y lo envía a la API para confirmar su validez. En caso de
        que la validación sea correcta retorna el token, en caso contrario devuelve el error y redirige al usuario a la pagina de login.

**visualizationData.js**: Contiene la clase que se encarga de la visualizacion dinamica de resultados en Resultados.html.

        - SchoolInfo: Partiendo de un diccionario json genera dinámicamente el contenido del DOM en forma de tablas y asigna un evento de descarga en xlsx a un boton. En esencia se encarga de proporcionar una visualización sencilla e intuitiva para el usuario basada en la posibilidad de expandir y contraer elementos.

**validations.js**: Contiene varios métodos de validación de formularios en el front, previos a las peticiones a la API. Todo esto lo hace instanciando popups 
                que muestran la informacion de los errores cometidos a la hora de completar los formularios. También cabe destacar que 
                aqui se implementa la funcionalidad de mostrar las claves que se han generado en el generateKeys.html para que el usuario pueda verificar que todo es correcto antes de enviar la solicitud de creación a la API.


