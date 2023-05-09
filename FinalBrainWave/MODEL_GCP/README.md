# Apache Beam Pipeline y despliegue en GCP

## Tabla de contenidos
1. [Información general del proyecto](#general-info)
2. [Tecnologías y dependencias](#technologies)
3. [Configuración inicial del proyecto](#installation)

<a name="general-info"></a>

## 1. Información general del proyecto

### 1.1: Introducción
Este proyecto tiene como objetivo crear un pipeline para procesar datos de encuestas realizadas a alumnos y predecir si están siendo víctimas de acoso escolar (bullying) o no.

El pipeline se encarga de leer los datos de entrada desde un archivo en formato CSV, realizar una limpieza y transformación de los datos, y luego entrenar y testear un modelo de aprendizaje automático para generar las predicciones.

Para este proyecto se ha utilizado Apache Beam, un modelo de programación de datos unificado que permite el procesamiento de grandes conjuntos de datos de manera eficiente y escalable. Además, se ha utilizado Python como lenguaje de programación, lo que permite un fácil acceso a diversas bibliotecas de aprendizaje automático y análisis de datos.

La idea de este pipeline, es poder ejecutarlo tanto en local como en GCP, mediante DataFlow y AI Platform.

### 1.2: Estructura de carpetas
El proyecto tiene la siguiente estructura de ficheros:
```
Model_gcp/
|---- __init__.py
|---- preprocess.py
|---- trainer.py
|---- predict.py
|---- retrainer.py
|---- setup.py
|---- .env
|---- requirements.txt
|---- core/
      |---- config.py
```

- __init__.py: indica que la carpeta es un paquete de Python.
- preprocess.py: script que contiene funciones para el preprocesamiento de los datos de entrada del modelo.
- trainer.py: script que contiene la lógica del entrenamiento del modelo de predicción.
- setup.py: fichero en donde se define los metadatos y las dependencias del proyecto de Python. Se incluye info sobre el proyecto
- predict.py: script que contiene la lógica para hacer predicciones con el modelo entrenado.
- retrainer.py: script que se encargaría de reagrupar los datasets y reentrenar el modelo
- .env: archivo de configuración que contiene variables de entorno.
- requirements.txt: archivo que contiene una lista de dependencias del proyecto.
- core/config.py: archivo que contiene la configuración principal del proyecto, como los parámetros de entrenamiento, los directorios de entrada/salida y las credenciales de acceso a los servicios de Google Cloud.


### 1.3: Pasos generales de proyecto

Estos son los pasos generales:

1. Preprocesamiento de los datos
2. Entrenando al modelo
3. Predicciones
4. Agrupamiento de datasets y reentrenamiento


#### Preprocesamiento. (preprocess.py)

Esta es una pipeline de Apache Beam que realizará todo el preprocesamiento necesario para entrenar un modelo de Deep Learning. Utiliza tf.Transform , que es parte de TensorFlow Extended , para realizar cualquier procesamiento que requiera un pase completo sobre el conjunto de datos.

Puesto que entrenaremos una red neuronal para realizar las predicciones, siempre es una buena idea normalizar las entradas a un rango pequeño (normalmente de 0 a 1). Para realizar este tipo de normalizaciones es necesario revisar todo el conjunto de datos para encontrar los recuentos mínimo y máximo. Afortunadamente, tf.Transform se integra con nuestra canalización de Apache Beam y lo hace por nosotros. 

Este fichero tendrá dos flujos de trabajo dependiendo si se trata de un conjunto de train o un conjuto de test. 
En el caso de que se trate de un set de train el pipeline consta de varios pasos:

- Lectura de del csv 
- Transformación de datos a un diccionario teniendo encuenta el esquema registrado de las preguntas/respuestas registrado en la fichero core/config.py
- Validación de los datos
- Eliminar filas que puedan contener datos nulos
- Convertimos los datos a float
- Escalado y almacenado del transform(que luego utilizaremos para scalar los datos de test)
- Split de los datos para obtener un conjunto de evaluación
- Almacenar los datos de train y eval

En el caso de que se trate del conjunto de test:

- Lectura de del csv 
- Transformación de datos a un diccionario teniendo encuenta el esquema registrado de las preguntas/respuestas registrado en la fichero core/config.py
- Validación de los datos
- Eliminar filas que puedan contener datos nulos
- Convertimos los datos a float
- Descarga del transform alojado en el bucket y scalado de los datos
- Almacenar conjunto de test 

#### Entrenamiento del modelo (training.py)  (sin implementar)

Entrenaremos una red neuranal profunda a través de TensorFlow. Este apartado del proyecto utilizará los datos almacenados en el directorio de trabajo. Durante la etapa de preprocesamiento, tf.Transform generó un gráfico de operaciones para normalizar los datos. 
Para conjuntos de datos pequeños, será más rápido ejecutarlo localmente. Si el conjunto de datos de entrenamiento es demasiado grande, se escalará mejor para entrenar en AI Platform .

#### Predicciones (predict.py) (sin implementar)

En esta fase del proyecto, podríamos obtar por dos tipos de predicciones, por lotes o en stream. Puesto que las necesidades del proyecto no exigen una predicción inmediata, podriamos optar por una predicción por lotes, mejorando el rendimiento a costa de sacrificar latencia. 

#### Predicciones (retraining.py) (sin implementar)

Este script, realizará la agrupación del datasat original, con todos los test que se han ido realizando y almacenado en la instancia de postgre, por lo que la idea seria que realizara de forma programada dicha agruación y posterior entreno, realizando una valoración de los datos del modelo en servicio y del modelo reentrenado

<a name="technologies"></a>

## 2. Tecnologías y dependencias 

### Apache Beam

Apache Beam es un modelo unificado de código abierto para definir canalizaciones por lotes y de procesamiento paralelo de datos de transmisión. El modelo de programación de Apache Beam simplifica la mecánica del procesamiento de datos a gran escala. Con uno de los SDK de Apache Beam, puedes compilar un programa que define la canalización. Luego, uno de los backends admitidos de procesamiento distribuido de Apache Beam, como Dataflow, ejecuta la canalización. Este modelo te permite concentrarte en la composición lógica de los trabajos de procesamiento de datos en lugar de en la organización física del procesamiento paralelo. Puedes enfocarte en lo que necesitas que haga tu trabajo en lugar de en cómo se ejecuta.

Aquí hay algunas razones por las que hemos elegido Apache Beam para realizar un pipeline:

- Portabilidad: Apache Beam permite escribir pipelines de procesamiento de datos una vez y ejecutarlos en diferentes plataformas, lo que facilita la migración entre diferentes proveedores de servicios en la nube o la ejecución en diferentes entornos locales.

- Escalabilidad: Apache Beam proporciona una capa de abstracción sobre la infraestructura de procesamiento de datos subyacente, lo que permite aprovechar la escalabilidad y el paralelismo ofrecidos por diferentes plataformas de procesamiento de datos.

- Flexibilidad: Apache Beam permite definir pipelines de procesamiento de datos flexibles y escalables que pueden manejar diferentes tipos de datos y diferentes fuentes y destinos de datos.

- Facilidad de uso: La API unificada de Apache Beam hace que sea fácil de aprender y utilizar, lo que permite a los desarrolladores escribir pipelines de procesamiento de datos de manera más rápida y eficiente.

-multi-lenguaje: Otra ventaja que aporta Apache Bean es la capacidad de que cada runner funciona con cada lenguaje, por lo que se pueden implementar pipelines multi-lenguaje con transformaciones cross-language.

### DataFlow 

Dataflow es el servicio de procesamiento de datos serverless en Google Cloud Platform (GCP) que permite procesar y analizar grandes cantidades de datos en tiempo real o en batches de manera unificada. Es la solución estándar de ETL en Google Cloud, más moderna y ágil que alternativas como Dataproc.
Dataflow está basado en Apache Beam (proyecto open source que combina procesamiento streaming y batch, de donde viene su nombre) y permite crear flujos de trabajo para procesar, transformar y analizar datos utilizando una variedad de herramientas y lenguajes de programación.

### AI-Platform

AI Platform permite entrenar tus modelos de aprendizaje automático a gran escala, alojar tu modelo entrenado en la nube y usar tu modelo con el propósito de realizar predicciones sobre datos nuevos. El servicio de entrenamiento de AI Platform te permite entrenar modelos con una amplia variedad de opciones de personalización diferentes.
Puedes seleccionar numerosos tipos de máquina diferentes para potenciar tus trabajos de entrenamiento, habilitar el entrenamiento distribuido, usar el ajuste de hiperparámetros y acelerar con GPU y TPU.

<a name="installation"></a>

## 3 Configuración inicial del proyecto

El proyecto esta preparado para poder ejecutarlo en modo local o en CGP mediante DataFlow y AI-Platform.

### Clonar el repositorio

Primeramente debe clonar el repositorio de git y navegar al directorio principal, mediante los siguientes comandos:

```git clone https://github.com/BrainWaveBullying/BullyingProject.git
cd BullyingProject/MODEL_GCP
```

### Entorno virtual

Ejecute lo siguiente para configurar y activar un nuevo entorno virtual:

```
python3.7 -m virtualenv env
source env/bin/activate
```

### Instalación de las dependencias

Puede realizar la instalación de las dependencias necesarias mediante el archivo requirements.txt mediante el siguiente comando:

```
pip install -r requirements.txt
```




