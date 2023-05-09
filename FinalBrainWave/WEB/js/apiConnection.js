//-- Constante de modo debug en local
const debug_mode = false;

/*
** Constante del endpoint base de la API **

-- Para trabajar con la api en local: 'http://127.0.0.1:8000/'
-- Para trabajar con la API remota: 'https://brainwave-382317.ew.r.appspot.com/'
*/

const base_endpoint = "http://127.0.0.1:8000/";

class ApiFormAuthRequest {
  /**
   * Esta clase se encargara de enviar formularios html a la Api con autenticacion y procesar una respuesta
   * 
   * param apiUrl (str): Url de la API completa
   * param data (dict): El diccionario json que va a recibir la API
   * param callback (function): La función que se va a encargar de realizar una tarea con los datos devueltos
   */
    constructor(apiUrl, data, callback) {
      this.apiUrl = apiUrl;
      this.data = data;
      this.callback = callback;
      console.log(data)
    }
  
    async postData() {
      const authToken = checkAuthentication();
      console.log(authToken)
      try {
        const response = await fetch(this.apiUrl, {
          method: "POST",
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json'
        },
          body: JSON.stringify(this.data),
        });
  
        const responseData = await response.json();
  
        if (this.callback) {
          this.callback(responseData);
        }
  
        return responseData;
      } catch (error) {
            console.error("Error en la petición:", error);
        throw error;
      }

    }
}


class ApiFormRequest {
/**
 * Esta clase se encargara de enviar formularios html a la Api y procesar una respuesta
 * 
 * param apiUrl (str): Url de la API completa
 * param data (dict): El diccionario json que va a recibir la API
 * param callback (function): La función que se va a encargar de realizar una tarea con los datos devueltos
 */
  constructor(apiUrl, data, callback) {
    this.apiUrl = apiUrl;
    this.data = data;
    this.callback = callback;
      console.log(data)
  }

  async postData() {
    try {
      const response = await fetch(this.apiUrl, {
        method: "POST",
        headers: {
          'Content-Type': 'application/json'
      },
        body: JSON.stringify(this.data),
      });

      const responseData = await response.json();

      if (this.callback) {
        this.callback(responseData);
      }

      return responseData;
    } catch (error) {
          console.error("Error en la petición:", error);
      throw error;
    }

  }
}


class ApiLoginAndGetTokenRequest {
  /**
   * Esta clase se encargara de enviar los datos de login a la API y almacenar el token en la sesion del navegador
   * 
   * param username (str): Nombre de usuario
   * param password (pass): Password del usuario
   * param endpoint (str): Endpoint de la api donde se valida el inicio de sesion
   * param redirectPage (str): Ruta a la pagina donde se redirige al usuario en caso de login exitoso
   */
  constructor(username, password, endpoint, redirectPage) {
      this.username = username;
      this.password = password;
      this.endpoint = endpoint;
      this.redirectPage = redirectPage;
  }   

  async login() {
    
    try {
        const response = await fetch(this.endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `grant_type=password&username=${encodeURIComponent(this.username)}&password=${encodeURIComponent(this.password)}`
        });

        if (response.status === 200) {
            const data = await response.json();
            console.log(data.access_token)
            // Almacenar el token en sessionStorage
            sessionStorage.setItem('authToken', data.access_token);

            // Redirigir al usuario a una página restringida o protegida
            window.location.href = this.redirectPage;
            
        } else {
            throw new Error('Credenciales incorrectas');
        }
    } catch (error) {
        console.error('Error al iniciar sesión:', error);
        // Mostrar un mensaje de error al usuario
    }
  }
}


class QuizApp {
  /**
   * Esta clase se encarga de generar el quiz dinamico y enviar los datos a la API una vez finalizado
   * 
   * param questionTextId (str): Id del Div que albergara las preguntas
   * param answerSelectId (str): Id del Select desplegable que contendra las respuestas
   * param nextButtonId (str): Id del Button que permite ir a la pregunta siguiente
   * param thanksContainerId (str): Id del Div que mostrara un texto al acabar el quiz
   * homeButtonId (str): Id del Button que redirige al usuario a la pagina de inicio una vez ha terminado
   * apiUrl (str): Ruta a la raiz de la API (por ejemplo "http://localhost:8000")
   * fetchQuestionsEndpoint (str): Endpoint de la API que nos da el cuestionario por ej: ("/questions")
   * submitAnswersEndpoint (str): Endpoint de la API que va a procesar el cuestionario completo
   */
  constructor(data, questionTextId, answerSelectId, nextButtonId, thanksContainerId, homeButtonId, apiUrl, fetchQuestionsEndpoint, submitAnswersEndpoint) {
    this.data = data;
    this.questionText = document.getElementById(questionTextId);
    this.answerSelect = document.getElementById(answerSelectId);
    this.nextButton = document.getElementById(nextButtonId);
    this.thanksContainer = document.getElementById(thanksContainerId);
    this.homeButton = document.getElementById(homeButtonId);
    this.apiUrl = apiUrl;
    this.fetchQuestionsEndpoint = fetchQuestionsEndpoint;
    this.submitAnswersEndpoint = submitAnswersEndpoint;

    // Crear un input type number aquí y ocultarlo inicialmente
    this.numberInput = document.createElement("input");
    this.numberInput.setAttribute("type", "number");
    this.numberInput.setAttribute("required", "");
    this.numberInput.style.display = "none";
    this.answerSelect.parentElement.insertBefore(this.numberInput, this.answerSelect);


    //-- Definimos las propiedades de clase que vamos a usar 
    this.startedQuizData = "";
    this.currentQuestionIndex = 0;
    this.userAnswers = {
      student_id: data,
      answers: {},
    };

    //-- Asignamos los listener a los eventos de pulsar estos dos botones
    this.nextButton.addEventListener("click", () => this.onNextButtonClick());
    this.homeButton.addEventListener("click", () => this.onHomeButtonClick());

    //-- Llamamos al metodo que extrae la data de la API e inicia el proceso
    this.fetchQuizData(this.data);
  }

  async fetchQuizData(data) {
    
    try {
      const data2 = {"student_id": data}
      const response = await fetch(this.apiUrl + this.fetchQuestionsEndpoint + data, {
        method: 'POST',
        //-- NO TENGO CLARO SI REQUIERE AUTH, teóricamente no debería ir aquí
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data2),
      });
  
      if (response.ok) {
        const responseData = await response.json(); 
        //console.log("RESPONSEDATA QUESTIONS");
        //console.log(responseData["questions"]);
        this.startedQuizData = responseData["questions"];
        this.initializeQuiz(responseData["questions"]);
      } else {
        console.error("Error al obtener los datos del quiz:", response.status);
      }
    } catch (error) {
      console.error("Error al realizar la solicitud:", error);
    }
    
  }

  initializeQuiz(quizData) {
    const questionKeys = Object.keys(quizData);
    //console.log(questionKeys);

    if (this.currentQuestionIndex < questionKeys.length) {
      const questionKey = quizData[questionKeys[this.currentQuestionIndex]]["quest"];
      //console.log(questionKey);
      this.questionText.textContent = questionKey;

      const answers = JSON.parse(quizData[questionKeys[this.currentQuestionIndex]]["answer"]);
      //console.log(answers);

      // Cambia el contenido de answerSelect a un div vacío
      this.answerSelect.innerHTML = "";

      if (answers.length > 0) {
        this.answerSelect.style.display = "block";
        this.numberInput.style.display = "none";

        for (const answer of answers) {
          const option = document.createElement("option");
          option.value = answer;
          option.textContent = answer;
          this.answerSelect.appendChild(option);
        }

        this.answerSelect.selectedIndex = -1;
      } else {
        // Mostrar el input type number y ocultar el select
        this.answerSelect.style.display = "none";
        this.numberInput.style.display = "block";
        this.numberInput.value = "";
      }
    } else {
      this.sendAnswersToApi();
    }
  }

  onNextButtonClick() {
    let selectedAnswerValue;

    if (this.numberInput.style.display === "block") {
      selectedAnswerValue = this.numberInput.value;
    } else {
      selectedAnswerValue = this.answerSelect.value;
    }

    // Verifica si se ha seleccionado una respuesta en el caso del select, o si se ha ingresado un valor en el caso del input number
    if ((this.answerSelect.style.display === "block" && this.answerSelect.selectedIndex !== -1) || (this.numberInput.style.display === "block" && this.numberInput.value !== "")) {
      this.userAnswers.answers[this.questionText.textContent] = selectedAnswerValue;
      console.log(this.userAnswers);
      this.currentQuestionIndex++;
      this.initializeQuiz(this.startedQuizData);
    } else {
      alert("Por favor, selecciona o ingresa una respuesta.");
    }
  }

  async sendAnswersToApi() {
    //-- Ojo que lo que envía es el índice de la respuesta, no el valor
    try {
      console.log(this.userAnswers)
      const response = await fetch(this.apiUrl + this.submitAnswersEndpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(this.userAnswers),
      });
      

      if (response.ok) {
        const result = await response.json();
        console.log("RESPUESTA: ");
        console.log(result["master"]);
        if (result["master"] === true) {
          this.questionText.style.display = "none";
          this.answerSelect.style.display = "none";
          this.nextButton.style.display = "none";
          this.thanksContainer.style.display = "block";
          const thanksMessage = document.getElementById("thanks-message");
          thanksMessage.textContent = "Gracias por responder, en breve te contactaremos.";
        } else {
          console.log(response);
          throw new Error("La API devolvió un resultado incorrecto.");
        }
      } else {
        throw new Error("Error al enviar las respuestas a la API: " + response.status);
      }
    } catch (error) {
      console.error(error);
      this.questionText.style.display = "none";
      this.answerSelect.style.display = "none";
      this.nextButton.style.display = "none";
      this.thanksContainer.style.display = "block";
      const thanksMessage = document.getElementById("thanks-message");
      thanksMessage.textContent = "Ha ocurrido un error al enviar las respuestas: " + error;
    }
  }

  onHomeButtonClick() {
    window.location.href = '../index.html'; 
  }
}


class GenerateDynamicKeys {
  constructor(contenedorId, maxCursos, maxAulas, maxAlumnos) {
    this.app = document.getElementById(contenedorId);
    this.maxCursos = maxCursos;
    this.maxAulas = maxAulas;
    this.maxAlumnos = maxAlumnos;

    this.init();
  }

  init() {
    //-- Selector de cursos
    this.cursoSelect = this.createSelect(1, this.maxCursos);
    this.cursoSelect.setAttribute("class", "selectorCurso");
    
    this.app.appendChild(this.cursoSelect);

    //-- Contenedor para los cursos
    this.cursosContainer = document.createElement("div");
    this.app.appendChild(this.cursosContainer);

    //-- Escuchar cambios en el selector de cursos
    this.cursoSelect.addEventListener("change", (e) => {
      const numCursos = parseInt(e.target.value);
      this.generarCursos(numCursos);
    });
  }

  createSelect(min, max) {
    const select = document.createElement("select");
    const option0 = document.createElement("option");
    option0.value = 0;
    option0.textContent = "Seleccione";
    select.appendChild(option0);
    for (let i = min; i <= max; i++) {
      const option = document.createElement("option");
      option.value = i;
      option.textContent = i;
      select.appendChild(option);
    }
    return select;
  }

  generarCursos(numCursos) {
    this.cursosContainer.innerHTML = "";
    for (let i = 1; i <= numCursos; i++) {
      const cursoDiv = document.createElement("div");

      const label = document.createElement("label");
      label.setAttribute("class", "labelCurso");
      label.textContent = `Curso ${i}`;
      cursoDiv.appendChild(label);

      const aulasSelect = this.createSelect(1, this.maxAulas);
      aulasSelect.setAttribute("class", "selectorAula");
      aulasSelect.dataset.curso = i;
      aulasSelect.addEventListener("change", (e) => {
        const numAulas = parseInt(e.target.value);
        const curso = parseInt(e.target.dataset.curso);
        this.generarAulas(numAulas, curso);
      });
      cursoDiv.appendChild(aulasSelect);

      const aulasContainer = document.createElement("div");
      cursoDiv.appendChild(aulasContainer);

      this.cursosContainer.appendChild(cursoDiv);

    }
  }

  generarAulas(numAulas, curso) {
    const aulasContainer = this.cursosContainer.children[curso - 1].lastElementChild;
    aulasContainer.innerHTML = "";
    for (let i = 1; i <= numAulas; i++) {
      
      const aulaDiv = document.createElement("div");

      const label = document.createElement("label");
      label.setAttribute("class", "labelAula");
      label.textContent = `Aula ${i}`;
      aulaDiv.appendChild(label);

      const alumnosSelect = this.createSelect(1, this.maxAlumnos);
      alumnosSelect.setAttribute("class", "selectorAlumnos");
      aulaDiv.appendChild(alumnosSelect);

      aulasContainer.appendChild(aulaDiv);
    }
  }

  obtenerDatos() {
    const json = {};
    for (const cursoDiv of this.cursosContainer.children) {
      const curso = {};
      const aulasContainer = cursoDiv.lastElementChild;
      for (const aulaDiv of aulasContainer.children) {
        const aula = aulaDiv.querySelector("select").value;
        const aulaName = aulaDiv.querySelector("label").textContent;
        curso[aulaName] = parseInt(aula);
      }
      const cursoName = cursoDiv.querySelector("label").textContent;
      json[cursoName] = curso;
    }
    return json;
  }
}


function checkAuthentication() {
  /**
   * Esta función valida si el usuario está autenticado usando el token de sesion almacenado en el navegador
   */
  const authToken = sessionStorage.getItem('authToken');
  console.log("El token almacenado en la cookie es");
  console.log({
    'Authorization': `Bearer ${authToken}`,
    'Content-Type': 'application/json'
  });

  if (!authToken) {
      // Si no hay token, redirigir al usuario a la página de inicio de sesión
      //window.location.href = '/ruta/a/la/pagina/de/login';
  } else {
      // Realizar una solicitud a tu API para verificar si el token es válido
      fetch('https://brainwave-382317.ew.r.appspot.com/school/me', {
          method: 'GET',
          headers: {
              'Authorization': `Bearer ${authToken}`,
              'Content-Type': 'application/json'
          }
      })
      .then(response => {
          if (response.status !== 200) {
              // Si el token no es válido, redirigir al usuario a la página de inicio de sesión
              //window.location.href = '/ruta/a/la/pagina/de/login';
              console.error(response)
              console.error("Respuesta de autenticacion != 200")
              return
          }
      })
      .catch(error => {
          console.error('Error al verificar el token:', error);
          return
      });
  }
  console.log("TODO Ok")
  return(authToken);
}
