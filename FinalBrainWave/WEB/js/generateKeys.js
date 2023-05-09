// Obtener los botones para generar las claves y para realizar el test
const generateKeysButton = document.getElementById("keys-button");
const testButton = document.getElementById("test-button");

// Agregar eventos click a los botones
generateKeysButton.addEventListener("click", function() {
  // Obtener los valores de los campos del formulario
  const curso = prompt("Ingrese el curso:");
  const clase = prompt("Ingrese la clase:");
  const numAlumnos = prompt("Ingrese el número de alumnos:");
  const password = prompt("Ingrese la contraseña:");
  
  // Crear un objeto FormData para enviar los datos al servidor
  const formData = new FormData();
  formData.append("password", password);
  formData.append("clase", clase);
  formData.append("curso", curso);
  formData.append("numAlumnos", numAlumnos);
  
  //// Enviar la información al servidor //////
  fetch("assets/generateKeys.html", {
    method: "POST",
    body: formData
  }).then(response => {
    console.log("La respuesta del servidor fue:", response);
  }).catch(error => {
    console.error("Ocurrió un error al enviar la información:", error);
  });
});

testButton.addEventListener("click", function() {
  ///// Obtener el valor del campo de la clave /////
  const clave = prompt("Ingrese la clave:");
  
  ///// Crear un objeto FormData para enviar los datos al servidor/////
  const formData = new FormData();
  formData.append("clave", clave);
  
  ///// Enviar la información al servidor//////
  fetch("assets/test.html", {
    method: "POST",
    body: formData
  }).then(response => {
    console.log("La respuesta del servidor fue:", response);
  }).catch(error => {
    console.error("Ocurrió un error al enviar la información:", error);
  });
});
