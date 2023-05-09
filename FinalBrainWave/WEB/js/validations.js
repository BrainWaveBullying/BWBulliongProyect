//-- Funcion para validar el formulario de login
function validateLogin() {
    const errores = [];
    
    //-- Validacion de cif
    const inputCif = document.getElementById('username');
    if (inputCif.value.length != 9) {
      errores.push('El campo "CIF" debe contener 9 caracteres por ej: A12345678');
    }

    //-- Validacion de pass
    const inputPasswd = document.getElementById('password');
    if (inputPasswd.value.length < 8) {
        errores.push('El campo "Contraseña" debe contener al menos 8 caracteres');
      }
  
    return errores;
}

//-- Funcion para validar el formulario de registro
function validateRegister(){
    const errores = [];

    //-- Validacion de nombreCentro
    const inputNombreCentro = document.getElementById('desc_school');
    if (!inputNombreCentro.value) {
      errores.push('El campo "Nombre del centro" no puede estar vacío');
    }
    
    //-- Validacion de cif
    const inputCif = document.getElementById('cif');
    if (inputCif.value.length != 9) {
      errores.push('El campo "CIF" debe contener 9 caracteres por ej: A12345678');
    }

    //-- Validacion de telefoo
    const inputTlf = document.getElementById('phone');
    if (!inputTlf.value) {
      errores.push('El campo "Teléfono" no puede estar vacío');
    }

    //-- Validacion de mail
    const inputMail = document.getElementById('email');
    const emailValido = /^\S+@\S+\.\S+$/.test(inputMail.value);
    if (!emailValido) {
      errores.push('El campo "Email" debe contener una dirección válida');
    }

    //-- Validacion de CP
    const inputCp = document.getElementById('zip_code');
    if (!inputCp.value) {
      errores.push('El campo "Código postal" no puede estar vacío');
    }

    //-- Validacion de pass
    const inputPasswd = document.getElementById('password2');
    if (inputPasswd.value.length < 8) {
        errores.push('El campo "Contraseña" debe contener al menos 8 caracteres');
    }

    //-- Validacion de Pais
    const inputCountry = document.getElementById('country_id');
    if (!inputCountry.value) {
      errores.push('El campo "País" no puede estar vacío');
    }

    //-- Validacion de Ciudad
    const inputCity = document.getElementById('city');
    if (!inputCity.value) {
      errores.push('El campo "Ciudad" no puede estar vacío');
    }
    
  
    return errores;
}

//-- Funcion para validar el formulario de login del alumno antes de hacer el test
function validateStudentLogin() {
    const errores = [];
    
    //-- Validacion de cif
    const inputStudent = document.getElementById('student-id');
    if (inputStudent.value.length < 11) {
      errores.push('Debes introducir el ID en un formato adecuado para comenzar');
    }
    return errores;
}

//-- Funcion para mostrar errores de validacion en un popup
function popup(titulo, lista) {
    const popup = document.createElement('div');
    popup.id = 'popup';


    const h2 = document.createElement('h2');
    h2.innerText = titulo;
    popup.appendChild(h2);

    const ul = document.createElement('ul');
    lista.forEach(item => {
        const li = document.createElement('li');
        li.innerText = item;
        ul.appendChild(li);
    });
    popup.appendChild(ul);

    const botonEntendido = document.createElement('button');
    botonEntendido.innerText = 'Entendido';
    botonEntendido.addEventListener('click', () => {
        popup.remove();
    });
    popup.appendChild(botonEntendido);

    document.body.appendChild(popup);
}

//-- Funciones para mostrar la lista de cursos y aulas al pulsar EnviarInformacion en generateKeys
function createNestedList(data) {
    const ul = document.createElement('ul');

    for (const key in data) {
        const li = document.createElement('li');
        li.style.cursor = 'pointer';
        const span = document.createElement('span');
        span.innerText = '+ ' + key;
        li.appendChild(span);

        const subUl = document.createElement('ul');
        subUl.style.display = 'none';

        for (const subkey in data[key]) {
            const subLi = document.createElement('li');
            subLi.style.cursor = 'pointer';
            const subSpan = document.createElement('span');
            subSpan.innerText = '- ' + subkey;
            subLi.appendChild(subSpan);

            const aulaUl = document.createElement('ul');
            aulaUl.style.display = 'none';

            for (const aula in data[key][subkey]) {
                const aulaLi = document.createElement('li');
                aulaLi.innerText = '> ' + aula + ': ' + data[key][subkey][aula] + ' ALUMNOS';
                aulaUl.appendChild(aulaLi);
            }

            subLi.appendChild(aulaUl);
            subUl.appendChild(subLi);

            subSpan.addEventListener('click', (event) => {
                event.stopPropagation();
                aulaUl.style.display = aulaUl.style.display === 'none' ? '' : 'none';
            });
        }

        li.appendChild(subUl);
        ul.appendChild(li);

        span.addEventListener('click', () => {
            subUl.style.display = subUl.style.display === 'none' ? '' : 'none';
        });
    }

    return ul;
}

function popupConfirmKeys(titulo, data, button1Text, button1Callback, button2Text, button2Callback) {
    const popup = document.createElement('div');
    popup.id = 'popupKeys';

    const h2 = document.createElement('h2');
    h2.innerText = titulo;
    popup.appendChild(h2);

    const nestedList = createNestedList(data);
    popup.appendChild(nestedList);

    const boton1 = document.createElement('button');
    boton1.innerText = button1Text;
    boton1.addEventListener('click', () => {
        button1Callback();
        popup.remove();  
    });

    //boton1.addEventListener('click', button1Callback);
    popup.appendChild(boton1);

    const boton2 = document.createElement('button');
    boton2.innerText = button2Text;
    boton2.addEventListener('click', () => {
        button2Callback();
        popup.remove();
    });
    boton2.addEventListener('click', button2Callback);
    popup.appendChild(boton2);

    document.body.appendChild(popup);
}