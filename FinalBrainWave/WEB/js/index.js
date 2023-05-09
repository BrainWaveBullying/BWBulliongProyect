const $d = document;

//////////////////////// Texto flotante /////////////////
window.onload = function() {

    var elemento1 = document.querySelector('.flotante');
    var elemento2 = document.querySelector('.segundo');
    var elemento3 = document.querySelector('.tercero');

    elemento1.style.opacity = 0;
    elemento2.style.opacity = 0;
    elemento3.style.opacity = 0;

    setTimeout(function(){
      elemento1.style.opacity = 1;
      elemento1.style.transform = 'translate(0%, 100%)';
    }, 500);
    setTimeout(function(){
      elemento2.style.opacity = 1;
      elemento2.style.transform = 'translate(-50%, 100%)';
    }, 1000);
    setTimeout(function(){
        elemento3.style.opacity = 1;
        elemento3.style.transform = 'translate(-50%, 100%)';
      }, 1500);
  };
