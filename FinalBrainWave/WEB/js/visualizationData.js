class SchoolInfo2 {
    constructor(data) {
        this.data = data;
        this.container = document.getElementById('displayContainer');
        this.createNiveles();
    }

    createTable(aula, aulaData) {
        const table = document.createElement("table");
        const thead = document.createElement("thead");
        const trHead = document.createElement("tr");
        const thAula = document.createElement("th");
        thAula.colSpan = 3;
        thAula.style.cursor = "pointer";
        thAula.innerText = aula;
        trHead.appendChild(thAula);
        thead.appendChild(trHead);
    
        const trSubHead = document.createElement("tr");
        trSubHead.style.display = "none"; // Aplicar el estilo aquí
        const thId = document.createElement("th");
        thId.innerText = "ID";
        const thPrediccion = document.createElement("th");
        thPrediccion.innerText = "Predicción";
        const thGrupo = document.createElement("th");
        thGrupo.innerText = "Grupo";
        trSubHead.appendChild(thId);
        trSubHead.appendChild(thPrediccion);
        trSubHead.appendChild(thGrupo);
        thead.appendChild(trSubHead);
        table.appendChild(thead);
    
        const tbody = document.createElement("tbody");
        tbody.style.display = "none";
        for (const id in aulaData) {
            console.log(id);
            const tr = document.createElement("tr");
            const tdId = document.createElement("td");
            tdId.innerText = id;
            const tdPrediccion = document.createElement("td");
            tdPrediccion.innerText = aulaData[id].prob_prediction;
            const tdGrupo = document.createElement("td");
            tdGrupo.innerText = aulaData[id].prob_category;
            tr.appendChild(tdId);
            tr.appendChild(tdPrediccion);
            tr.appendChild(tdGrupo);
            tbody.appendChild(tr);
        }
        table.appendChild(tbody);
    
        thAula.addEventListener("click", () => {
            const isCollapsed = tbody.style.display === "none";
            tbody.style.display = isCollapsed ? "" : "none";
            trSubHead.style.display = isCollapsed ? "" : "none"; // Cambiar el estilo aquí
        });
    
        return table;
    }

    createAulas(cursosData) {
        const aulasDiv = document.createElement("div");
        for (const aula in cursosData) {
            const table = this.createTable(aula, cursosData[aula]);
            aulasDiv.appendChild(table);
        }
        return aulasDiv;
    }

    createCursos(nivelData) {
        const cursosDiv = document.createElement("div");
        for (const curso in nivelData) {
            const cursoBtn = document.createElement("button");
            cursoBtn.style.cursor = "pointer";
            cursoBtn.innerText = "+ " + curso;
            cursosDiv.appendChild(cursoBtn);

            const aulasDiv = this.createAulas(nivelData[curso]);
            aulasDiv.style.display = "none";
            cursosDiv.appendChild(aulasDiv);

            cursoBtn.addEventListener("click", () => {
                const isCollapsed = aulasDiv.style.display === "none";
                aulasDiv.style.display = isCollapsed ? "" : "none";
                cursoBtn.innerText = (isCollapsed ? "- " : "+ ") + curso;
            });
        }
        return cursosDiv;
    }

    createNiveles() {
        for (const nivel in this.data) {
            const columnDiv = document.createElement("div");
            columnDiv.classList.add("column");
            const h2 = document.createElement("h2");
            h2.innerText = nivel;
            columnDiv.appendChild(h2);

            const cursosDiv = this.createCursos(this.data[nivel]);
            columnDiv.appendChild(cursosDiv);
            this.container.appendChild(columnDiv);

            h2.addEventListener("click", () => {
                const cursosDisplayed = Array.from(cursosDiv.children).filter(child => child.tagName === 'DIV');
                const anyVisible = cursosDisplayed.some(course => course.style.display !== 'none');
                cursosDisplayed.forEach(course => {
                    course.style.display = anyVisible ? 'none' : '';
                });
            });
        }
    }

    exportToExcel() {
        const wb = XLSX.utils.book_new();
    
        // Recorrer cada nivel y agregarlo como una hoja de cálculo
        for (const nivel in this.data) {
            let nivelData = [];
            let nivelObj = this.data[nivel];
    
            // Recorrer cada curso dentro del nivel
            for (const curso in nivelObj) {
                let cursoObj = nivelObj[curso];
    
                // Recorrer cada aula dentro del curso
                for (const aula in cursoObj) {
                    let aulaData = cursoObj[aula];
    
                    // Recorrer cada fila dentro del aula
                    for (const id in aulaData) {
                        let row = {
                            'Nivel': nivel,
                            'Curso': curso,
                            'Aula': aula,
                            'ID': id,
                            'Predicción': aulaData[id].prediccion,
                            'Grupo': aulaData[id].grupo
                        };
                        nivelData.push(row);
                    }
                }
            }
    
            const ws = XLSX.utils.json_to_sheet(nivelData);
            XLSX.utils.book_append_sheet(wb, ws, nivel);
        }
    
        XLSX.writeFile(wb, 'School_Info.xlsx');
    }

    exportToExcel2() {
        const wb = XLSX.utils.book_new();
    
        // Recorrer cada nivel y agregarlo como una hoja de cálculo
        for (const nivel in this.data) {
            let ws = XLSX.utils.aoa_to_sheet([]);
            let currentRow = 0;
            let nivelObj = this.data[nivel];
    
            // Recorrer cada curso dentro del nivel
            for (const curso in nivelObj) {
                let cursoObj = nivelObj[curso];
                XLSX.utils.sheet_add_aoa(ws, [[`${curso}`]], { origin: { r: currentRow, c: 0 } });
                currentRow++;
    
                // Recorrer cada aula dentro del curso
                for (const aula in cursoObj) {
                    let aulaData = cursoObj[aula];
                    XLSX.utils.sheet_add_aoa(ws, [[`${aula}`]], { origin: { r: currentRow, c: 1 } });
                    currentRow++;
    
                    // Agregar encabezados
                    const headers = [['ID', 'Predicción', 'Grupo']];
                    XLSX.utils.sheet_add_aoa(ws, headers, { origin: { r: currentRow, c: 2 } });
                    currentRow++;
    
                    // Recorrer cada fila dentro del aula
                    for (const id in aulaData) {
                        let row = [
                            id,
                            aulaData[id].prediccion,
                            aulaData[id].grupo
                        ];
                        XLSX.utils.sheet_add_aoa(ws, [row], { origin: { r: currentRow, c: 2 } });
                        currentRow++;
                    }
                }
            }
    
            XLSX.utils.book_append_sheet(wb, ws, nivel);
        }
    
        XLSX.writeFile(wb, 'School_Info.xlsx');
    }
  

}