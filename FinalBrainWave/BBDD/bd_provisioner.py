from hx_postgresSql_lib import PostgresConnMethods

"""
/**
-- INSTRUCCIONES:

-> Rellenar el create BD architecture y abajo rellenar los parametros del constructor de la clase (en la llamada)
*/
"""


class Provisioner(PostgresConnMethods):
    def __init__(self, host: str, passwd: str, user: str, db: str, schema: str):
        super().__init__(host, user, passwd, db, schema)
        self.host = host
        self.passwd = passwd
        self.user = user
        self.db = db
        self.schema = schema

    def createBdTablesArchitecture(self):
        print("PROVISIONER PASO 1: Creando schemas..")
        self.createSchema(self.schema)
        print("Schemas creados con exito\n")
        tables_list = list(self.getSchemaTables(self.schema))

        print("PROVISIONER PASO 2: Creando tablas...")

        ## -- crear tabla school_types y añadir la primary key
        if "school_types" not in tables_list:
            self.createTableBasic("school_types", """
                type_id serial,
                desc_type varchar (255) not null,
                dt_insert timestamp not null,
                dt_updated timestamp null default '2000-01-01 00:00:00'
            """)
            self.addPrimaryKey("school_types", "type_id_PK", "type_id")

        # -- TODO -> crear la tabla school, student dara error hasta que no se cree

        ## -- student: NOTA, TIENE CLAVE PRIMARIA COMPUESTA
        if "student" not in tables_list:
            self.createTableBasic("student", """
                        student_id serial,
                        school_id int not null,
                        credits int not null,
                        times_done int not null,
                        dt_insert timestamp not null,
                        dt_updated timestamp null default '2000-01-01 00:00:00',
                        comments varchar not null
                    """)
            self.addPrimaryKey("student", "student_school_id_PK", "student_id, school_id")
            self.addForeignKey("student", "student_school_id_FK", "school_id", "school", "school_id")
            self.addForeignKey("student", "credits_school_FK", "credits", "school", "credits")

        print(f"Tablas creadas con exito\n")
        self.closeConenection()


Provisioner(host="", passwd="", user="", db="", schema="").createBdTablesArchitecture()
