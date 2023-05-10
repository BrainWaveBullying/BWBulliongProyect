import psycopg2
import datetime
import numpy as np


"""
/*
###### ------> VERSION: 2023.01.25

## -- REQUISITOS PARA CORRECTO FUNCIONAMIENTO DE TODOS LOS METODOS:

- Las columnas 'ID' deben ser strings (a menos que vaya con autoincrement)
- En caso de usar la funcion update, ten en cuenta que de existir una columna que almacene el valor 'fecha de insert'
    dicha columna debera llamarse 'FECHA_INSERT'
    
- Las funciones update e insert into solo aceptan tipos primitivos de python, si se pretende insertar objetos o 
    estructuras mas complejas, probablemente falle.

*/
"""


class Connection:

    def __init__(self, host: str, user: str, password: str, database: str):
        self.connection = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.connection.cursor()
        self.dataBase = database
        self.user = user

    def closeConenection(self):
        self.connection.close()

    def executeQuery(self, query: str, print_query: bool = False):
        if print_query:
            print(query)

        self.cursor.execute(query)
        self.connection.commit()


class BBDD:

    def __init__(self, host, user, password):
        self.connection = psycopg2.connect(
            host=host,
            user=user,
            passwd=password
        )
        self.cursor = self.connection.cursor()

    def closeConnection(self):
        self.connection.close()

    def executeQuery(self, query: str, print_query=False):
        if print_query:
            print(query)
        self.cursor.execute(query)
        self.connection.commit()

    def createDataBase(self, db_name: str):
        self.executeQuery("CREATE DATABASE IF NOT EXISTS " + db_name)

    def dropDataBase(self, db_name: str):
        self.executeQuery("DROP DATABASE IF EXISTS " + db_name)


class PostgresConnMethods(Connection):
    def __init__(self, host, user, password, database, schema):

        super().__init__(host, user, password, database)
        self.schema = schema

    def anyQuery(self, query: str, print_query: bool = False):
        self.executeQuery(query, print_query)
        return np.array(self.cursor.fetchall())

    @staticmethod
    def idStringFormat(valor: int, characters: int = 7) -> str:
        """
        Recoge un valor entero y lo transforma en un string de 5 digitos
        :param valor: El entero que se recibe
        :param characters: Numero de digitos que va a tener el resultado por defecto 5: 00000-99999
        :return: string
        """
        valor = str(valor)
        while len(valor) < characters:
            valor = "0" + valor
        return valor

    ###############################################################################
    ##############                      SCHEMAS                      ##############
    ###############################################################################
    def createSchema(self, schema_name):
        self.executeQuery(f"CREATE SCHEMA IF NOT EXISTS {schema_name} AUTHORIZATION {self.user}")

    def getSchemaTables(self, schema: str):
        self.executeQuery(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'")
        return np.array(self.cursor.fetchall()).flatten()

    ###############################################################################
    ##############                       TABLAS                      ##############
    ###############################################################################

    def createTableBasic(self, tablename: str, queryColumns: str, print_query: bool = False):
        self.executeQuery(f"CREATE TABLE IF NOT EXISTS {self.schema}.{tablename}({queryColumns})", print_query)

    def dropTable(self, tablename, print_query: bool = False):
        self.executeQuery(f"DROP TABLE IF EXISTS {self.schema}.{tablename}", print_query)

    def truncateTable(self, tablename, print_query: bool = False):
        self.executeQuery(f"TRUNCATE TABLE IF EXISTS {self.schema}.{tablename}", print_query)

    def getTableCols(self, tablename: str, print_query: bool = False) -> list:
        self.executeQuery(f"SELECT * FROM {self.schema}.{tablename} LIMIT 1", print_query)
        return [i[0] for i in self.cursor.description]

    ###############################################################################
    ##############                       SELECT                      ##############
    ###############################################################################

    def getLastIdValue(self, tablename: str, id_col_name: str) -> int:
        self.executeQuery(f"SELECT MAX({id_col_name}) FROM {self.schema}.{tablename}")
        try:
            return int(self.cursor.fetchall()[0][0])
        except IndexError:
            return int(0)
        except TypeError:
            return int(0)

    def limitSelectAll(self, tablename: str, limit: int, print_query: bool = False):
        self.executeQuery(f"SELECT * FROM {self.schema}.{tablename} LIMIT {limit}", print_query=print_query)
        return self.cursor.fetchall()

    def selectAll(self, tablename: str, whereCondition: str = None, orderCondition: str = None, print_query=False):
        whereSentence = ""
        orderSentence = ""
        if whereCondition is not None:
            whereSentence = "WHERE " + whereCondition
        if orderCondition is not None:
            orderSentence = "ORDER BY " + orderCondition

        query_to_execute = f"SELECT * FROM {self.schema}.{tablename} {whereSentence} {orderSentence}"
        self.executeQuery(query_to_execute, print_query=print_query)
        return self.cursor.fetchall()

    def selectSpecific(self, columnList: list, tablename: str, whereCondition: str = None,
                       orderCondition: str = None, print_query: bool = False):
        whereSentence = ""
        orderSentence = ""
        cols = ""
        for i in range(len(columnList)):
            if i == 0:
                cols = "" + columnList[i] + ""
            else:
                cols = cols + "," \
                              "" + columnList[i] + ""

        if whereCondition is not None:
            whereSentence = "WHERE " + whereCondition
        if orderCondition is not None:
            orderSentence = "ORDER BY " + orderCondition

        query_to_execute = f"SELECT {cols} FROM {self.schema}.{tablename} {whereSentence} {orderSentence}"
        self.executeQuery(query_to_execute, print_query)

        return np.array(self.cursor.fetchall()).flatten()

    def selectIdByDesc(self, tablename: str, column_id_name: str, column_desc_name: str, desc_value, print_query=False):

        self.executeQuery(
            f"SELECT {column_id_name} FROM {self.schema}.{tablename} WHERE {column_desc_name}='{desc_value}'",
            print_query)

        return np.array(self.cursor.fetchall()).flatten()[0]

    def selectDescById(self, tablename: str, column_id_name: str, column_desc_name: str, id_value: str,
                       print_query=False):

        self.executeQuery(
            f"SELECT {column_desc_name} FROM {self.schema}.{tablename} WHERE {column_id_name}='{id_value}'",
            print_query)
        return self.cursor.fetchall()[0][0]

    def selectColCount(self, tablename: str, column_id_name: str, print_query=False):
        self.executeQuery(f"SELECT COUNT({column_id_name}) FROM {self.schema}.{tablename}", print_query)
        return self.cursor.fetchall()[0][0]

    def selectIfRepeatValue(self, tablename: str, column_desc_name: str, desc_value, print_query=False) -> bool:
        """
        This method will be used to know if a value exists in one column
        :param tablename: Table
        :param column_desc_name: Column that owns the value
        :param desc_value: Value that may be repeated
        :param print_query: Choose print the query or not
        :return: True if value already exists, false else.
        """

        self.executeQuery(
            f"SELECT COUNT({column_desc_name}) FROM {self.schema}.{tablename} WHERE {column_desc_name}='{desc_value}'",
            print_query)

        if self.cursor.fetchall()[0][0] != 0:
            return True
        else:
            return False

    ###############################################################################
    ##############                    INSERT INTO                    ##############
    ###############################################################################

    def insertIntoSpecificCols(self, tablename: str, columnList: list, valueList: list, print_query=False):
        insertValues = ""
        columnNames = ""
        for i in range(len(valueList)):
            if i == 0:
                if type(valueList[i]) is str and valueList[i] != "NOW()":
                    insertValues = "'{}'".format(valueList[i])
                elif isinstance(valueList[i], datetime.datetime) or isinstance(valueList[i], datetime.date):
                    insertValues = "'{}'".format(valueList[i])
                else:
                    insertValues = "{}".format(valueList[i])
                columnNames = columnList[i]
            else:
                if type(valueList[i]) is str and valueList[i] != "NOW()":
                    insertValues = "{}, '{}'".format(insertValues, valueList[i])
                elif isinstance(valueList[i], datetime.datetime) or isinstance(valueList[i], datetime.date):
                    insertValues = "{}, '{}'".format(insertValues, valueList[i])
                else:
                    insertValues = "{}, {}".format(insertValues, valueList[i])
                columnNames = columnNames + "," + columnList[i]

        self.executeQuery(f"INSERT INTO {self.schema}.{tablename} ({columnNames}) VALUES ({insertValues})", print_query)

    ###############################################################################
    ##############                       UPDATE                      ##############
    ###############################################################################

    def updateSpecificCols(self, name_id_column, tablename, id_get_value, final_value_list, print_query: bool = False):
        table_cols_list = self.getTableCols(tablename)
        table_cols_list = [z.upper() for z in table_cols_list]

        # First, we remove the id col because primary key can't be updated
        if name_id_column in table_cols_list:
            table_cols_list.remove(name_id_column)

        # Second, we remove insert date if exists. SUPPOSED ITS NAME IS FECHA_INSERT
        if "FECHA_INSERT" in table_cols_list:
            table_cols_list.remove("FECHA_INSERT")

        # Third, we build the query using dynamic temp_col_list
        query = ""
        for i in range(len(final_value_list)):
            if i == 0:
                if type(final_value_list[i]) is str and final_value_list[i] != "NOW()":
                    query = f"UPDATE {self.schema}.{tablename} SET {table_cols_list[i]}='{final_value_list[i]}'"
                elif isinstance(final_value_list[i], datetime.datetime) or isinstance(final_value_list[i],
                                                                                      datetime.date):
                    query = f"UPDATE {self.schema}.{tablename} SET {table_cols_list[i]}='{final_value_list[i]}'"
                else:
                    query = f"UPDATE {self.schema}.{tablename} SET {table_cols_list[i]}={final_value_list[i]}"

            else:
                if type(final_value_list[i]) is str and final_value_list[i] != "NOW()":
                    query = f"{query}, {table_cols_list[i]}='{final_value_list[i]}'"
                elif isinstance(final_value_list[i], datetime.datetime) or isinstance(final_value_list[i],
                                                                                      datetime.date):
                    query = f"{query}, {table_cols_list[i]}='{final_value_list[i]}'"
                else:
                    query = f"{query} , {table_cols_list[i]}={final_value_list[i]}"

        # Fourth, we add the where clause
        query = f"{query} WHERE {name_id_column}={id_get_value}"

        # Last, we execute the query
        self.executeQuery(query, print_query=print_query)

    ###############################################################################
    ##############                       DELETE                      ##############
    ###############################################################################

    def deleteById(self, tablename: str, id_col_name: str, id_value: str, print_query: bool = False):
        query = f"DELETE FROM {self.schema}.{tablename} WHERE {id_col_name}='{id_value}'"
        self.executeQuery(query, print_query=print_query)

    ###############################################################################
    ##############                   CONSTRAINT KEYS                 ##############
    ###############################################################################

    def addForeignKey(self, tablename: str, fk_name: str, fk_column: str, ref_table: str, ref_col: str, printq=False):
        """
        Crea una foreign key

        :param tablename: Nombre de la tabla en la que vamos a crear la fk
        :param fk_name: Nombre de la fk: Normalmente columnaReferencia_FK
        :param fk_column: Nombre de la columna fk
        :param ref_table: Nombre de la tabla a la que apunta la fk
        :param ref_col: Nombre de columna referencia a la que apunta la fk
        :param printq: Print Query
        :return: None
        """

        self.executeQuery(f"ALTER TABLE {self.schema}.{tablename} ADD CONSTRAINT {fk_name} FOREIGN KEY ({fk_column}) "
                          f"REFERENCES {self.schema}.{ref_table}({ref_col})", printq)

    def addPrimaryKey(self, tablename: str, pk_name: str, pk_col_name, print_query: bool = False):

        self.executeQuery(f"ALTER TABLE {self.schema}.{tablename} ADD CONSTRAINT {pk_name} PRIMARY KEY ({pk_col_name})",
                          print_query)

    def addUniqueKey(self, tablename: str, uq_name: str, uq_col_name, print_query: bool = False):

        self.executeQuery(f"ALTER TABLE {self.schema}.{tablename} ADD CONSTRAINT {uq_name} UNIQUE ({uq_col_name})",
                          print_query)

    def addPrimaryKeyCompose(self, tablename: str, pk_col_name, print_query: bool = False):

        self.executeQuery(f"ALTER TABLE {self.schema}.{tablename} ADD UNIQUE ({pk_col_name})",
                          print_query)
