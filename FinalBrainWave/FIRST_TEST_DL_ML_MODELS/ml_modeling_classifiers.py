import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from tools.graphic_methods import Graphics

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, make_scorer, recall_score


class MetricsCalculations(Graphics):
    def __init__(self):
        Graphics.__init__(self)

        self.verbose: int = 2

    @staticmethod
    def plotEvaluationResults(df: pd.DataFrame, title, dirpathandname):
        # Calcular la cantidad de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos
        verdaderos_positivos = len(df[(df["aciertos"]) & (df["Valor Real"] == 1)])
        falsos_positivos = len(df[(~df["aciertos"]) & (df["Valor Real"] == 0)])
        verdaderos_negativos = len(df[(df["aciertos"]) & (df["Valor Real"] == 0)])
        falsos_negativos = len(df[(~df["aciertos"]) & (df["Valor Real"] == 1)])

        # Crear las etiquetas y los valores para el gráfico de barras
        labels = ['V.Pos', 'F.Pos', 'V.Neg', 'F.Neg']
        values = [verdaderos_positivos, falsos_positivos, verdaderos_negativos, falsos_negativos]

        # Crear el gráfico de barras
        colors = ['blue', 'red', 'blue', 'red']
        fig, ax = plt.subplots()
        bars = ax.bar(labels, values, color=colors)

        # Añadir etiquetas a los ejes y título al gráfico
        ax.set_ylabel('Cantidad')
        ax.set_title(f"{title}")

        # Función para añadir etiquetas encima de cada barra
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2, height,
                        f'{height} / {df.shape[0]} ({round(height / df.shape[0] * 100, 2)}%)',
                        ha='center', va='bottom')

        # Añadir etiquetas encima de cada barra
        autolabel(bars)

        # Mostrar el gráfico
        # plt.show()
        plt.savefig(f"{dirpathandname}")

        # -- Liberamos la memoria
        plt.close()

    @staticmethod
    def calculateClassifierAccuracy(df: pd.DataFrame):
        """
        Metodo para matchear los aciertos del modelo
        :param df: Dataframe
        :return: bool
        """

        # Si valor real es Si
        if int(df["Valor Real"]) == int(1):
            # Si el algoritmo ha dicho que si True, sino False
            if df['Valor Predicho'] == int(1):
                return True
            else:
                return False
        # Si valor real es es No
        else:
            # Si el algoritmo ha dicho que es No, True. Sino False
            if df['Valor Predicho'] == int(0):
                return True
            else:
                return False

    def evaluateClasification(self, df: pd.DataFrame, dirpath: str):

        reverse_probability: bool = True
        df["aciertos"] = df.apply(self.calculateClassifierAccuracy, axis=1)
        df_true = df[df["aciertos"] == True]
        # df_false = df[df["aciertos"] is False]
        accuracy = round(df_true.shape[0] / df.shape[0] * 100, 3)
        print(f"\nAccuracy: {accuracy}")

        # -- Sacamos los elementos que el modelo predice si y es si (VERDADEROS POSITIVOS)
        true_positive = df[(df["Valor Real"] == 1.0) & (df["aciertos"] == True)]

        # -- Sacamos los elementos que el modelo predice no y es no (VERDADEROS NEGATIVOS)
        true_negative = df[(df["Valor Real"] == 0.0) & (df["aciertos"] == True)]

        # -- Sacamos los elementos que el modelo predice no y en realidad es si (FALSOS NEGATIVOS)
        false_negative = df[(df["Valor Real"] == 1.0) & (df["aciertos"] == False)].shape[0]

        # -- Sacamos los elementos que el modelo predice si y en realidad es no (FALSOS POSITIVOS)
        false_positive = df[(df["Valor Real"] == 0.0) & (df["aciertos"] == False)].shape[0]

        # -- Sacamos la SENSIBILIDAD del modelo
        df_sensibility_totals: int = df[df["Valor Real"] == 1.0].shape[0]
        sensibility = round(true_positive.shape[0] / (df_sensibility_totals + false_negative) * 100, 3)
        print(f"Sensibility : {true_positive.shape[0]}/({df_sensibility_totals}+{false_negative})*100 = {sensibility}")

        # -- Sacamos la especificidad del modelo
        df_specificity_totals: int = df[df["Valor Real"] == 0.0].shape[0]
        specificity = round(true_negative.shape[0] / (df_specificity_totals + false_positive) * 100, 3)
        print(f"Specificity: {true_negative.shape[0]}/({df_specificity_totals}+{false_positive})*100 ={specificity}")

        result_dict: dict = {"accuracy": accuracy,
                             "sensibility": sensibility,
                             "specifity": specificity,
                             "A": {"TP": None, "FP": None, "TN": None, "FN": None},
                             "B": {"TP": None, "FP": None, "TN": None, "FN": None},
                             "C": {"TP": None, "FP": None, "TN": None, "FN": None},
                             "D": {"TP": None, "FP": None, "TN": None, "FN": None},
                             "ALL": {"TP": None, "FP": None, "TN": None, "FN": None},
                             }

        if reverse_probability:
            # Sacamos los A: Probabilidad de claudicar > 90
            df_A = df[df["Probabilidad Positiva"] >= 0.9]

            # Scamos los B, probabilidad de claudicar entre 75 y 90
            df_B = df[(0.75 < df["Probabilidad Positiva"]) & (df["Probabilidad Positiva"] <= 0.9)]

            # Scamos los C, probabilidad de claudicar entre 50 y 75
            df_C = df[(0.5 < df["Probabilidad Positiva"]) & (df["Probabilidad Positiva"] <= 0.75)]

            # Sacamos los D, probabilidad menor de 50
            df_D = df[df["Probabilidad Positiva"] <= 0.5]

            # print(df_A)
            self.plotEvaluationResults(df_A, "Grupo A: El modelo predice claudicación con probabilidad > 90%",
                                       f"{dirpath}/A.png")
            # print(df_B)
            self.plotEvaluationResults(df_B, "Grupo B: El modelo predice claudicación con probabilidad > 75% y =< 90%",
                                       f"{dirpath}/B.png")
            # print(df_C)
            self.plotEvaluationResults(df_C, "Grupo C: El modelo predice claudicación con probabilidad > 50% y =< 75%",
                                       f"{dirpath}/C.png")
            # print(df_D)
            self.plotEvaluationResults(df_D, "Grupo D: El modelo predice no claudicación", f"{dirpath}/D.png")

            self.plotEvaluationResults(df, "Conjunto de todas las predicciones", f"{dirpath}/ALL.png")

            result_dict: dict = {"metrics": {"accuracy": accuracy,
                                             "sensibility": sensibility,
                                             "specifity": specificity,
                                             "A": {"TP": df_A[df_A["Valor Real"] == 1.0].shape[0],
                                                   "FP": df_A[df_A["Valor Real"] == 0.0].shape[0]},
                                             "B": {"TP": df_B[df_B["Valor Real"] == 1.0].shape[0],
                                                   "FP": df_B[df_B["Valor Real"] == 0.0].shape[0]},
                                             "C": {"TP": df_C[df_C["Valor Real"] == 1.0].shape[0],
                                                   "FP": df_C[df_C["Valor Real"] == 0.0].shape[0]},
                                             "D": {"TN": df_D[df_D["Valor Real"] == 0.0].shape[0],
                                                   "FN": df_D[df_D["Valor Real"] == 1.0].shape[0]},
                                             "ALL": {"TP": true_positive.shape[0], "FP": false_positive,
                                                     "TN": true_negative.shape[0], "FN": false_negative},
                                             }}

        return accuracy, sensibility, specificity, result_dict


class DecisionTreeMethod(MetricsCalculations):
    def __init__(self, data_dict: dict, dirpath: str, show_info: bool = True, max_precission: bool = False):
        MetricsCalculations.__init__(self)

        # Definimos show_info para evaluar si queremos informacion detallada
        self.show_info = show_info
        self.metrics_path: str = dirpath

        # Definimos max_precission para evaluar si queremos que el grid se realize 1 vez o en cada vuelta
        self.max_precission = max_precission

        # Almacenamos los datos devueltos en objetos de clase:
        self.original_train: pd.DataFrame = data_dict["original_train"]
        self.original_test: pd.DataFrame = data_dict["original_test"]
        self.X_train: pd.DataFrame = data_dict["X_train"]
        self.y_train: pd.DataFrame = data_dict["y_train"]
        self.X_test: pd.DataFrame = data_dict["X_test"]
        self.y_test: pd.DataFrame = data_dict["y_test"]

        # Definimos el diccionario de resultados
        self.master_result_dict: dict = {}

    def decisionTreeApp(self):
        self.introPrint("Modeling 3: Aplicamos DecisionTree e iteramos hasta encontrar el mejor numero de predictores")

        self.subIntroPrint(f"3.1: ¿Hacer GridCV en cada iteración?: {'SI' if self.max_precission else 'NO'}")
        dt_best_hiperparameters: dict | None = None
        if not self.max_precission:
            dt_best_hiperparameters = self.decisionTreeGridSearchCV(self.X_train, self.y_train)

        self.subIntroPrint("3.2: Comenzamos las iteraciones")
        curr_cols = list(self.X_train.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            # Dependiendo de max_precission hacemos grid en cada iteracion
            if self.max_precission:
                dt_best_hiperparameters = self.decisionTreeGridSearchCV(self.X_train[curr_cols], self.y_train)

            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.decisionTreeInfo(self.X_train[curr_cols], self.y_train, self.X_test[curr_cols],
                                                         self.y_test, dt_best_hiperparameters)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 15:
                break

            iter_security_lock += 1

        # -- Devolvemos el diccionario con las metricas
        return self.master_result_dict

    def decisionTreeGridSearchCV(self, X_train: np.array, y_train: np.array, cv_number: int = 20):
        """
        max_depth: Este hiperparámetro controla la profundidad máxima del árbol de decisión.
        Si el árbol es muy profundo, es más probable que el modelo se ajuste demasiado a los datos de
        entrenamiento, lo que puede causar overfitting. Por lo tanto, se puede reducir el valor de max_depth para
        limitar la complejidad del modelo.

        min_samples_split: Este hiperparámetro especifica el número mínimo de muestras requeridas para dividir un
        nodo en el árbol de decisión. Si este valor es muy bajo, el modelo puede ajustarse demasiado a los datos de
        entrenamiento y causar overfitting. Por lo tanto, se puede aumentar el valor de min_samples_split para
        limitar la complejidad del modelo.

        min_samples_leaf: Este hiperparámetro especifica el número mínimo de muestras requeridas en una hoja del
        árbol de decisión. Si este valor es muy bajo, el modelo puede ajustarse demasiado a los datos de
        entrenamiento y causar overfitting. Por lo tanto, se puede aumentar el valor de min_samples_leaf para limitar
        la complejidad del modelo.
        """

        # Si buscamos sensibilidad
        # scorer = make_scorer(recall_score)

        # Si buscamos accuracy
        scorer = make_scorer(accuracy_score)

        param_grid = {
            'max_depth': range(1, 20),
            'min_samples_leaf': range(1, 20)
        }

        grid = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid=param_grid, cv=cv_number,
                            verbose=self.verbose, scoring=scorer)

        grid.fit(X_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        # Devolvemos el mejor conjunto de hiperparámetros que pasaremos a decisionTreeInfo
        return grid.best_params_

    def decisionTreeInfo(self, X_train, y_train, X_test, y_test, hiperparams):
        print(f'\n----------   Tenemos {X_train.shape[1]} columnas en training')
        print(f'----------   Tenemos {X_test.shape[1]} columnas en testing\n')

        dt = DecisionTreeClassifier(**hiperparams)
        dt.fit(X_train, y_train)

        y_pred_train = dt.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)

        y_pred_test = dt.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)

        print(f'La accuracy de train es : {acc_train}')
        print(f'La accuracy de test es : {acc_test}')

        # Seleccionamos las caracteristicas con los pesos mas importantes
        sfm = SelectFromModel(dt, threshold="mean")
        sfm.fit(X_train, y_train)

        # Pintamos las mejores caracteristicas
        selected_features = list(X_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        # Obtener las probabilidades de las predicciones en el conjunto de prueba
        probs = dt.predict_proba(X_test)

        # Crear un DataFrame que contenga el valor real, las dos probabilidades y el valor predicho
        df_result = pd.DataFrame(
            {'Valor Real': y_test, 'Probabilidad Negativa': probs[:, 0], 'Probabilidad Positiva': probs[:, 1],
             'Valor Predicho': y_pred_test})

        # Añado el numero de columnas al metrics_path y creo una nueva carpeta
        cols_metrics_path = f"{self.metrics_path}/{X_train.shape[1]}"
        os.makedirs(cols_metrics_path)

        accuracy, sensibility, specificity, metrics_dict = self.evaluateClasification(df_result, cols_metrics_path)

        # -- Ploteamos los pesos
        importances = dt.feature_importances_
        indices = np.argsort(importances)[::-1]
        self.plotRandomForestWeights(X_train, importances, indices, f"{cols_metrics_path}/PESOS.png",
                                     f"Acc: {accuracy}--Sens: {sensibility}--Spec: {specificity}")

        # Rellenamos el diccionario
        self.master_result_dict[f"{X_train.shape[1]}"]: dict = {}
        self.master_result_dict[f"{X_train.shape[1]}"]["columns"] = [z for z in X_train.columns]
        self.master_result_dict[f"{X_train.shape[1]}"]["best_params"] = hiperparams
        self.master_result_dict[f"{X_train.shape[1]}"]["metrics"] = metrics_dict

        return selected_features, excluded_features


class GradientBoostingMethod(MetricsCalculations):
    def __init__(self, data_dict: dict, dirpath: str, show_info: bool = True, max_precission: bool = False):
        MetricsCalculations.__init__(self)

        # -- Definimos show info para evaluar si queremos informacion detallada
        self.show_info = show_info
        self.metrics_path: str = dirpath

        # -- Definimos max_precission para evaluar si queremos que el grid se realize 1 vez o en cada vuelta
        self.max_precission = max_precission

        # -- Almacenamos los datos devueltos  en objetos de clase:
        self.original_train: pd.DataFrame = data_dict["original_train"]
        self.original_test: pd.DataFrame = data_dict["original_test"]
        self.X_train: pd.DataFrame = data_dict["X_train"]
        self.y_train: pd.DataFrame = data_dict["y_train"]
        self.X_test: pd.DataFrame = data_dict["X_test"]
        self.y_test: pd.DataFrame = data_dict["y_test"]

        # -- Definimos el diccionario de resultados
        self.master_result_dict: dict = {}

    def gradientBoostingApp(self):
        self.introPrint(
            "Modeling 3: Aplicamos GradientBoosting e iteramos hasta encontrar el mejor numero de predictores")

        self.subIntroPrint(f"3.1: ¿Hacer GridCV en cada iteración?: {'SI' if self.max_precission else 'NO'}")
        gb_best_hiperparameters: dict | None = None
        if not self.max_precission:
            gb_best_hiperparameters = self.gradientBoostingGridSearchCV(self.X_train, self.y_train)

        self.subIntroPrint("3.2: Comenzamos las iteraciones")
        curr_cols = list(self.X_train.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            # -- Dependiendo de max_precission hacemos grid en cada iteracion
            if self.max_precission:
                gb_best_hiperparameters = self.gradientBoostingGridSearchCV(self.X_train, self.y_train)

            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.gradientBoostingInfo(self.X_train[curr_cols], self.y_train,
                                                             self.X_test[curr_cols],
                                                             self.y_test, gb_best_hiperparameters)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 15:
                break

            iter_security_lock += 1

        # -- Devolvemos el diccionario con las metricas
        return self.master_result_dict

    def gradientBoostingGridSearchCV(self, X_train: np.array, y_train: np.array, cv_number: int = 5):
        """
        n_estimators: El número de árboles en el boosting.

        max_depth: Profundidad máxima de cada árbol.

        learning_rate: tasa de aprendizaje para el ajuste de pesos de los árboles.

        min_samples_split: el número mínimo de muestras necesarias para dividir un nodo interno.

        min_samples_leaf: El número mínimo de muestras necesarias para ser una hoja.

        max_features: El número de características que se consideran al buscar la mejor división. Puede ser un valor fijo o una fracción del total de características.

        """

        # Si buscamos sensibilidad
        # scorer = make_scorer(recall_score)

        # Si buscamos accuracy
        scorer = make_scorer(accuracy_score)

        param_grid = {
            'n_estimators': [100, 400, 800],
            'max_depth': [1, 4, 10, 15, 20],
            'learning_rate': [0.001, 0.01, 0.1]
        }

        grid = GridSearchCV(GradientBoostingClassifier(random_state=0),
                            param_grid=param_grid, cv=cv_number, verbose=self.verbose, scoring=scorer)

        grid.fit(X_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        # Devolvemos el mejor conjunto de hiperparámetros que pasaremos a gradientBoostingInfo
        return grid.best_params_

    def gradientBoostingInfo(self, X_train, y_train, X_test, y_test, hiperparams):
        print(f'\n----------   Tenemos {X_train.shape[1]} columnas en training')
        print(f'----------   Tenemos {X_test.shape[1]} columnas en testing\n')

        gb = GradientBoostingClassifier(**hiperparams)
        gb.fit(X_train, y_train)

        y_pred_train = gb.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)

        y_pred_test = gb.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)

        print(f'La accuracy de train es : {acc_train}')
        print(f'La accuracy de test es : {acc_test}')

        # -- Seleccionamos las caracteristicas con los pesos mas importantes
        sfm = SelectFromModel(gb, threshold="mean")
        sfm.fit(X_train, y_train)

        # Pintamos las mejores caracteristicas
        selected_features = list(X_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        # Obtener las probabilidades de las predicciones en el conjunto de prueba
        probs = gb.predict_proba(X_test)

        # Crear un DataFrame que contenga el valor real, las dos probabilidades y el valor predicho
        df_result = pd.DataFrame(
            {'Valor Real': y_test, 'Probabilidad Negativa': probs[:, 0], 'Probabilidad Positiva': probs[:, 1],
             'Valor Predicho': y_pred_test})

        # Añado el numero de columnas al metrics_path y creo una nueva carpeta
        cols_metrics_path = f"{self.metrics_path}/{X_train.shape[1]}"
        os.makedirs(cols_metrics_path)

        accuracy, sensibility, specificity, metrics_dict = self.evaluateClasification(df_result, cols_metrics_path)

        # -- Ploteamos los pesos
        importances = gb.feature_importances_
        indices = np.argsort(importances)[::-1]
        self.plotRandomForestWeights(X_train, importances, indices, f"{cols_metrics_path}/PESOS.png",
                                     f"Acc: {accuracy}--Sens: {sensibility}--Spec: {specificity}")

        # Rellenamos el diccionario
        self.master_result_dict[f"{X_train.shape[1]}"]: dict = {}
        self.master_result_dict[f"{X_train.shape[1]}"]["columns"] = [z for z in X_train.columns]
        self.master_result_dict[f"{X_train.shape[1]}"]["best_params"] = hiperparams
        self.master_result_dict[f"{X_train.shape[1]}"]["metrics"] = metrics_dict

        return selected_features, excluded_features


class LogisticRegressionMethod(MetricsCalculations):
    def __init__(self, data_dict: dict, dirpath: str, show_info: bool = True, max_precission: bool = False):
        MetricsCalculations.__init__(self)

        # -- Definimos show info para evaluar si queremos informacion detallada
        self.show_info = show_info
        self.metrics_path: str = dirpath
        # -- Definimos max_precission para evaluar si queremos que el grid se realize 1 vez o en cada vuelta
        self.max_precission = max_precission

        # -- Almacenamos los datos devueltos  en objetos de clase:
        self.original_train: pd.DataFrame = data_dict["original_train"]
        self.original_test: pd.DataFrame = data_dict["original_test"]
        self.X_train: pd.DataFrame = data_dict["X_train"]
        self.y_train: pd.DataFrame = data_dict["y_train"]
        self.X_test: pd.DataFrame = data_dict["X_test"]
        self.y_test: pd.DataFrame = data_dict["y_test"]

        # -- Definimos el diccionario de resultados
        self.master_result_dict: dict = {}

    def logisticRegressionApp(self):
        self.introPrint(
            "Modeling 2: Aplicamos Logistic Regression e iteramos hasta encontrar el mejor numero de predictores")

        self.subIntroPrint(f"2.1: ¿Hacer GridCV en cada iteración?: {'SI' if self.max_precission else 'NO'}")
        lr_best_hiperparameters: dict | None = None
        if not self.max_precission:
            lr_best_hiperparameters = self.logisticRegressionGridSearchCV(self.X_train, self.y_train)

        self.subIntroPrint("2.2: Comenzamos las iteraciones")
        curr_cols = list(self.X_train.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            # -- Dependiendo de max_precission hacemos grid en cada iteracion
            if self.max_precission:
                lr_best_hiperparameters = self.logisticRegressionGridSearchCV(self.X_train, self.y_train)

            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.logisticRegressionInfo(self.X_train[curr_cols], self.y_train,
                                                               self.X_test[curr_cols],
                                                               self.y_test, lr_best_hiperparameters)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 15:
                break

            iter_security_lock += 1

        # -- Devolvemos el diccionario con las metricas
        return self.master_result_dict

    def logisticRegressionGridSearchCV(self, X_train: np.array, y_train: np.array, cv_number: int = 20):
        """
        El coste C es el parámetro libre que permite controlar la complejidad del algoritmo,
        penalizando los errores que se comenten en clasificación. Este parámetro supone un compromiso entre la
        exactitud de la solución y la complejidad del algoritmo.

        - Cuanto mayor es C, más penalizamos los errores en clasificación y la frontera se ajusta mucho a los datos.
        Riesgo de overfitting pero con potencial menor error de clasificación.

        - Cuanto menor es C, menos penalizamos los errores en clasificación y tenderemos hacia modelos más sencillos
        (fronteras menos ajustadas, menor riesgo de overfitting pero potencialmente con más error de clasificación)

        NOTA: por defecto, C=1 en scikit-learn.
        """

        # Si buscamos sensibilidad
        # scorer = make_scorer(recall_score)

        # Si buscamos accuracy
        scorer = make_scorer(accuracy_score)

        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
        grid = GridSearchCV(LogisticRegression(random_state=0, fit_intercept=False, max_iter=1000),
                            param_grid=param_grid, cv=cv_number, verbose=self.verbose, scoring=scorer)

        grid.fit(X_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        # Devolvemos el mejor conjunto de hiperparámetros que pasaremos a LRinfo
        return grid.best_params_

    def logisticRegressionInfo(self, X_train, y_train, X_test, y_test, hiperparams):
        print(f'\n----------   Tenemos {X_train.shape[1]} columnas en training')
        print(f'----------   Tenemos {X_test.shape[1]} columnas en testing\n')

        lr = LogisticRegression(**hiperparams, random_state=0)
        lr.fit(X_train, y_train)

        y_pred_train = lr.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)

        y_pred_test = lr.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)

        print(f'La accuracy de train es : {acc_train}')
        print(f'La accuracy de test es : {acc_test}')

        # -- Seleccionamos las características con los pesos más importantes
        sfm = SelectFromModel(lr, threshold="mean")
        sfm.fit(X_train, y_train)

        # Pintamos las mejores características
        selected_features = list(X_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        # Obtener las probabilidades de las predicciones en el conjunto de prueba
        probs = lr.predict_proba(X_test)

        # Crear un DataFrame que contenga el valor real, las dos probabilidades y el valor predicho
        df_result = pd.DataFrame(
            {'Valor Real': y_test, 'Probabilidad Negativa': probs[:, 0], 'Probabilidad Positiva': probs[:, 1],
             'Valor Predicho': y_pred_test})

        # Añado el numero de columnas al metrics_path y creo una nueva carpeta
        cols_metrics_path = f"{self.metrics_path}/{X_train.shape[1]}"
        os.makedirs(cols_metrics_path)

        accuracy, sensibility, specificity, metrics_dict = self.evaluateClasification(df_result, cols_metrics_path)

        # Obtener los coeficientes de las características del modelo
        coeficientes = lr.coef_[0]

        # Calcular la importancia de las características utilizando el valor absoluto de los coeficientes
        importances = abs(coeficientes)

        # Ordenar las características por importancia descendente
        indices = np.argsort(importances)[::-1]
        self.plotRandomForestWeights(X_train, importances, indices, f"{cols_metrics_path}/PESOS.png",
                                     f"Acc: {accuracy}--Sens: {sensibility}--Spec: {specificity}")

        # Rellenamos el diccionario
        self.master_result_dict[f"{X_train.shape[1]}"]: dict = {}
        self.master_result_dict[f"{X_train.shape[1]}"]["columns"] = [z for z in X_train.columns]
        self.master_result_dict[f"{X_train.shape[1]}"]["best_params"] = hiperparams
        self.master_result_dict[f"{X_train.shape[1]}"]["metrics"] = metrics_dict

        return selected_features, excluded_features


class RandomForestMethod(MetricsCalculations):
    def __init__(self, data_dict: dict, dirpath: str, show_info: bool = True, max_precission: bool = False):
        MetricsCalculations.__init__(self)

        # -- Definimos show info para evaluar si queremos informacion detallada
        self.show_info = show_info
        self.metrics_path: str = dirpath

        # -- Definimos max_precission para evaluar si queremos que el grid se realize 1 vez o en cada vuelta
        self.max_precission = max_precission

        # -- Almacenamos los datos devueltos  en objetos de clase:
        self.original_train: pd.DataFrame = data_dict["original_train"]
        self.original_test: pd.DataFrame = data_dict["original_test"]
        self.X_train: pd.DataFrame = data_dict["X_train"]
        self.y_train: pd.DataFrame = data_dict["y_train"]
        self.X_test: pd.DataFrame = data_dict["X_test"]
        self.y_test: pd.DataFrame = data_dict["y_test"]

        # -- Definimos el diccionario de resultados
        self.master_result_dict: dict = {}

    def randomForestApp(self):
        self.introPrint("Modeling 2: Aplicamos RandomForest e iteramos hasta encontrar el mejor numero de predictores")

        self.subIntroPrint(f"2.1: ¿Hacer GridCV en cada iteración?: {'SI' if self.max_precission else 'NO'}")
        rf_best_hiperparameters: dict | None = None
        if not self.max_precission:
            rf_best_hiperparameters = self.randomForestGridSearchCV(self.X_train, self.y_train)

        self.subIntroPrint("2.2: Comenzamos las iteraciones")
        curr_cols = list(self.X_train.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            # -- Dependiendo de max_precission hacemos grid en cada iteracion
            if self.max_precission:
                rf_best_hiperparameters = self.randomForestGridSearchCV(self.X_train, self.y_train)

            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.randomForestInfo(self.X_train[curr_cols], self.y_train, self.X_test[curr_cols],
                                                         self.y_test, rf_best_hiperparameters)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 15:
                break

            iter_security_lock += 1

        # -- Devolvemos el diccionario con las metricas
        return self.master_result_dict

    def randomForestGridSearchCV(self, X_train: np.array, y_train: np.array, maxDepth=range(15, 16),
                                 cv_number: int = 20):
        """
        max_depth: Este hiperparámetro controla la profundidad máxima de los árboles de decisión en el Random Forest.
        Si los árboles son muy profundos, es más probable que el modelo se ajuste demasiado a los datos de
        entrenamiento, lo que puede causar overfitting. Por lo tanto, se puede reducir el valor de max_depth para
        limitar la complejidad del modelo.

        n_estimators: Este hiperparámetro controla el número de árboles en el Random Forest. Un mayor número de
        árboles puede mejorar la capacidad de generalización del modelo, pero también puede aumentar la complejidad
        del modelo y causar overfitting. Por lo tanto, puede ser útil probar diferentes valores de n_estimators para
        encontrar el equilibrio adecuado.

        min_samples_split: Este hiperparámetro especifica el número mínimo de muestras requeridas para dividir un
        nodo en el árbol de decisión. Si este valor es muy bajo, el modelo puede ajustarse demasiado a los datos de
        entrenamiento y causar overfitting. Por lo tanto, se puede aumentar el valor de min_samples_split para
        limitar la complejidad del modelo.

        min_samples_leaf: Este hiperparámetro especifica el número mínimo de muestras requeridas en una hoja del
        árbol de decisión. Si este valor es muy bajo, el modelo puede ajustarse demasiado a los datos de
        entrenamiento y causar overfitting. Por lo tanto, se puede aumentar el valor de min_samples_leaf para limitar
        la complejidad del modelo.

        max_features: Este hiperparámetro controla el número máximo de características que se consideran al dividir
        un nodo en el árbol de decisión. Si se consideran demasiadas características, el modelo puede ajustarse
        demasiado a los datos de entrenamiento y causar overfitting. Por lo tanto, se puede reducir el valor de
        max_features para limitar la complejidad del modelo.
                """

        # Si buscamos sensibilidad
        # scorer = make_scorer(recall_score)

        # Si buscamos accuracy
        scorer = make_scorer(accuracy_score)

        param_grid = {
            'n_estimators': [1500],
            'max_depth': maxDepth

        }
        grid = GridSearchCV(RandomForestClassifier(random_state=0, max_features='sqrt'),
                            param_grid=param_grid, cv=cv_number, verbose=self.verbose, scoring=scorer)

        grid.fit(X_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        # scores = np.array(grid.cv_results_['mean_test_score'])
        # self.plotAlphaValues(maxDepth, scores, f"AlphaRf{X_train.shape[1]}.png")

        # Devolvemos el mejor conjunto de hiperparametros que pasaremos a RFinfo
        return grid.best_params_

    def randomForestInfo(self, X_train, y_train, X_test, y_test, hiperparams):
        print(f'\n----------   Tenemos {X_train.shape[1]} columnas en training')
        print(f'----------   Tenemos {X_test.shape[1]} columnas en testing\n')

        rf = RandomForestClassifier(**hiperparams)
        rf.fit(X_train, y_train)

        y_pred_train = rf.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)

        y_pred_test = rf.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)

        print(f'La accuracy de train es : {acc_train}')
        print(f'La accuracy de test es : {acc_test}')

        # -- Seleccionamos las caracteristicas con los pesos mas importantes
        sfm = SelectFromModel(rf, threshold="mean")
        sfm.fit(X_train, y_train)

        # Pintamos las mejores caracteristicas
        selected_features = list(X_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        # Obtener las probabilidades de las predicciones en el conjunto de prueba
        probs = rf.predict_proba(X_test)

        # Crear un DataFrame que contenga el valor real, las dos probabilidades y el valor predicho
        df_result = pd.DataFrame(
            {'Valor Real': y_test, 'Probabilidad Negativa': probs[:, 0], 'Probabilidad Positiva': probs[:, 1],
             'Valor Predicho': y_pred_test})

        # Añado el numero de columnas al metrics_path y creo una nueva carpeta
        cols_metrics_path = f"{self.metrics_path}/{X_train.shape[1]}"
        os.makedirs(cols_metrics_path)

        accuracy, sensibility, specificity, metrics_dict = self.evaluateClasification(df_result, cols_metrics_path)

        # -- Ploteamos los pesos
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        self.plotRandomForestWeights(X_train, importances, indices, f"{cols_metrics_path}/PESOS.png",
                                     f"Acc: {accuracy}--Sens: {sensibility}--Spec: {specificity}")

        # Rellenamos el diccionario
        self.master_result_dict[f"{X_train.shape[1]}"]: dict = {}
        self.master_result_dict[f"{X_train.shape[1]}"]["columns"] = [z for z in X_train.columns]
        self.master_result_dict[f"{X_train.shape[1]}"]["best_params"] = hiperparams
        self.master_result_dict[f"{X_train.shape[1]}"]["metrics"] = metrics_dict

        return selected_features, excluded_features


class XGBoostMethod(MetricsCalculations):
    def __init__(self, data_dict: dict, dirpath: str, show_info: bool = True, max_precission: bool = False):
        MetricsCalculations.__init__(self)

        # -- Definimos show info para evaluar si queremos informacion detallada
        self.show_info = show_info
        self.metrics_path: str = dirpath

        # -- Definimos max_precission para evaluar si queremos que el grid se realize 1 vez o en cada vuelta
        self.max_precission = max_precission

        # -- Almacenamos los datos devueltos  en objetos de clase:
        self.original_train: pd.DataFrame = data_dict["original_train"]
        self.original_test: pd.DataFrame = data_dict["original_test"]
        self.X_train: pd.DataFrame = data_dict["X_train"]
        self.y_train: pd.DataFrame = data_dict["y_train"]
        self.X_test: pd.DataFrame = data_dict["X_test"]
        self.y_test: pd.DataFrame = data_dict["y_test"]

        # -- Definimos el diccionario de resultados
        self.master_result_dict: dict = {}

    def xgboostApp(self):
        self.introPrint("Modeling 3: Aplicamos XGBoost e iteramos hasta encontrar el mejor numero de predictores")

        self.subIntroPrint(f"3.1: ¿Hacer GridCV en cada iteración?: {'SI' if self.max_precission else 'NO'}")
        xgboost_best_hiperparameters: dict | None = None
        if not self.max_precission:
            xgboost_best_hiperparameters = self.xgboostGridSearchCV(self.X_train, self.y_train)

        self.subIntroPrint("3.2: Comenzamos las iteraciones")
        curr_cols = list(self.X_train.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            if self.max_precission:
                xgboost_best_hiperparameters = self.xgboostGridSearchCV(self.X_train[curr_cols], self.y_train)

            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.xgboostInfo(self.X_train[curr_cols], self.y_train, self.X_test[curr_cols],
                                                    self.y_test, xgboost_best_hiperparameters)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 15:
                break

            iter_security_lock += 1

        # -- Devolvemos el diccionario con las metricas
        return self.master_result_dict

    def xgboostGridSearchCV(self, X_train: np.array, y_train: np.array, cv_number: int = 5):
        scorer = make_scorer(accuracy_score)

        '''param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.001, 0.01, 0.1],
            'gamma': [0, 0.01, 0.1],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'subsample': [0.5, 0.7, 0.9],
            'reg_alpha': [0, 0.1, 1, 10],
            'reg_lambda': [0, 0.1, 1, 10],
            'min_child_weight': [1, 3, 5]
        }'''

        param_grid = {
            'n_estimators': [100, 200, 400, 800, 1500],
            'max_depth': [1, 4, 7, 10, 13, 16, 19],
            'learning_rate': [0.001, 0.01, 0.1]
        }

        # Modelo base de XGBoost
        xgb = XGBClassifier(objective='binary:logistic', random_state=0)

        # Realizar la búsqueda de hiperparámetros
        grid = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=cv_number, scoring=scorer, verbose=self.verbose)
        grid.fit(X_train, y_train)

        # Mostrar los resultados de la búsqueda de hiperparámetros
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        # Devolver los mejores hiperparámetros encontrados
        return grid.best_params_

    def xgboostInfo(self, X_train, y_train, X_test, y_test, hiperparams):
        print(f'\n----------   Tenemos {X_train.shape[1]} columnas en training')
        print(f'----------   Tenemos {X_test.shape[1]} columnas en testing\n')

        # -- Creamos un clasificador XGBoost con los hiperparámetros seleccionados
        xgb_clf = XGBClassifier(**hiperparams, random_state=42)

        # -- Entrenamos el modelo
        xgb_clf.fit(X_train, y_train)

        # -- Realizamos las predicciones sobre los conjuntos de train y test
        y_pred_train = xgb_clf.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)

        y_pred_test = xgb_clf.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)

        print(f'La accuracy de train es : {acc_train}')
        print(f'La accuracy de test es : {acc_test}')

        # -- Seleccionamos las características con los pesos más importantes
        sfm = SelectFromModel(xgb_clf, threshold="mean")
        sfm.fit(X_train, y_train)

        # -- Obtenemos las características seleccionadas
        selected_features = list(X_train.columns[sfm.get_support()])

        # -- Obtenemos las características excluidas
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        # -- Obtenemos las probabilidades de las predicciones en el conjunto de prueba
        probs = xgb_clf.predict_proba(X_test)

        # -- Creamos un DataFrame que contenga el valor real, las dos probabilidades y el valor predicho
        df_result = pd.DataFrame(
            {'Valor Real': y_test, 'Probabilidad Negativa': probs[:, 0], 'Probabilidad Positiva': probs[:, 1],
             'Valor Predicho': y_pred_test})

        # Añado el numero de columnas al metrics_path y creo una nueva carpeta
        cols_metrics_path = f"{self.metrics_path}/{X_train.shape[1]}"
        os.makedirs(cols_metrics_path)

        accuracy, sensibility, specificity, metrics_dict = self.evaluateClasification(df_result, cols_metrics_path)

        # -- Ploteamos los pesos
        importances = xgb_clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        self.plotRandomForestWeights(X_train, importances, indices, f"{cols_metrics_path}/PESOS.png",
                                     f"Acc: {accuracy}--Sens: {sensibility}--Spec: {specificity}")

        # Rellenamos el diccionario
        self.master_result_dict[f"{X_train.shape[1]}"]: dict = {}
        self.master_result_dict[f"{X_train.shape[1]}"]["columns"] = [z for z in X_train.columns]
        self.master_result_dict[f"{X_train.shape[1]}"]["best_params"] = hiperparams
        self.master_result_dict[f"{X_train.shape[1]}"]["metrics"] = metrics_dict

        return selected_features, excluded_features


class KNNMethod(MetricsCalculations):
    def __init__(self, data_dict: dict, dirpath: str, show_info: bool = True, max_precission: bool = False):
        MetricsCalculations.__init__(self)

        self.show_info = show_info
        self.metrics_path: str = dirpath
        self.max_precission = max_precission

        self.original_train: pd.DataFrame = data_dict["original_train"]
        self.original_test: pd.DataFrame = data_dict["original_test"]
        self.X_train: pd.DataFrame = data_dict["X_train"]
        self.y_train: pd.DataFrame = data_dict["y_train"]
        self.X_test: pd.DataFrame = data_dict["X_test"]
        self.y_test: pd.DataFrame = data_dict["y_test"]

        self.master_result_dict: dict = {}

    def knnApp(self):
        self.introPrint("Modeling 2: Aplicamos KNN e iteramos hasta encontrar el mejor número de vecinos")

        self.subIntroPrint(f"2.1: ¿Hacer GridCV en cada iteración?: {'SI' if self.max_precission else 'NO'}")
        knn_best_hiperparameters: dict | None = None
        if not self.max_precission:
            knn_best_hiperparameters = self.knnGridSearchCV(self.X_train, self.y_train)

        self.subIntroPrint("2.2: Comenzamos las iteraciones")
        curr_cols = list(self.X_train.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            if self.max_precission:
                knn_best_hiperparameters = self.knnGridSearchCV(self.X_train, self.y_train)

            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.knnInfo(self.X_train[curr_cols], self.y_train, self.X_test[curr_cols],
                                                self.y_test, knn_best_hiperparameters)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 15:
                break

            iter_security_lock += 1

        return self.master_result_dict

    def knnGridSearchCV(self, X_train: np.array, y_train: np.array, cv_number: int = 10):
        scorer = make_scorer(accuracy_score)

        param_grid = {
            'n_neighbors': list(range(1, 31)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=cv_number, verbose=self.verbose,
                            scoring=scorer)

        grid.fit(X_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        return grid.best_params_

    def knnInfo(self, X_train, y_train, X_test, y_test, hiperparams):
        print(f'\n----------   Tenemos {X_train.shape[1]} columnas en training')

        print(f'----------   Tenemos {X_test.shape[1]} columnas en testing\n')

        knn = KNeighborsClassifier(**hiperparams)
        knn.fit(X_train, y_train)

        y_pred_train = knn.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)

        y_pred_test = knn.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)

        print(f'La accuracy de train es : {acc_train}')
        print(f'La accuracy de test es : {acc_test}')

        sfm = SelectFromModel(knn, threshold="mean")
        sfm.fit(X_train, y_train)

        selected_features = list(X_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        probs = knn.predict_proba(X_test)

        df_result = pd.DataFrame(
            {'Valor Real': y_test, 'Probabilidad Negativa': probs[:, 0], 'Probabilidad Positiva': probs[:, 1],
             'Valor Predicho': y_pred_test})

        cols_metrics_path = f"{self.metrics_path}/{X_train.shape[1]}"
        os.makedirs(cols_metrics_path)

        accuracy, sensibility, specificity, metrics_dict = self.evaluateClasification(df_result, cols_metrics_path)

        # Selección de características usando SelectKBest y mutual_info_classif
        selector = SelectKBest(mutual_info_classif, k="all")
        selector.fit(X_train, y_train)

        # Obtener las puntuaciones de importancia de características
        importances = selector.scores_
        indices = np.argsort(importances)[::-1]

        self.plotRandomForestWeights(X_train, importances, indices, f"{cols_metrics_path}/PESOS.png",
                                     f"Acc: {accuracy}--Sens: {sensibility}--Spec: {specificity}")

        self.master_result_dict[f"{X_train.shape[1]}"]: dict = {}
        self.master_result_dict[f"{X_train.shape[1]}"]["columns"] = [z for z in X_train.columns]
        self.master_result_dict[f"{X_train.shape[1]}"]["best_params"] = hiperparams
        self.master_result_dict[f"{X_train.shape[1]}"]["metrics"] = metrics_dict

        return selected_features, excluded_features


class MlModelingClassifier(Graphics):
    def __init__(self, data_dict: dict, model: str, dirpath: str, show_info: bool = True, max_precission: bool = False):
        Graphics.__init__(self)
        self.show_info = show_info
        self.metrics_path: str = dirpath

        # -- Almacenamos el modelo escogido en self.model
        self.model: str = model

        self.data_dict: dict = data_dict
        self.max_precission: bool = max_precission

        # -- Almacenamos los datos devueltos  en objetos de clase:
        self.original_train: pd.DataFrame = data_dict["original_train"]
        self.original_test: pd.DataFrame = data_dict["original_test"]
        self.X_train: pd.DataFrame = data_dict["X_train"]
        self.y_train: pd.DataFrame = data_dict["y_train"]
        self.X_test: pd.DataFrame = data_dict["X_test"]
        self.y_test: pd.DataFrame = data_dict["y_test"]

    def run(self):
        match self.model:
            case "DecisionTreeClassifier":
                return DecisionTreeMethod(self.data_dict, self.metrics_path, self.show_info,
                                          self.max_precission).decisionTreeApp()

            case "GradientBoostingClassifier":
                return GradientBoostingMethod(self.data_dict, self.metrics_path, self.show_info,
                                              self.max_precission).gradientBoostingApp()

            case "LogisticRegression":
                return LogisticRegressionMethod(self.data_dict, self.metrics_path, self.show_info,
                                                self.max_precission).logisticRegressionApp()

            case "RandomForestClassifier":
                return RandomForestMethod(self.data_dict, self.metrics_path, self.show_info,
                                          self.max_precission).randomForestApp()

            case "XGBoostClassifier":
                return XGBoostMethod(self.data_dict, self.metrics_path, self.show_info,
                                     self.max_precission).xgboostApp()

            case "KNNMethod":
                return KNNMethod(self.data_dict, self.metrics_path, self.show_info,
                                 self.max_precission).knnApp()

            case _:
                return
