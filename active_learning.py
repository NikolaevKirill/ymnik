from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.spatial import cKDTree
from tqdm import tqdm


class ActiveLearning:
    """
    Класс, для обучения классификатора режимов работы объекта с помощью активного обучения.
    """

    def __init__(self):
        """
        Инициализация параметров класса: модель объекта, классификатор,
        диапазоны объектных параметров.
        """
        self.model_of_object = None  # model of object
        self.model = None  # self.model + Scaler
        self.bounds_a = None  # bounds of parameters alpha-modes
        self.bounds_b = None  # bounds of parameters beta-modes
        self.scaler_a = MinMaxScaler()  # scaler of parameters alpha-modes
        self.scaler_b = MinMaxScaler()  # scaler of parameters alpha-modes
        self.scaler_model = MinMaxScaler()  # scaler of features modes

        self.x_a = None  # object parameters of alpha-modes
        self.x_b = None  # object parameters of beta-modes

        self.v_a = None  # features of alpha-modes
        self.v_b = None  # features of beta-modes

        ###

        self.clf = None  # classifier
        self.pf_degree = 4  # degree of polinomial features
        self.coef = None  # coefficients of logreg
        self.intercept = None  # intercept
        self.s_r = None  # resistance of alpha-modes during learning

        self.start_time = None  # time of start learning
        self.end_time = None  # time of end learning

    def initialize(self, model, bounds_a, bounds_b, n_init=100):

        """
        Функция обновляет параметры модели к моменту до начала обучения.

        Args:
            model (func): Имитационная модель объекта со встроенным замером, например, провоимости.

            bounds_a (list or ndarray of shape (n_obj_params, 2)): Массив с диапазонами объектных
            параметров для альфа-режимов.

            bounds_b (list or ndarray of shape (n_obj_params, 2)): Массив с диапазонами объектных
            параметров для бета-режимов.

            n_init (int): Количество сгенерированных до обучения режимов каждого класса
        """
        self.model_of_object = model
        self.bounds_a = bounds_a
        self.bounds_b = bounds_b
        self.scaler_a.fit(np.array(bounds_a).T)
        self.scaler_b.fit(np.array(bounds_b).T)

        np.random.seed(43)
        x_a = np.random.uniform(
            bounds_a[:, 0], bounds_a[:, 1], size=(n_init, len(bounds_a))
        )
        x_b = np.random.uniform(
            bounds_b[:, 0], bounds_b[:, 1], size=(n_init, len(bounds_b))
        )

        self.x_a = x_a
        self.x_b = x_b

        v_b = model(x_b)

        self.scaler_model.fit(v_b)
        self.model = lambda x: self.scaler_model.transform(self.model_of_object(x))

        v_a = self.model(x_a)
        v_b = self.model(x_b)

        self.v_a = v_a
        self.v_b = v_b
        v, y = self.__make_dataset(v_b, v_a)

        clf = Pipeline(
            [
                (
                    "pf",
                    PolynomialFeatures(
                        self.pf_degree,
                    ),
                ),
                ("scaler2", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=100000,
                        class_weight={1: 1, 0: 1},
                        C=5000,
                        warm_start=True,
                    ),
                ),
            ]
        )

        clf, _ = self.__get_selective_clf(clf, v, y)

        self.clf = clf

    def __make_dataset(self, xb, xa):
        """
        Функция формирует датасет и метки классов для обучения модели.

        Args:
            Xb (ndarray of shape (n_samples_b, n_features)): массив, содержащий величины,
            используемые в качестве признаков, соответсвующих бета-режимам.

            Xa (ndarray of shape (n_samples_a, n_features)): массив, содержащий величины,
            используемые в качестве признаков, соответсвующих альфа-режимам.

        Returns:
            X (ndarray of shape (n_samples_b+n_samples_a, n_features)): массив, содержащий величины,
            используемые в качестве признаков для обучения модели.

            y (ndarray of shape (n_samples_b+n_samples_a,)): массив,
            содержащий метки классов для соответствующих объектов.
        """
        x = np.r_[xb, xa]
        y = np.zeros(len(x))
        y[: len(xb)] = 1

        return x, y.astype(int)

    def __get_selective_clf(self, clf, x, y, max_iter=10):
        """
        Функция подбирает вес класса бета-режимов таким образом, чтобы выполнялось
        условие селективности: все бета-режимы (положительный класс) классифицируются верно.

        Args:
            clf (class sklearn.linear_model.LogisticRegression): предварительно обученный
            классификатор, для которого необходимо подобрать вес положительного класса.

            X (ndarray of shape (n_samples, n_features)): массив с признаками объектов для обучения.

            y (n_samples,): массив с метками классов объектов для обучения.

            max_iter (int), default=10: количество попыток уточнения веса положительного класса.

        Returns:
            clf (class sklearn.linear_model.LogisticRegression): обученный классификатор,
            с подобранным весом положительного класса.

            i (int): индекс минимального подходящего веса из списка [0.1, 1, 2, 5, 10]
        """
        class_weights = [0.1, 1, 2, 5, 10]

        precs, recs = [], []
        for class_weight in class_weights:
            clf[-1].set_params(**{"class_weight": {0: 1, 1: class_weight}})
            clf.fit(x, y)
            y_pred = clf.predict(x)
            prec = precision_score(y, y_pred, pos_label=1)
            rec = recall_score(y, y_pred, pos_label=1)
            precs.append(prec)
            recs.append(rec)

        for i, rec in enumerate(recs):
            if rec == 1:
                break

        max_class_weight = class_weights[i]
        min_class_weight = class_weights[i - 1]

        for _ in range(max_iter):
            cur_class_weight = np.mean([max_class_weight, min_class_weight])
            clf[-1].set_params(**{"class_weight": {0: 1, 1: cur_class_weight}})
            clf.fit(x, y)
            y_pred = clf.predict(x)
            rec = recall_score(y, y_pred, pos_label=1)
            if rec == 1:
                max_class_weight = cur_class_weight
            else:
                min_class_weight = cur_class_weight

        clf[-1].set_params(**{"class_weight": {0: 1, 1: max_class_weight}})
        clf.fit(x, y)

        return clf, i

    def __decimation(self, points, radius):
        """
        Функция прореживает области вокруг точек по заданному радиусу.
        Функция возвращает точки, оставшиеся после прореживания.

        Args:
            points (ndarray of shape(n_samples, n_obj_params)): Точки, которые необходимо проредить.

            radius (float): Минимальное расстояние до ближайших точек в прореженном массиве.

        Returns:
            ind_of_point (ndarray of shape(n_samples_selected, )): Массив индексов точек,
            которые остались после прореживания.
        """
        # Массив points преобразуется в индексированный массив (N,3)
        ind_points = np.arange(0, len(points), 1)
        points = np.concatenate(
            (ind_points.reshape(len(ind_points), 1), points), axis=1
        )
        len_p = len(points)
        ind_of_point = points[0, 0]
        while len_p != 0:
            point = points[0, 1:]  # текущая точка, вокруг которой все удаляется
            ind_of_point = np.hstack((ind_of_point, points[0, 0]))
            tree = cKDTree(points[:, 1:])  # формирование дерева
            results = tree.query_ball_point(
                point, radius
            )  # вокруг точки удаляются все точки, а также и она сама
            points = np.delete(points, results, axis=0)
            len_p = len(points)

        return ind_of_point[1:]

    def __selection(self, param, x, radius=1.6):
        """
        Функция возвращает прореженную область точек, подаваемых на вход функции.

        Args:
            param (ndarray of shape(n_samples, n_obj_params)): Объектные параметры точек,
            которые необходимо проредить.

            X (ndarray of shape(n_samples, n_features)): Наблюдаемые параметры точек,
            которые необходимо проредить.

            radius (float), default=1.6: Минимальное расстояние до ближайших точек
            в прореженном массиве.

        Returns:
            param (ndarray of shape(n_param_n_samples_selected, n_obj_params)): Объектные параметры
            точек, которые подверглись прореживанию.

            X (ndarray of shape(n_param_n_samples_selected, n_features)): Наблюдаемые параметры
            точек, которые подверглись прореживанию.
        """
        ind_dec = np.intp(self.__decimation(x, radius))
        x = x[ind_dec, :]
        param = param[ind_dec, :]

        return param, x

    def __get_scores_dist(self, x, k=2):
        """
        Функция возвращет отмасштабированное расстояние от каждого объекта
        до его "ближайшего соседа".

        Args:
            X (ndarray of shape (n_samples, n_features)): массив с признаками для обучения.

            k (int), default=2: Количество ближайших соседей, до которых рассчитывается расстояние,
            плюс сама точка (расстояние до неё 0).

        Returns:
            result (ndarray of shape(n_samples,)): массив с отмасштабированным расстоянием
            от каждого объекта до его "ближайшего соседа".
        """
        tree = cKDTree(x)
        dists, _ = tree.query(x, k=k)

        return MinMaxScaler().fit_transform(dists.mean(axis=1)[:, None]).flatten()

    def __make_beta(self, modes, ab=10, shift=0.01):
        """
        Функция принимает моду, величину нерасчётного параметра и смещение параметров.
        Функция возвращает параметры a и b бета-распределения.

        Args:
            modes (ndarray of shape (n_samples,) or int): Массив с модами распределений или мода.

            ab (int or float), default=10: Нерасчётный параметр. Его повышение уменьшает дисперсию.

            shift (int or float), default=0.01: Сдвиг параметра.
            Расчётный параметр сдвигается на эту величину, вследствие чего,
            при достаточно близком значении моды к границе области определения
            реальная мода "перейдет" на саму границу.

        Returns:
            a (float): Параметр a бета-распределения.

            b (float): Параметр b бета-распределения.
        """
        cond = modes > 0.5
        a = np.zeros(len(modes))
        b = np.zeros(len(modes))

        a[cond] = ab
        b[cond] = (a[cond] + modes[cond] * (2 - a[cond]) - 1) / modes[cond] - shift

        b[~cond] = ab
        a[~cond] = (-b[~cond] * modes[~cond] + 2 * modes[~cond] - 1) / (
            modes[~cond] - 1
        ) - shift

        return a, b

    def __sample_dots(self, dots_inp, scaler, n=5, **kwargs):
        """
        Функция сэмплирует новые точки путём генерации параметров согласно бета-распределению
        вокруг имеющихся точек. Функция возвращает насэмпленные точки.

        Args:
            dots_inp (ndarray of shape (n_samples, n_obj_params)): Массив с объектными параметрами
            имеющихся точек.

            scaler (class sklearn.preprocessing.MinMaxScaler): Предобученный MinMaxScaler
            для нормализации объектных параметров.
            Необходим поскольку на выходе бета-распределения генерируются значения от 0 до 1.
            Следовательно изначальные параметры, от которых происходит сэмплирование,
            нужно привести к диапазону от 0 до 1.

            n (int), default=5: Количество сэмплируемых точек вокруг изначальной точки.

        Returns:
            result (ndarray of shape(n_samples*n, n_obj_params): Массив с насэмпленными точками.
        """
        # семплинг делается отдельно по осям в объектном пространстве
        result = np.zeros((len(dots_inp) * n, dots_inp.shape[-1]))
        dots = scaler.transform(dots_inp)
        for i in range(dots.shape[-1]):
            modes = dots[:, i]
            a, b = self.__make_beta(modes, **kwargs)
            result[:, i] = stats.beta.rvs(a, b, size=(n, len(modes))).flatten()

        return scaler.inverse_transform(result)

    def step(
        self,
        n_for_sample=100,
        n_samples=5,
        radius=0.02,
        ab_start=80,
    ):
        """
        Функция обучения классификатора по принципу активного обучения. Выполняется один шаг обучения.

        Args:
            n_for_sample (int), default=100: Количество точек, от которых происходит сэмплирование.

            n_samples (int), default=5: Количество мэплируемых точек.

            radius (float), default=0.02: Минимальное расстояние до ближайших точек
            в прореженном массиве.

            ab_start (int or float), default=80: Нерасчётный параметр.
            Его повышение уменьшает дисперсию.
        """
        start_t = str(datetime.now()).split()
        start_time = (start_t[1][:8], start_t[0])
        self.start_time = start_time

        s_r = []

        x_a = self.x_a
        x_b = self.x_b
        v_a = self.v_a
        v_b = self.v_b

        scaler_a = self.scaler_a
        scaler_b = self.scaler_b

        model = self.model
        clf = self.clf

        s_r.append(x_a[:, 0].sum())

        ab = ab_start

        cond = clf.predict(v_a) != 0
        v_a = v_a[~cond]
        x_a = x_a[~cond]

        x_a, v_a = self.__selection(x_a, v_a, radius=radius)
        try:
            proba_a = clf.decision_function(v_a)
        except AttributeError:
            proba_a = clf.predict_proba(v_a)[:, 1]

        proba_a = MinMaxScaler().fit_transform(proba_a[:, None]).flatten()
        proba_a += (
            self.__get_scores_dist(v_a) * 0.3
        )  # максимальные альфы ближе к границе, поэтому прибавляем
        best_a = np.argsort(-proba_a)[: n_for_sample * 1]
        best_a = np.random.choice(best_a, size=n_for_sample)

        x_to_sample_a = x_a[best_a]
        x_sampled_a = self.__sample_dots(x_to_sample_a, scaler_a, n=n_samples, ab=ab)
        v_sampled_a = model(x_sampled_a)

        x_a = np.r_[x_a, x_sampled_a]
        v_a = np.r_[v_a, v_sampled_a]

        #####

        x_b, v_b = self.__selection(x_b, v_b, radius=radius)
        try:
            proba_b = clf.decision_function(v_b)
        except AttributeError:
            proba_b = clf.predict_proba(v_b)[:, 1]

        proba_b = MinMaxScaler().fit_transform(proba_b[:, None]).flatten()
        proba_b -= (
            self.__get_scores_dist(v_b) * 0.3
        )  # минимальные беты ближе к границе, поэтому вычитаем
        best_b = np.argsort(-proba_b)[-n_for_sample * 1 :]
        best_b = np.random.choice(best_b, size=n_for_sample)

        x_to_sample_b = x_b[best_b]
        x_sampled_b = self.__sample_dots(x_to_sample_b, scaler_b, n=n_samples, ab=ab)
        v_sampled_b = model(x_sampled_b)

        x_b = np.r_[x_b, x_sampled_b]
        v_b = np.r_[v_b, v_sampled_b]

        v, y = self.__make_dataset(v_b, v_a)
        clf, _ = self.__get_selective_clf(clf, v, y)

        cond = clf.predict(v_a) != 0
        v_a = v_a[~cond]
        x_a = x_a[~cond]

        x_a, v_a = self.__selection(x_a, v_a, radius=radius)
        x_b, v_b = self.__selection(x_b, v_b, radius=radius)

        #####

        self.clf = clf
        self.coef = clf[-1].coef_
        self.intercept = clf[-1].intercept_

        self.x_a = x_a
        self.x_b = x_b
        self.v_a = v_a
        self.v_b = v_b
        self.s_r = s_r

        # Запись времени окончания расчёта
        end_t = str(datetime.now()).split()
        end_time = (end_t[1][:8], end_t[0])
        self.end_time = end_time

    def learning(
        self,
        model,
        bounds_a,
        bounds_b,
        n_for_sample=100,
        n_samples=5,
        radius=0.02,
        ab_start=80,
        maxiter=200,
    ):
        """
        Функция обучения классификатора по принципу активного обучения.

        Args:
            model (func): Имитационная модель объекта со встроенным замером, например, провоимости.

            bounds_a (list or ndarray of shape (n_obj_params, 2)): Массив с диапазонами объектных
            параметров для альфа-режимов.

            bounds_b (list or ndarray of shape (n_obj_params, 2)): Массив с диапазонами объектных
            параметров для бета-режимов.

            n_for_sample (int), default=100: Количество точек, от которых происходит сэмплирование.

            n_samples (int), default=5: Количество мэплируемых точек.

            radius (float), default=0.02: Минимальное расстояние до ближайших точек
            в прореженном массиве.

            ab_start (int or float), default=80: Нерасчётный параметр.
            Его повышение уменьшает дисперсию.

            maxiter (int), default=200: Максимальное количество итераций.
        """
        # Запись времени загрузки данных
        start_t = str(datetime.now()).split()
        start_time = (start_t[1][:8], start_t[0])

        self.initialize(model, bounds_a, bounds_b)

        for _ in tqdm(range(maxiter)):
            self.step(
                n_for_sample=n_for_sample,
                n_samples=n_samples,
                radius=radius,
                ab_start=ab_start,
            )

        # Запись времени окончания расчёта
        end_t = str(datetime.now()).split()
        end_time = (end_t[1][:8], end_t[0])

        self.start_time = start_time
        self.end_time = end_time

    def plot_clf(self, clf=None, ranges=None, ax=None, scaler=lambda x: x):
        """
        Функция отображает области, соответствующие классам (альфа- и бета-режимам).

        Args:
            clf (class sklearn.linear_model.LogisticRegression): обученный классификатор.

            ranges (list or ndarray of shape(2, 2)), default=[[-0.5,1.5],[-0.5,1.5]]: Список или
            массив, содержащий граничные значения признаков, в пределах которых будут отображены
            области классов.

            ax (class matplotlib.axes.Axes), default=None: Передаётся в случае отображения
            нескольких изображений в plt.subplots. В ином случае - None.

            scaler (class sklearn.preprocessing.MinMaxScaler), default=lambdax:x: Задаётся в
            случае необходимости масштабирования признаков.
        """
        if clf is None:
            clf = self.clf

        v_a = self.v_a
        v_b = self.v_b

        x_min, x_max = v_b[:, 0].min() - 0.5, v_b[:, 0].max() + 0.5
        y_min, y_max = v_b[:, 1].min() - 0.5, v_b[:, 1].max() + 0.5

        lsx1 = np.linspace(x_min, x_max, 400)
        lsx2 = np.linspace(y_min, y_max, 400)
        xx0, xx1 = np.meshgrid(lsx1, lsx2)
        yy = clf.predict(np.c_[xx0.flatten(), xx1.flatten()]).reshape(xx0.shape)

        x = scaler(np.c_[np.c_[xx0.flatten(), xx1.flatten()]])
        xx0, xx1 = [i.reshape(yy.shape) for i in x.T]

        if ax == None:
            plt.contourf(xx0, xx1, yy, zorder=10, alpha=0.3, cmap="bwr_r")
        else:
            ax.contourf(xx0, xx1, yy, zorder=10, alpha=0.3, cmap="bwr_r")

        plt.scatter(*v_b.T, edgecolor="k")
        plt.scatter(*v_a.T, edgecolor="k")
        plt.grid()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

    def create_report(self):
        """
        Функия создаёт отчёт, содержащий формулы для вычисления polynomial features,
        значения весов, данные о модели, выражение для разграничивающей кривой,
        время загрузки данных и время формирования отчёта. Отчёт выполнен в виде строки,
        которую необходимо записать в файл.

        Args:
            filename (str), default="report": Название отчёта.
        Returns:
            report (str): Строка с содержанием отчёта.
        """

        report = ""
        name_of_model = str(self.model_of_object).split()[1]
        bounds_a = self.bounds_a
        bounds_b = self.bounds_b

        clf = self.clf
        coefs = self.coef
        intercept = self.intercept
        n_poly_f = clf[0].n_output_features_
        mean = clf[1].mean_
        st_dev = np.sqrt(clf[1].var_)

        start_time = self.start_time
        end_time = self.end_time

        # Получение выражений для вычисления polynomial features

        columns = clf[0].powers_.shape[1]
        features = [f"f{i}" for i in range(columns)]
        xs = [f"x{i}" for i in range(n_poly_f)]
        ms = [f"m{i}" for i in range(n_poly_f)]
        sds = [f"std{i}" for i in range(n_poly_f)]
        poly_features = []

        # Для масштабирования
        means = list(zip(ms, mean.tolist()))
        stdevs = list(zip(sds, st_dev.tolist()))

        # Для вывода признаков пользователя
        fs = features[0]
        for elem in features[1:]:
            fs += ", " + elem

        # Выражения для признаков разграничивающей модели
        for j, raw in enumerate(clf[0].powers_):
            equation = "("
            for i, degree in enumerate(raw):
                equation += features[i] + f"^{degree}"
                if i != (columns - 1):
                    equation += "+"
                else:
                    equation += (
                        f"-{ms[j]}"
                        + ") / "
                        + f"{sds[j]}"
                        + "; "
                        + f"{means[j][0]}"
                        + " = "
                        + f"{means[j][1]}"
                        + "; "
                        + f"{stdevs[j][0]}"
                        + " = "
                        + f"{stdevs[j][1]}"
                    )
            poly_features.append(equation)
        poly_features = list(zip(xs, poly_features))

        # Получение коэффициентов

        ws = [f"w{i}" for i in range(n_poly_f)]
        info_coefs = list(zip(ws, coefs.tolist()[0]))
        info_coefs.append(("b", float(intercept)))

        # Получение выражения для вычисление score

        eq2 = [f"w{i}*x{i}" for i in range(n_poly_f)]
        eq_for_model = ""
        for i in range(n_poly_f):
            eq_for_model += eq2[i]
            eq_for_model += "+"
        eq_for_model += "b"

        # Запись данных в файл

        report += f"Название модели: {name_of_model}\n"
        report += "Диапазоны объектных параметров альфа-режимов:\n"
        for line in bounds_a:
            report += f"{line}\n"
        report += "\n"

        report += "Диапазоны объектных параметров бета-режимов:\n"
        for line in bounds_b:
            report += f"{line}\n"
        report += "\n"

        report += f"Время начала: {start_time[0]}, {start_time[1]}\n"
        report += f"Время окончания расчёта: {end_time[0]}, {end_time[1]}\n"
        report += "\n"

        report += f"Признаки, которые передаёт пользователь: {fs}.\n"
        report += "Нумерация соответствует признакам, получающимся на выходе модели.\n"
        report += "\n"

        report += "Формулы для вычисления признаков, которые использует разграничивающая модель:\n"
        for line in poly_features:
            report += f"{line[0]}" + " = " + f"{line[1]}\n"
        report += "Если какое-либо std равно нулю, то соответствующий этому признак приравнивается к нулю.\n"
        report += "\n"

        report += (
            "Значения коэффициентов, которые использует разграничивающая модель:\n"
        )
        for line in info_coefs:
            report += f"{line[0]}" + " = " + f"{line[1]}\n"
        report += "\n"

        report += "Выражение разграничивающей кривой:\n"
        report += f"{eq_for_model}\n"
        report += "\n"
        report += "Если выражение меньше нуля, то это альфа-режим, если больше нуля, то бета-режим.\n"
        report += "В случае равенства выражения нулю режим нельзя отнести к какому-либо классу.\n"
        report += "Но для обеспечения селективности необходимо отнести этот режим к бета-режимам.\n"

        return report
