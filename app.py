from datetime import datetime

import numpy as np
import streamlit as st

import utils.active_learning as active_learning

header = st.container()
inputs = st.container()
body = st.container()

# Для оповещений
if "info" not in st.session_state:  # Для показа справки
    st.session_state.info = 0

if "start_learning" not in st.session_state:  # Для показа справки
    st.session_state.start_learning = False

if "success_learning" not in st.session_state:  # Для вывода скачивания отчёта
    st.session_state.success_learning = False

if "clf" not in st.session_state:  # Для скачивания отчёта
    st.session_state.clf = None

if "iter_learning" not in st.session_state:  # Для progress bar
    st.session_state.iter_learning = 0

if "success_stop_criterion" not in st.session_state:  # flag of stop learning
    st.session_state.success_stop_criterion = False

success_input_params = False  # Для уведомления о загрузке параметров
success_input_model_and_params = False  # Для вывода сообщений о параметрах
#  и модели
success_learning = False  # Для уведомления об окончании обучения

success_params = True  # Правильный файл с параметрами
success_model = True  # Параметры подходят модели
success_choose_model = True  # Выбрана какая-либо модель
success_func_model = True  # В модели нет ошибок
success_data_params = True  # Переданные значения являются числами
#  (могут быть конвертированы в числа)

type_er = None  # type of error

model = None
maxiter = 400  # number of iterations
window_shape = 10  # shape mooving window for stop ctiterion
threshold = 0.5  # threshold of difference resistance of alpha-modes
# for stop learning in percentage

with header:
    st.markdown(
        "### АВТОМАТИЗИРОВАННАЯ СИСТЕМА ПОСТРОЕНИЯ ИНТЕЛЛЕКТУАЛЬНОЙ РЕЛЕЙНОЙ ЗАЩИТЫ"
    )
    st.markdown(
        "###### Данный сервис позволяет рассчитать полином, описывающий разделяющую кривую между нормальными и аварийными режимами для заданной модели и заданных диапазонов объектных параметров."
    )
    info = st.button("Справка")
    if info:
        st.session_state.info += 1
    if st.session_state.info % 2 != 0:
        st.info(
            """Для выполнения расчёта необходимо задать модель объекта, а также диапазоны объектных параметров:
1) Модель
Модель описывается на языке Python. Скачать пример модели можно нажав соответствующую кнопку. Рекомендуется создавать модель таким образом, чтобы она была способна принимать данные в виде матрицы (в качестве примера приводится именно такая модель), такой подход позволит сократить время расчёта. Однако, для сложных объектов такой подход затруднителен, поэтому допускается использование моделей, которые за одну итерацию способны рассчитать только 1 режим. Система автоматически определит тип модели. В приведённой модели в качестве объекта используется линия с двухсторонним питанием.

2) Объектные параметры
Файл с объектными параметрами - обыкновенный текстовый файл в формате *.txt. Порядок следования параметров в файле должно строго соответствовать порядку следования переменных, которые принимает модель. При этом, данный файл содержит как параметры альфа-режимов, так и параметры бета-режимов.

Альфа-режимы - режимы, в которых необходимо постараться обеспечить срабатывание защиты;

Бета-режимы - режимы, в которых необходимо гарантировать несрабатывание защиты.

Структура файла: каждая строка соответствует одному объектному параметру (например, угол передачи линии, переходное сопротивление в месте КЗ и т.д.), в каждой строке имеется 2 числа (допускается целый и вещественный типы), разделённые пробелом. При этом, первое число соответсвует левой границе диапазона, а второе число правой границе диапазона. Первое число должно быть меньше второго. Сначала перечисляются параметры альфа режимов, затем параметры бета-режимов. При этом, порядок следования параметров должен совпадать как в альфа-режимах и в бета-режимах, так и в параметрах принимаемых моделью. Например, если модель принимает 2 параметра: Rf, xf, то файл с объектными параметрами должен содержать 4 строки, где первые 2 строки будут соответсовать Rf и xf для альфа-режимов, а строки 3-4 будут соответсовать Rf и xf бета-режимов.
Скачать пример можно нажав соответствующую кнопку.""",
            icon="ℹ️",
        )
        st.info(
            """Работа с сервисом

После подготовки файла модели (в формате .py), а также файла с диапазоном объектных параметров (в формате .txt), можно приступить к расчёту. Для этого необходимо загрузить подготовленные файлы в соответствующие поля "Загрузить модель" и "Загрузить диапазоны объектных параметров". Поддерживается Drag'n'Drop, либо можно воспользоваться кнопкой "Browse files".

После загрузки файлов сервис выполнит их автоматическую проверку и сообщит о возможности выполнить расчёт, либо сообщит об ошибке в загруженных файлах.

По завершению расчета нажмите на кнопку "Сформировать конфигурационный файл".
""",
            icon="ℹ️",
        )
        st.info(
            """Структура выходного файла

Название модели (соответсвует названию загруженного файла модели).

Диапазоны альфа- и бета-режимов (соответсвуют содержимому файла с объектными параметрами).

Время начала и завершения расчёта (в случае, когда web-интерфейс и непосредственно сервис работают на разных ПК, используется время устройства, на котором работает сервис).

Признаки, которые передаёт пользователь - то есть выходные параметры модели, загруженной пользователем.

Формулы, необходимые для расчёта признаков, используемых в разграничивающей кривой (на основе выходных параметров модели).

Значение коээфициентов, используемых разграничивающей кривой.

Выражение разграничивающей кривой.

Короткое описание разграничивающей кривой.""",
            icon="ℹ️",
        )
        with open("models/model.py", "rb") as file:
            st.download_button(
                label="Скачать пример модели",
                data=file,
                file_name="example_of_model.py",
                mime="application/octet-stream",
            )
        with open("models/params.txt", "rb") as file:
            st.download_button(
                label="Скачать пример диапазонов...",
                data=file,
                file_name="example_of_params.txt",
            )
        st.markdown("""---""")

with inputs:
    input_model = st.file_uploader("Загрузить модель:", type=["py"])
    if input_model is not None:
        exec(input_model.read())
        name_of_model = input_model.name.split(".")[0]

    params = st.file_uploader("Загрузить диапазоны объектных парметров:", type=["txt"])
    if params is not None:
        success_input_params = True
        try:
            inp_bounds = np.loadtxt(params)
        except ValueError:
            success_data_params = False
        if success_data_params:
            if (inp_bounds.shape[0] % 2 != 0) or np.any(
                inp_bounds.T[0] > inp_bounds.T[1]
            ):
                success_params = False  # Количество объектных параметров
                # альфа- и бета-режимов отличаются или границы указаны
                # не по возрастанию
            else:
                inp_bounds_a = inp_bounds[: int(len(inp_bounds) / 2)]
                inp_bounds_b = inp_bounds[int(len(inp_bounds) / 2) :]

                try:
                    model(inp_bounds_a.T[0])
                    model(inp_bounds_b.T[0])
                except ZeroDivisionError as error_type:
                    success_func_model = False  # происходит деление на ноль
                    type_er = error_type
                except NameError as error_type:
                    success_func_model = False  # Что-то не так в модели
                    # (в коде)
                    type_er = error_type
                except ValueError as error_type:
                    success_model = (
                        False  # Количество передаваемых объектных параметров
                    )
                    type_er = error_type
                    # не соответствует, количеству принимаемых моделью
                    # параметров
                except TypeError:
                    success_choose_model = False
    if success_input_params:
        if (
            success_params
            and success_data_params
            and success_model
            and success_func_model
            and success_choose_model
        ):
            success_input_model_and_params = True
            st.success(
                """Диапазоны объектных параметров получены. Количество
                объектных параметров соответствует количеству параметров,
                принимаемых моделью.""",
                icon="✅",
            )
        elif (not success_params) or (not success_data_params):
            st.error(
                """Файл с параметрами составлен в неправильном формате.
                Прочтите, как должен выглядеть файл с параметрами.""",
                icon="🚨",
            )
        elif not success_func_model:
            st.error(
                f"""В переданной функции есть ошибка: {type_er}.
                Перепроверьте функцию.""",
                icon="🚨",
            )
        elif not success_choose_model:
            st.error(
                "Загрузите модель.",
                icon="🚨",
            )
        elif not success_model:
            st.error(
                """Количество передаваемых объектных параметров не
                соответствует, количеству принимаемых моделью
                параметров""",
                icon="🚨",
            )
    else:
        st.info(
            """Для выполнения расчёта загрузите модель и объектные
             параметры.""",
            icon="ℹ️",
        )

    st.text("")

with body:
    start_flag = st.button("Начать расчёт")
    if start_flag:
        st.session_state.success_learning = False
        st.session_state.success_stop_criterion = False
        st.session_state.start_learning = True
        my_bar = st.progress(0)
    if st.session_state.start_learning:
        if success_input_model_and_params and start_flag:
            clf = active_learning.ActiveLearning()

            start_t = str(datetime.now()).split()
            start_time = (start_t[1][:8], start_t[0])

            clf.initialize(model=model, bounds_a=inp_bounds_a, bounds_b=inp_bounds_b)
            st.button(
                "Остановить расчёт", disabled=st.session_state.success_stop_criterion
            )

            for iteration in range(int(maxiter / 2)):
                iter_learning = iteration / maxiter
                my_bar.progress(iter_learning)
                clf.step()

            for iteration in range(int(maxiter / 2), maxiter):
                iter_learning = iteration / maxiter
                my_bar.progress(iter_learning)
                clf.step()
                if (iteration - maxiter / 2) >= window_shape:
                    s_r = np.array(clf.s_r)[-(window_shape + 1) :]
                    relative_diff = np.abs(np.diff(s_r)) / s_r[:-1] * 100
                    if np.sum(relative_diff < threshold) == window_shape:
                        st.session_state.success_stop_criterion = True
                        break

            st.session_state.iter_learning = 1
            my_bar.progress(st.session_state.iter_learning)

            # Запись времени окончания расчёта
            end_t = str(datetime.now()).split()
            end_time = (end_t[1][:8], end_t[0])

            clf.start_time = start_time
            clf.end_time = end_time

            st.session_state.clf = clf
            st.session_state.success_learning = True

        elif start_flag:
            st.error(
                """Вы не можете начать расчёт, пока не загружены модель и
                объектные параметры""",
                icon="🚨",
            )
        if st.session_state.start_learning and st.session_state.success_learning:
            st.success(
                """Обучение закончено. Можете сформировать конфигурационный
                файл.""",
                icon="✅",
            )
            if not st.session_state.success_stop_criterion:
                st.info(
                    """Обучение завершено, но требуется дополнительная
                    проверка селективности алгоритма.""",
                    icon="ℹ️",
                )
        if st.session_state.success_learning:
            st.download_button(
                label="Сформировать конфигурационный файл",
                data=st.session_state.clf.create_report(),
                file_name=f"{name_of_model}.txt",
            )
