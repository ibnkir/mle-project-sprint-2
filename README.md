## Яндекс Практикум, курс "Инженер Машинного Обучения" (2024 г.)
## Проект 2-го спринта: "Улучшение baseline-модели"
### Выполнил: Кирилл Н., email: ibnkir@yandex.ru, бакет в S3: s3-student-mle-20240227-804436ded9

### Краткое описание
Целью проекта является практическое освоение MLflow для мониторинга процесса обучения, логирования параметров, метрик, моделей и других артефактов, а также использование различных инструментов для улучшения моделей, включая генерацию признаков и подбор гипер-параметров. В качестве бейзлайна используется модель предсказания цен на квартиры из проекта 1-го спринта.

Основные инструменты:
- MLflow,
- Autofeat, Featuretools - библиотеки для генерации признаков,
- mlxtend - библиотека для отбора признаков,
- Optuna для байевской оптимизации поиска гипер-параметров.

### Установка
git clone https://github.com/ibnkir/mle-project-sprint-2

cd ./mle-project-sprint-2

pip install -r requirements.txt

### Руководство по проекту
#### Этап 1: Развертывание MLflow и регистрация модели
- Скрипт для развертывания MLflow и ноутбук для пошагового выполнения кода находятся в папке ./mlflow_server.
- Запуск скрипта из командной строки:

cd ./mlflow_server

sh run_mlflow_server

Результаты логирование базовой модели:
- Название эксперимента: 'mle-project-sprint-2';
- Название запуска: 'flats_price_baseline_model_logging';
- ID запуска: 
- Название зарегистрированной модели: 'flats_price_baseline_model'.


#### Этап 2: Проведение EDA 

Шаги EDA:
- Загружаем данные из таблиц flats и buildings;
- Переименовываем id на flat_id, чтобы не путать его с автоматически создаваемой индексной колонкой БД;
- Удаляем строки с пустыми/одинаковыми flat_id и пустыми/нулевыми/отрицательными ценами;
- Удаляем строки с одинаковыми признаками;
- Заполняем пропуски в признаках (для вещественных используем среднее значение, для остальных - моду);
- Удаляем выбросы у целых и вещественных признаков;
- Строим гистограммы целочисленных признаков, чтобы решить, какие их них можно считать количественными, а какие категориальными;

Выводы:
- Признак building_type_int имеет мало значений, поэтому будем считать его категориальным. Остальные целочисленные признаками считаем количественными. 
- Чтобы уменьшить масштаб значений, используем вместо build_year вводим новый признак - building_ageвместо building_type_intперед обучением модели будем вычислять возраст здания - building_age, как разницу между текущим годом и с, и использовать его вместо build_year.


#### Этап 3: Генерация признаков и обучение модели
Укажите, какие признаки были сгенерированы и как это повлияло на модель. Добавьте названия запусков в MLflow для этапа генерации признаков.

#### Этап 4: Отбор признаков и обучение новой версии модели
Опишите процесс отбора признаков, а также назовите запуски в MLflow, связанные с этим этапом.

#### Этап 5: Подбор гиперпараметров и обучение новой версии модели
Опишите использованные методы подбора гиперпараметров, дайте оценку их влияния на результаты модели. Добавьте названия соответствующих запусков в MLflow. 
