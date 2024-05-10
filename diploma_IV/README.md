# Методы федеративного обучения для агрегации моделей в анализе электрокардиограмм

## Структура репозитория

* ```dataset/``` - директория с всеми необходимыми данными и датасетами
    * ```data/``` - директория с данными
    * ```clients_dataset.csv``` - датасет для клиентов
    * ```server_dataset.csv``` - датасет для тренировки на стороне сервера (FLTrust, FLTrust_new)
    * ```test_dataset.csv``` - датасет для теста на стороне сервера (FLTrust_new)
* ```utils/``` - директория, содержащая config файл и все вспомогательные скрипты
    * ```config.yaml``` - общий конфиг-файл
    * ```data_utils.py``` - вспомогательные скрипты для работы с данными
    * ```loss_utils.py``` - скрипт, для получемния взвешиваной функции ошибки
    * ```metrics_utils.py``` - вспомогательные скрипты для подсчета метрик
    * ```model_utils.py``` - скрипт для вызова модели
    * ```resnet1d18.yaml``` - конфиг-файл архитектуры ResNet1D18
* ```federated_trainer.py``` - класс для федеративного обучения
* ```predict.py``` - скрипт для теста моделей
* ```train.py``` - скрипт для обучения моделей
* ```trainer.py``` - класс для централизованного обучения

## Тренировка

До тренировки нужно модифицировать ```utils/config.yaml``` следующим образом:

* **dataset_path**: абсолютный путь к ```dataset/clients_dataset.csv```
* **federated_params.server_dataset**: абсолютный путь к ```dataset/server_dataset.csv```
* **federated_params.test_dataset**: абсолютный путь к ```dataset/test_dataset.csv```

Для запуска тренировки нужно запустить ```train.py```, при желании поменяв следующие параметры:

* **mode**: Выбора режима, либо train, либо federated_train
* **training_params**:
    * **learning_rate**: Скорость обучения
    * **epochs**: Количество эпох (только для централизованного обучения)
    * **batch_size**: Размер пакета
* **federated_params**:
    * **method**: Метод федеративного обучения, доступны FedAvg, FedProx, FLTrust и FLTrust_new
    * **communication_rounds**: Количество коммуникационных раундов
    * **clients_num**: Количество клиентов
    * **round_epochs**: Количество локальных эпох у каждого клиента
    * **fed_prox_lambda**: Коэффициент для регуляризатора в методе FedProx
    * **attack**: Наличие атаки, значения либо True либо False
    * **amount_of_attackers**: Процент атакующих клиентов
    * **percent_of_changed_labels**: Процент записей, которая будет атакована у атакующего клиента

#### Пример запуска
```
python3 train.py mode=federated_train federated_params.method=FedAvg federated_params.attack=True
```

## Предсказание

Для запуска предсказания нужно запустить ```predict.py```, поменяв следующие параметры:

* **mode**: обязательно выставить test
* predict:
    * checkpoint_path: Путь к сохраненной после тренировки модели

#### Пример запуска
```
python3 predict.py predict.checkpoint_path=/path/to/chackpoint
```
