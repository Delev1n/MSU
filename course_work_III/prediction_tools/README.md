# Классификация

#### Установите необходимые библиотеки
```
pip install -r requirements.txt
```

## Тренировка

Для запуска тренировки нужно запустить ```train.py``` со следующими параметрами:

* model: Модель, которую желаете тренировать
* training_params:
    * folds: Количество фолдов в кросс валидации
    * batch_size: Размер пакета
    * lr: Скорость обучения
    * epochs: Количество эпох
* task_params:
    * pathology_names: Название патологии, которую желаете классифицировать
    * merge_map: Нужно для объединения нескольких патологий, например IRBBB, CRBBB
* ecg_record_params:
    * input_type: Тип входных данных (1D, 2D)
    * base (2D): Путь в директорию где находится папка "combined", которую создал конвертер
    * size (2D): Размер картинки, которая подается на вход

#### Пример запуска
```
python train.py model=resnet2d18 training_params.batch_size=32 task_params.pathology_names=[AFIB] ecg_record_params.input_type=2d ecg_record_params.base=E:/Work/PTB_XL/scalograms ecg_record_params.size=[512,1536]
```

## Предсказание

Для запуска предсказания нужно запустить ```predict.py``` со следующими параметрами:

* model: Модель, которую желаете тренировать
* predict:
    * checkpoint_path: Путь к сохраненной после тренировки модели
* task_params:
    * pathology_names: Название патологии, которую желаете классифицировать
    * merge_map: Нужно для объединения нескольких патологий, например IRBBB, CRBBB
* ecg_record_params:
    * input_type: Тип входных данных (1D, 2D)
    * base (2D): Путь в директорию где находится папка "combined", которую создал конвертер
    * size (2D): Размер картинки, которая подается на вход

#### Пример запуска
```
python predict.py model=resnet2d18 task_params.pathology_names=[AFIB] ecg_record_params.input_type=2d ecg_record_params.base=E:/Work/PTB_XL/scalograms ecg_record_params.size=[512,1536] predict.checkpoint_path=E:/Work/MSU/course_work_III/prediction_tools/outputs/2023-03-21/12-36-07/12_leads_experiment_ptbxl_resnet2d18_PVC_0.pt
```
