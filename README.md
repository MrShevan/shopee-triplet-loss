# Запуск 
```bash
python3 data_split.py --csv_path /data/shopee-product-matching/train.csv
```
Обучение модели
```bash
python3 train_model.py
```

Валидация модели
```bash
python3 validate_model.py
```


# Аугментации
В качестве аугментаций используется:
```
* transforms. - аугментация из стандартной библиотеки торча torchvision
  mytransforms. - аугментации реализованные самостоятельно в lib.transforms
``` 

- `transforms.RandomHorizontalFlip`

  Случайные повороты картинок.
  
  ![alt](doc/randomflip_original.png)  ![alt](doc/randomflip_flipped.png)

- `mytransforms.RandomWatermark`

  Случайное добавление вотермарки, может быть иконка или лого компаний.
  
  ![alt](doc/randomwatermark_2.png)  ![alt](doc/randomwatermark_3.png)
  
  ![alt](doc/randomwatermark_5.png)  ![alt](doc/randomwatermark_4.png)

- `mytransforms.RandomText`

  Добавление случайного текста, разного размера, разных шрифтов.
  
  ![alt](doc/randomtext_1.png)  ![alt](doc/randomtext_3.png)
  
  ![alt](doc/randomtext_5.png)  ![alt](doc/randomtext_4.png)


- `mytransforms.RandomBound`

  Добавление границы.
  
  ![alt](doc/randombound_1.png)  ![alt](doc/randombound_2.png)
  

- `transforms.Resize`

  Изменение разрешения изображения.
 

- `mytransforms.RandomResizeWithPad`
   
  Уменьшение размера картинки с педдингом выбранным значением, до исходного размера.
  
  ![alt](doc/randomresizepad_1.png)  ![alt](doc/randomresizepad_2.png)
  

# Семплирование

В каждом батче содержится `n_classes`, число уникальных лейблов в нем и `n_samples`, сколько 
представителей этого класса взять в батч. Если представителей лейбла не хватает в выборке, то 
используется стратегия *over_sampling* и картинки берутся из уже существующих, в итоге 
все картинки получаются разные, так как у каждой своя аугментация.

Пример Батча после аугментации:

   ![alt](doc/randombatch.png)


# Генерация триплетов



Примеры получившихся триплетов:

   ![alt](doc/triplets_1.png)
    
   ![alt](doc/triplets_2.png)
    
   ![alt](doc/triplets_3.png)
   
 
# Архитектуры моделей