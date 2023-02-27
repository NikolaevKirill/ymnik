# Активное обучение для классификации режимов работы объекта
Запуск через streamlit: 
```
docker-compose up
```
Остановка: 
```
docker-compose down
```
Доступен по адресу:   ```http://localhost:8501```

---

Пример запуска файла ```active_learning.py```:
```
python utils\active_learning.py models\model.py models\params.txt
```
Тут передаются пути:
- до файла ```active_learning.py```;
- до файла с моделью (функция модели обязательно названа ```model```);
- до файла с параметрами.