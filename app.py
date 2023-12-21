import os
import shutil
import pickle

from model import Model
from BankNotes import BankNote
from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.get("/")
async def root():
    """
    Эта функция возвращает приветственное сообщение.
    """
    return {"message": "Hello World"}


@app.get("/models")
async def get_models():
    """
    Эта функция возвращает список доступных моделей.
    """
    return {"models": ["LGBM", "GBClassifier"]}


@app.post("/train/{model_name}")
async def train_model(model_type: str,
                      model_name: str,
                      file: UploadFile = File(...)):
    """
    Эта функция обучает модель на основе переданных данных,
    и сохраняет ее в файл.
    :param model_type: Тип модели для обучения.
    :param model_name: Имя модели для сохранения.
    :param file: Файл с данными для обучения.
    """
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    model = Model(model_type)
    model.fit(file.filename)
    with open(f'{model_type}_{model_name}.pkl', 'wb') as file:
        pickle.dump(model, file)
    return {"message": f"Model {model_type}_{model_name} trained successfully"}


@app.post("/predict/{model_name}")
async def predict(model_name: str, data: BankNote):
    """
    Эта функция выполняет предсказание на основе
    переданных данных с использованием сохраненной модели.
    :param model_name: Имя модели для загрузки.
    :param data: Данные для предсказания.
    """
    try:
        with open(f'{model_name}.pkl', 'rb') as file:
            model = pickle.load(file)
    except Exception:
        return {"status": "Failed",
                "message": "Модели с таким названием не существует."}
    try:
        data = data.dict()
        variance = data['variance']
        skewness = data['skewness']
        curtosis = data['curtosis']
        entropy = data['entropy']
        prediction = model.predict([[variance, skewness, curtosis, entropy]])
    except Exception:
        return {"status": "Failed",
                "message": "Проверьте корректность данных."}
    return {"prediction": prediction.tolist()}


@app.post("/existing_models")
async def get_all_exiscting_models():
    """
    Эта функция возвращает список всех сохраненных моделей.
    """
    pkl_files = []
    for file in os.listdir():
        if file.endswith('.pkl'):
            pkl_files.append(file[:-4])
    if pkl_files == '':
        return {"message": "Нет обученных моделей."}
    else:
        return {"message": f"Список всех обученных моделей: {pkl_files}"}


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """
    Эта функция удаляет сохраненную модель.
    :param model_name: Имя модели для удаления.
    """
    try:
        os.remove(f"{model_name}.pkl")
        return {"message": f"Модель {model_name} удалена успешно"}
    except FileNotFoundError:
        return {"message": f"Модель {model_name} не найдена"}
