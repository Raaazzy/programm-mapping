from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pymorphy3 import MorphAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyvis.network import Network
import os
from fastapi.responses import HTMLResponse
from datetime import datetime
import json
from pathlib import Path
from fastapi import Response
from fastapi import status
import ast
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import seaborn as sns
import uuid
from datetime import datetime
import base64
sns.set_theme(style="whitegrid")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # Фронтенд на этом порту
    allow_credentials=False,                  
    allow_methods=["*"],                      
    allow_headers=["*"],                      
)

DB_PATH = Path("data") / "graph.db"
EXPORTS_DIR = "exports"

def get_db_connection():
    """Создание соединения с базой данных"""
    return sqlite3.connect(DB_PATH)

class GraphRequest(BaseModel):
    unique_keys: List[int]

@app.get("/disciplines")
async def get_disciplines():
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM max_table", conn)
        return df.to_dict(orient='records')
    except Exception as e:
        return {"error": f"Ошибка при чтении таблицы max_table: {str(e)}"}

@app.get("/disciplines/json")
async def get_disciplines():
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM max_table", conn)
        
        data = df.to_dict(orient='records')
        
        # Создаем папку exports, если не существует
        os.makedirs(EXPORTS_DIR, exist_ok=True)
        
        # Генерируем имя файла
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"disciplines_{timestamp}.json"
        filepath = os.path.join(EXPORTS_DIR, filename)
        
        # Сохраняем данные в JSON файл
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Возвращаем пустой ответ со статусом 200 OK
        return Response(status_code=200)
        
    except Exception as e:
        # В случае ошибки все равно возвращаем пустой ответ
        return Response(status_code=204)
    
@app.post("/graph")
async def get_graph(request: GraphRequest):
    try:
        with get_db_connection() as conn:
            # Создаем плейсхолдеры для SQL запроса
            placeholders = ','.join(['?'] * len(request.unique_keys))
            query = f"SELECT * FROM graph_table WHERE `Уникальный ключ` IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=request.unique_keys)
        
        # Если DataFrame пустой, возвращаем ошибку
        if df.empty:
            return {"error": "Не найдено данных по указанным ключам"}
        
        # Вычисление TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
        
        # Вычисление косинусного сходства
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Построение графа
        G = nx.Graph()
        for i, discipline in enumerate(df['Название дисциплины']):
            G.add_node(i, label=discipline)
        
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                similarity = similarity_matrix[i, j]
                if similarity > 0.2:  # Порог для добавления связи
                    G.add_edge(i, j, weight=similarity)
        
        # Создание интерактивного графа
        net = Network(notebook=False, width="100%", height="100vh")  # notebook=False для FastAPI
        for i, discipline in enumerate(df['Название дисциплины']):
            net.add_node(i, label=discipline, title=discipline)
        
        for i, j, data in G.edges(data=True):
            net.add_edge(i, j, value=data['weight'])
        
        # Генерируем HTML как строку
        html_content = net.generate_html()
        
        # Возвращаем чистый HTML
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content, status_code=200)
        
    except Exception as e:
        return {"error": f"Ошибка при построении графа: {str(e)}"}

@app.post("/graph/html")
async def get_graph_html(request: GraphRequest):
    try:
        with get_db_connection() as conn:
            # Создаем плейсхолдеры для SQL запроса
            placeholders = ','.join(['?'] * len(request.unique_keys))
            query = f"SELECT * FROM graph_table WHERE `Уникальный ключ` IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=request.unique_keys)
        
        # Если DataFrame пустой, возвращаем ошибку
        if df.empty:
            return {"error": "Не найдено данных по указанным ключам"}, status.HTTP_404_NOT_FOUND
        
        # Вычисление TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
        
        # Вычисление косинусного сходства
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Построение графа
        G = nx.Graph()
        for i, discipline in enumerate(df['Название дисциплины']):
            G.add_node(i, label=discipline)
        
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                similarity = similarity_matrix[i, j]
                if similarity > 0.2:  # Порог для добавления связи
                    G.add_edge(i, j, weight=similarity)
        
        # Создание интерактивного графа
        net = Network(notebook=False, width="100%", height="100%")
        for i, discipline in enumerate(df['Название дисциплины']):
            net.add_node(i, label=discipline, title=discipline)
        
        for i, j, data in G.edges(data=True):
            net.add_edge(i, j, value=data['weight'])
        
        # Создаем папку exports если её нет
        os.makedirs("exports", exist_ok=True)
        
        # Сохраняем файл в папку exports
        filename = f"graph_{len(request.unique_keys)}_items.html"
        filepath = os.path.join("exports", filename)
        net.save_graph(filepath)
        
        # Возвращаем успешный статус и информацию о файле
        return {
            "message": "Граф успешно создан и сохранен",
            "filename": filename,
            "filepath": filepath,
            "items_processed": len(request.unique_keys)
        }, status.HTTP_200_OK
        
    except Exception as e:
        return {"error": f"Ошибка при построении графа: {str(e)}"}, status.HTTP_500_INTERNAL_SERVER_ERROR
    
@app.get("/faculties")
async def get_faculties():
    try:
        with get_db_connection() as conn:
            # Запрос для получения уникальных значений факультетов
            query = "SELECT DISTINCT `Факультет` FROM max_table WHERE `Факультет` IS NOT NULL"
            df = pd.read_sql_query(query, conn)
        
        # Преобразуем DataFrame в список
        faculties_list = df['Факультет'].tolist()
        
        return {
            "faculties": faculties_list,
        }
        
    except Exception as e:
        return {"error": f"Ошибка при получении списка факультетов: {str(e)}"}, status.HTTP_500_INTERNAL_SERVER_ERROR

@app.get("/programs")
async def get_educational_programs():
    try:
        with get_db_connection() as conn:
            # Запрос для получения уникальных значений образовательных программ
            query = "SELECT DISTINCT `ОП` FROM max_table WHERE `ОП` IS NOT NULL"
            df = pd.read_sql_query(query, conn)
        
        # Преобразуем DataFrame в список
        programs_list = df['ОП'].tolist()
        
        return {
            "programs": programs_list,
        }
        
    except Exception as e:
        return {"error": f"Ошибка при получении списка образовательных программ: {str(e)}"}, status.HTTP_500_INTERNAL_SERVER_ERROR

@app.get("/keywords")
async def get_unique_keywords():
    try:
        with get_db_connection() as conn:
            # Запрос для получения всех ключевых слов
            query = "SELECT `Ключевые слова` FROM max_table WHERE `Ключевые слова` IS NOT NULL"
            df = pd.read_sql_query(query, conn)
        
        # Обрабатываем данные в формате "['граф', 'число', 'функция']"
        all_keywords = []
        for keywords_str in df['Ключевые слова']:
            if keywords_str and keywords_str.strip():
                try:
                    # Используем ast.literal_eval для безопасного преобразования строки в список
                    keywords_list = ast.literal_eval(keywords_str)
                    if isinstance(keywords_list, list):
                        all_keywords.extend(keywords_list)
                except (ValueError, SyntaxError):
                    # Если не удалось преобразовать, пропускаем эту запись
                    continue
        
        # Получаем уникальные ключевые слова, убираем пустые значения и сортируем
        unique_keywords = sorted(set([kw for kw in all_keywords if kw and kw.strip()]))
        
        return {
            "keywords": unique_keywords,
        }
        
    except Exception as e:
        return {"error": f"Ошибка при получении списка ключевых слов: {str(e)}"}, status.HTTP_500_INTERNAL_SERVER_ERROR

@app.post("/cloud")
async def generate_tag_cloud(request: GraphRequest):
    try:
        with get_db_connection() as conn:
            # Загружаем данные из таблицы max_table
            placeholders = ','.join(['?'] * len(request.unique_keys))
            query = f"SELECT * FROM max_table WHERE `Уникальный ключ` IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=request.unique_keys)
        
        # Если DataFrame пустой, возвращаем ошибку
        if df.empty:
            return {"error": "Не найдено данных в таблице max_table"}, status.HTTP_404_NOT_FOUND
        
        # Обрабатываем ключевые слова из формата "['слово1', 'слово2']"
        def parse_keywords(keywords_str):
            if keywords_str and keywords_str.strip():
                try:
                    keywords_list = ast.literal_eval(keywords_str)
                    if isinstance(keywords_list, list):
                        return keywords_list
                except (ValueError, SyntaxError):
                    return []
            return []
        
        # Преобразуем строки с ключевыми словами в списки
        df['parsed_keywords'] = df['Ключевые слова'].apply(parse_keywords)
        
        # Группируем по ОП и объединяем ключевые слова
        keywords_by_op = df.groupby("ОП")["parsed_keywords"].apply(lambda x: [item for sublist in x for item in sublist]).reset_index()
        
        # Объединяем все ключевые слова в один список
        all_keywords = [keyword for sublist in keywords_by_op["parsed_keywords"] for keyword in sublist]
        
        # Подсчитываем частоту встречаемости
        keyword_counts = Counter(all_keywords)
        
        # Если нет ключевых слов, возвращаем ошибку
        if not keyword_counts:
            return {"error": "Не найдено ключевых слов для построения облака тегов"}, status.HTTP_404_NOT_FOUND
        
        # Создаем облако тегов
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color="white", 
            colormap="viridis",
            max_words=100
        ).generate_from_frequencies(keyword_counts)
        
        # Создаем изображение в памяти
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        
        # Сохраняем изображение в буфер памяти
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Получаем байты из буфера
        img_bytes = img_buffer.getvalue()
        img_buffer.close()
        
        # Возвращаем изображение как ответ (отображается в браузере)
        return Response(
            content=img_bytes,
            media_type="image/png"
        )
        
    except Exception as e:
        return {"error": f"Ошибка при создании облака тегов: {str(e)}"}, status.HTTP_500_INTERNAL_SERVER_ERROR

@app.post("/cloud/png")
async def generate_tag_cloud_png(request: GraphRequest):
    try:
        with get_db_connection() as conn:
            # Загружаем данные из таблицы max_table
            placeholders = ','.join(['?'] * len(request.unique_keys))
            query = f"SELECT * FROM max_table WHERE `Уникальный ключ` IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=request.unique_keys)
        
        # Если DataFrame пустой, возвращаем ошибку
        if df.empty:
            return {"error": "Не найдено данных в таблице max_table"}, status.HTTP_404_NOT_FOUND
        
        # Обрабатываем ключевые слова из формата "['слово1', 'слово2']"
        def parse_keywords(keywords_str):
            if keywords_str and keywords_str.strip():
                try:
                    keywords_list = ast.literal_eval(keywords_str)
                    if isinstance(keywords_list, list):
                        return keywords_list
                except (ValueError, SyntaxError):
                    return []
            return []
        
        # Преобразуем строки с ключевыми словами в списки
        df['parsed_keywords'] = df['Ключевые слова'].apply(parse_keywords)
        
        # Группируем по ОП и объединяем ключевые слова
        keywords_by_op = df.groupby("ОП")["parsed_keywords"].apply(lambda x: [item for sublist in x for item in sublist]).reset_index()
        
        # Объединяем все ключевые слова в один список
        all_keywords = [keyword for sublist in keywords_by_op["parsed_keywords"] for keyword in sublist]
        
        # Подсчитываем частоту встречаемости
        keyword_counts = Counter(all_keywords)
        
        # Если нет ключевых слов, возвращаем ошибку
        if not keyword_counts:
            return {"error": "Не найдено ключевых слов для построения облака тегов"}, status.HTTP_404_NOT_FOUND
        
        # Создаем облако тегов
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color="white", 
            colormap="viridis",
            max_words=100
        ).generate_from_frequencies(keyword_counts)
        
        # Создаем папку exports если её нет
        os.makedirs("exports", exist_ok=True)
        
        # Сохраняем изображение
        filename = f"tag_cloud_{len(keyword_counts)}_keywords.png"
        filepath = os.path.join("exports", filename)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Закрываем figure чтобы освободить память
        
        # Возвращаем информацию о созданном файле
        return {
            "message": "Облако тегов успешно создано",
            "filename": filename,
            "filepath": filepath,
            "total_keywords": len(keyword_counts),
            "top_keywords": dict(keyword_counts.most_common(10))  # Топ-10 ключевых слов
        }, status.HTTP_200_OK
        
    except Exception as e:
        return {"error": f"Ошибка при создании облака тегов: {str(e)}"}, status.HTTP_500_INTERNAL_SERVER_ERROR

@app.get("/dashboards")
async def get_graphics():
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM graphics_table", conn)
        return df.to_dict(orient='records')
    except Exception as e:
        return {"error": f"Ошибка при чтении таблицы graphics_table: {str(e)}"}
    
@app.post("/dashboards")
async def generate_control_charts(request: GraphRequest):
    try:
        with get_db_connection() as conn:
            # Загружаем данные из max_table и соединяем с graphics_table
            placeholders = ','.join(['?'] * len(request.unique_keys))
            query_max = f"SELECT * FROM max_table WHERE `Уникальный ключ` IN ({placeholders})"
            df_max = pd.read_sql_query(query_max, conn, params=request.unique_keys)
            
            # Получаем данные из graphics_table для JOIN
            query_graphics = "SELECT * FROM graphics_table"
            df_graphics = pd.read_sql_query(query_graphics, conn)
        
        # Если DataFrame пустой, возвращаем ошибку
        if df_max.empty or df_graphics.empty:
            return {"error": "Не найдено данных в таблицах"}, status.HTTP_404_NOT_FOUND
        
        # Выполняем INNER JOIN по уникальному ключу
        df = pd.merge(df_max, df_graphics, on='Уникальный ключ', how='inner')
        
        # Устанавливаем шрифт для поддержки кириллицы и избежания предупреждений
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
        
        charts_data = {}
        
        # 1. Распределение элементов контроля по типам
        plt.figure(figsize=(8, 5))
        sns.countplot(y=df["Тип"], hue=df["Тип"], palette="coolwarm", 
                     order=df["Тип"].value_counts().index, legend=False)
        plt.title("Распределение элементов контроля по типам")
        plt.xlabel("Количество")
        plt.ylabel("Тип элемента контроля")
        plt.tight_layout()
        
        img_buffer1 = io.BytesIO()
        plt.savefig(img_buffer1, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        charts_data["control_distribution_type"] = base64.b64encode(img_buffer1.getvalue()).decode('utf-8')
        img_buffer1.close()
        
        # 2. Элементы контроля по дисциплинам
        plt.figure(figsize=(10, 40))
        discipline_counts = df.groupby("Название дисциплины")["Название контроля"].count().sort_values(ascending=False)
        discipline_counts.plot(kind="barh", color="lightcoral")
        plt.title("Распределение элементов контроля по их количеству")
        plt.xlabel("Количество")
        plt.ylabel("Дисциплина")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        img_buffer2 = io.BytesIO()
        plt.savefig(img_buffer2, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        charts_data["control_distribution_number"] = base64.b64encode(img_buffer2.getvalue()).decode('utf-8')
        img_buffer2.close()
        
        # 3. Формат проведения (offline, online и т. д.)
        plt.figure(figsize=(6, 5))
        format_counts = df["Формат проведения"].value_counts()
        format_counts.plot(kind="pie", autopct='%1.1f%%', colors=["lightblue", "orange", "green"])
        plt.title("Распределение элементов контроля по формату проведения")
        plt.ylabel("")
        plt.tight_layout()
        
        img_buffer3 = io.BytesIO()
        plt.savefig(img_buffer3, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        charts_data["control_distribution_format"] = base64.b64encode(img_buffer3.getvalue()).decode('utf-8')
        img_buffer3.close()
        
        # 4. Доля экзаменов vs. неэкзаменов
        plt.figure(figsize=(5, 5))
        exam_counts = df["Является экзаменом"].value_counts()
        exam_labels = ["Не экзамен", "Экзамен"]
        plt.pie(exam_counts, labels=exam_labels, autopct='%1.1f%%', colors=["lightgreen", "red"])
        plt.title("Доля экзаменов в элементах контроля")
        plt.tight_layout()
        
        img_buffer4 = io.BytesIO()
        plt.savefig(img_buffer4, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        charts_data["control_distribution_exam"] = base64.b64encode(img_buffer4.getvalue()).decode('utf-8')
        img_buffer4.close()
        
        # 5. Тип блокирования
        plt.figure(figsize=(8, 5))
        sns.countplot(y=df["Тип блокирования"], hue=df["Тип блокирования"], palette="viridis", 
                     order=df["Тип блокирования"].value_counts().index, legend=False)
        plt.title("Распределение элементов контроля по типу блокирования")
        plt.xlabel("Количество")
        plt.ylabel("Тип блокирования")
        plt.tight_layout()
        
        img_buffer5 = io.BytesIO()
        plt.savefig(img_buffer5, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        charts_data["control_distribution_block"] = base64.b64encode(img_buffer5.getvalue()).decode('utf-8')
        img_buffer5.close()
        
        # 6. Распределение по названию элементов контроля
        plt.figure(figsize=(8, 60))
        sns.countplot(y=df["Название контроля"], hue=df["Название контроля"], palette="coolwarm", 
                     order=df["Название контроля"].value_counts().index, legend=False)
        plt.title("Распределение элементов контроля по названию")
        plt.xlabel("Количество")
        plt.ylabel("Название элемента контроля")
        plt.tight_layout()
        
        img_buffer6 = io.BytesIO()
        plt.savefig(img_buffer6, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        charts_data["control_distribution_name"] = base64.b64encode(img_buffer6.getvalue()).decode('utf-8')
        img_buffer6.close()
        
        # Возвращаем все графики как base64 строки
        return {
            "charts": charts_data
        }
        
    except Exception as e:
        return {"error": f"Ошибка при создании графиков контроля: {str(e)}"}, status.HTTP_500_INTERNAL_SERVER_ERROR

@app.post("/dashboards/png")
async def generate_control_charts_png(request: GraphRequest):
    try:
        with get_db_connection() as conn:
            # Загружаем данные из max_table и соединяем с graphics_table
            placeholders = ','.join(['?'] * len(request.unique_keys))
            query_max = f"SELECT * FROM max_table WHERE `Уникальный ключ` IN ({placeholders})"
            df_max = pd.read_sql_query(query_max, conn, params=request.unique_keys)
            
            # Получаем данные из graphics_table для JOIN
            query_graphics = "SELECT * FROM graphics_table"
            df_graphics = pd.read_sql_query(query_graphics, conn)
        
        # Если DataFrame пустой, возвращаем ошибку
        if df_max.empty or df_graphics.empty:
            return {"error": "Не найдено данных в таблицах"}, status.HTTP_404_NOT_FOUND
        
        # Выполняем INNER JOIN по уникальному ключу
        df = pd.merge(df_max, df_graphics, on='Уникальный ключ', how='inner')
        
        # Создаем уникальную папку для этого набора графиков
        folder_name = f"control_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        folder_path = os.path.join("exports", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Устанавливаем шрифт для поддержки кириллицы и избежания предупреждений
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
        
        charts_info = []
        
        # 1. Распределение элементов контроля по типам
        plt.figure(figsize=(8, 5))
        sns.countplot(y=df["Тип"], hue=df["Тип"], palette="coolwarm", 
                     order=df["Тип"].value_counts().index, legend=False)
        plt.title("Распределение элементов контроля по типам")
        plt.xlabel("Количество")
        plt.ylabel("Тип элемента контроля")
        plt.tight_layout()
        
        chart1_path = os.path.join(folder_path, "control_distribution_type.png")
        plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts_info.append({"name": "Типы контроля", "file": "control_distribution_type.png"})
        
        # 2. Элементы контроля по дисциплинам
        plt.figure(figsize=(10, 40))
        discipline_counts = df.groupby("Название дисциплины")["Название контроля"].count().sort_values(ascending=False)
        discipline_counts.plot(kind="barh", color="lightcoral")
        plt.title("Распределение элементов контроля по их количеству")
        plt.xlabel("Количество")
        plt.ylabel("Дисциплина")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        chart2_path = os.path.join(folder_path, "control_distribution_number.png")
        plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts_info.append({"name": "Контроль по дисциплинам", "file": "control_distribution_number.png"})
        
        # 3. Формат проведения (offline, online и т. д.)
        plt.figure(figsize=(6, 5))
        format_counts = df["Формат проведения"].value_counts()
        format_counts.plot(kind="pie", autopct='%1.1f%%', colors=["lightblue", "orange", "green"])
        plt.title("Распределение элементов контроля по формату проведения")
        plt.ylabel("")
        plt.tight_layout()
        
        chart3_path = os.path.join(folder_path, "control_distribution_format.png")
        plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts_info.append({"name": "Формат проведения", "file": "control_distribution_format.png"})
        
        # 4. Доля экзаменов vs. неэкзаменов
        plt.figure(figsize=(5, 5))
        exam_counts = df["Является экзаменом"].value_counts()
        exam_labels = ["Не экзамен", "Экзамен"]
        plt.pie(exam_counts, labels=exam_labels, autopct='%1.1f%%', colors=["lightgreen", "red"])
        plt.title("Доля экзаменов в элементах контроля")
        plt.tight_layout()
        
        chart4_path = os.path.join(folder_path, "control_distribution_exam.png")
        plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts_info.append({"name": "Доля экзаменов", "file": "control_distribution_exam.png"})
        
        # 5. Тип блокирования
        plt.figure(figsize=(8, 5))
        sns.countplot(y=df["Тип блокирования"], hue=df["Тип блокирования"], palette="viridis", 
                     order=df["Тип блокирования"].value_counts().index, legend=False)
        plt.title("Распределение элементов контроля по типу блокирования")
        plt.xlabel("Количество")
        plt.ylabel("Тип блокирования")
        plt.tight_layout()
        
        chart5_path = os.path.join(folder_path, "control_distribution_block.png")
        plt.savefig(chart5_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts_info.append({"name": "Тип блокирования", "file": "control_distribution_block.png"})
        
        # 6. Распределение по названию элементов контроля
        plt.figure(figsize=(8, 60))
        sns.countplot(y=df["Название контроля"], hue=df["Название контроля"], palette="coolwarm", 
                     order=df["Название контроля"].value_counts().index, legend=False)
        plt.title("Распределение элементов контроля по названию")
        plt.xlabel("Количество")
        plt.ylabel("Название элемента контроля")
        plt.tight_layout()
        
        chart6_path = os.path.join(folder_path, "control_distribution_name.png")
        plt.savefig(chart6_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts_info.append({"name": "Названия контроля", "file": "control_distribution_name.png"})
        
        # Возвращаем информацию о созданных графиках
        return {
            "message": "Графики контроля успешно созданы",
            "folder": folder_name,
            "folder_path": folder_path,
            "total_charts": len(charts_info),
            "charts": charts_info,
            "data_statistics": {
                "total_records": len(df),
                "unique_disciplines": df["Название дисциплины"].nunique(),
                "unique_control_types": df["Тип"].nunique()
            }
        }, status.HTTP_200_OK
        
    except Exception as e:
        return {"error": f"Ошибка при создании графиков контроля: {str(e)}"}, status.HTTP_500_INTERNAL_SERVER_ERROR
    
@app.get("/health")
async def health_check():
    return {"status": "healthy"}