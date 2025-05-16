import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = QdrantClient(url="http://localhost:6333")
collection_name = "document_chunks"

# def read_text_from_pdf(file_path):
#     try:
#         doc = fitz.open(file_path)
#         if doc.page_count == 0:
#             print(f"Файл {file_path} не містить сторінок.")
#             return None
#         text = ""
#         for i, page in enumerate(doc):
#             page_text = page.get_text("text")
#             text += f"[Сторінка {i + 1}]\n" + page_text + "\n"
#         print(f"Текст успішно зчитано з PDF-файлу: {file_path}")
#         return text
#     except Exception as e:
#         print(f"Помилка при зчитуванні PDF-файлу {file_path}: {e}")
#         return None

def read_text_from_txt(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не знайдено.")
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"Файл {file_path} порожній.")
            return None

        print(f"Текст успішно зчитано з TXT-файлу: {file_path}")
        return text
    except Exception as e:
        print(f"Помилка при зчитуванні TXT-файлу {file_path}: {e}")
        return None


def split_text_into_chunks(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    print(f"Розбито на {len(chunks)} чанків.")
    return chunks

def store_chunks_in_qdrant(chunks, model_name="multi-qa-mpnet-base-dot-v1"):
    try:
        print(f"Завантаження моделі {model_name}...")
        model = SentenceTransformer(model_name)
        print("Генерація ембеддингів для кожного чанку...")
        embeddings = model.encode(chunks, show_progress_bar=True)

        if client.collection_exists(collection_name=collection_name):
            print(f"Колекція '{collection_name}' вже існує. Видалення...")
            client.delete_collection(collection_name=collection_name)

        if not client.collection_exists(collection_name=collection_name):
            print(f"Створення Qdrant колекції '{collection_name}'...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embeddings.shape[1],
                    distance=Distance.COSINE
                )
            )

        points = [
            PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={"content": chunk}
            )
            for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))
        ]
        client.upsert(collection_name=collection_name, points=points)
        print(f"Успішно збережено {len(chunks)} чанків в Qdrant.")
        return True
    except Exception as e:
        print(f"Помилка при створенні Qdrant: {e}")
        return False

def search_qdrant(query, model_name="multi-qa-mpnet-base-dot-v1", top_k=30):
    try:
        model = SentenceTransformer(model_name)
        print(f"Генерація ембеддингу для запиту: '{query}'")
        query_embedding = model.encode([query])

        print("Пошук у Qdrant...")
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding[0],
            limit=top_k
        )

        results = [hit.payload["content"] for hit in search_result]

        return results[:top_k]
    except Exception as e:
        print(f"Помилка при пошуку в Qdrant: {e}")
        return []

def get_gemini_response(context_chunks, user_query):
    try:
        context = "\n".join(context_chunks)
        # Escape curly braces to avoid template variable errors
        context = context.replace("{", "{{").replace("}", "}}")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_api_key, temperature=0.0)

        messages = [
            {"role": "system", "content": "Ти помічник, який виконує пошук релевантної інформації за запитом, використовуючи контекст."},
            {"role": "user", "content": f"Контекст (витяги зі знайденого тексту):\n{context}\n\nДай відповідь на запит: {user_query}. Для кожної відповіді вкажи номер сторінки, звідки інформація була взята!"}
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        llm_response = chain.invoke({})
    except Exception as e:
        llm_response = str(e)

    return llm_response

def main(file_path, user_query):
    # long_text = read_text_from_pdf(file_path)
    long_text = read_text_from_txt(file_path)
    if not long_text:
        return "Файл не вдалося прочитати або він порожній."

    chunks = split_text_into_chunks(long_text)
    if not store_chunks_in_qdrant(chunks):
        return "Не вдалося створити Qdrant колекцію."

    results = search_qdrant(user_query)
    if results:
        print("Результати пошуку з Qdrant отримано успішно.")
    else:
        print("Не знайдено релевантних чанків для запиту в Qdrant.")

    final_response = get_gemini_response(results, user_query)
    return final_response

if __name__ == '__main__':
    file_path = ""
    user_query = ""
    response = main(file_path, user_query)
    print("###################")
    print(response)
