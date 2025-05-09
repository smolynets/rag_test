import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import PyPDF2

gemini_api_key = ""


def read_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print(f"Текст успішно зчитано з файлу: {file_path}")
        return text
    except FileNotFoundError:
        print(f"Файл {file_path} не знайдено. Переконайтеся, що файл існує.")
        return None
    except Exception as e:
        print(f"Помилка при зчитуванні файлу {file_path}: {e}")
        return None

def read_text_from_pdf(file_path):
    try:
        # Відкрити PDF-файл
        with open(file_path, 'rb') as file:
            # Створити PDF-читач
            reader = PyPDF2.PdfReader(file)
            
            # Перевірити, чи файл не порожній
            if len(reader.pages) == 0:
                print(f"Файл {file_path} не містить сторінок.")
                return None
            
            # Витягнути текст зі всіх сторінок PDF
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            print(f"Текст успішно зчитано з PDF-файлу: {file_path}")
            return text
    except FileNotFoundError:
        print(f"Файл {file_path} не знайдено. Переконайтеся, що файл існує.")
        return None
    except Exception as e:
        print(f"Помилка при зчитуванні PDF-файлу {file_path}: {e}")
        return None


def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


def store_chunks_in_faiss(chunks, model_name="all-MiniLM-L6-v2"):
    try:
        print(f"Завантаження моделі {model_name}...")
        model = SentenceTransformer(model_name)
        print("Генерація ембеддингів для кожного чанку...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        print("Створення FAISS індексу...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f"Успішно збережено {len(chunks)} чанків в FAISS.")
        return index, chunks
    except Exception as e:
        print(f"Помилка при створенні FAISS: {e}")
        return None


def search_faiss(index, chunks, query, model_name="all-MiniLM-L6-v2", top_k=50):
    try:
        model = SentenceTransformer(model_name)
        print(f"Генерація ембеддингу для запиту: '{query}'")
        query_embedding = model.encode([query])
        print("Пошук у FAISS...")
        distances, indices = index.search(query_embedding, top_k)
        results = [chunks[i] for i in indices[0]]
        return results
    except Exception as e:
        print(f"Помилка при пошуку в FAISS: {e}")
        return []


def get_gemini_response(context_chunks, user_query):
    try:
        context = "\n".join(context_chunks)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_api_key)
        prompt_text = f"Контекст:\n{context}\nДай відповідь на запит: {user_query}. Для кожної відповіді дай номер сторінки з інформацією звідки це взято!"
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | llm | StrOutputParser()
        llm_response = chain.invoke({})
    except Exception as e:
        llm_response = str(e)
    return llm_response


def main(file_path, user_query):
    # long_text = read_text_from_file(file_path)
    long_text = read_text_from_pdf(file_path)
    chunks = split_text_into_chunks(long_text)
    index, stored_chunks = store_chunks_in_faiss(chunks)
    if index is None:
        print("Не вдалося створити FAISS.")
        return
    results = search_faiss(index, stored_chunks, user_query)
    if results:
        print("Результати пошуку з фейсу успішні")
    else:
        print("Не знайдено релевантних чанків для запиту в фейс.")
    final_response = get_gemini_response(results, user_query)
    return final_response


if __name__ == '__main__':
    # file_path = "book.txt"
    file_path = "Інструкція-з-експлуатації-Renault-Fluence-та-Megane-3.pdf"
    user_query = ""
    response = main(file_path, user_query)
    print("###################")
    print(response)
