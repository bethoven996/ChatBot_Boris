import os
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

print("1yendo medicamentos y reglas de obras sociales...")
loader_csv = CSVLoader(file_path="datos.csv", encoding="utf-8", csv_args={"delimiter": ";"})
docs_csv = loader_csv.load()

loader_txt = TextLoader(file_path="obras_sociales.txt", encoding="utf-8")
docs_txt = loader_txt.load()

documentos = docs_csv + docs_txt
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documentos_split = splitter.split_documents(documentos)

print("2️⃣ Creando el espacio vectorial...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(documentos_split, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 6})




print("3️⃣ Conectando al cerebro en la nube ultrarrápida (Groq)...")
llm = ChatGroq(
    model="openai/gpt-oss-20b", 
    temperature=0.4,
    api_key="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",
)



prompt_memoria = ChatPromptTemplate.from_messages([
    ("system", "Teniendo en cuenta el historial de la conversación y la nueva pregunta del usuario, reformula la pregunta para que se entienda por sí sola, sin necesidad del historial. NO la respondas, solo reescribila."),
    MessagesPlaceholder("historial_chat"),
    ("human", "{input}")
])

buscador_con_memoria = create_history_aware_retriever(llm, retriever, prompt_memoria)

prompt_personalizado = ChatPromptTemplate.from_messages([
    ("system", """Sos Boris, el empleado de mostrador de una farmacia. Sos cálido, amable y directo. Hablás como un argentino real.

INSTRUCCIONES CLAVE:
1. Sé breve y respondé con precisión.
2. Si el cliente menciona una presentación (ej: "la de 1000" o "de 50"), buscá esa cantidad o miligramos en tu contexto y dale el precio de inmediato.
3. No vuelvas a preguntar qué presentación quiere si el cliente ya te la acaba de decir.
4. Si hace cuentas con obras sociales, da el resultado final y el descuento de forma simple.

Contexto de la base de datos:
{context}"""),
    MessagesPlaceholder("historial_chat"),
    ("human", "{input}")
])

combine_docs_chain = create_stuff_documents_chain(llm, prompt_personalizado)
qa_chain = create_retrieval_chain(buscador_con_memoria, combine_docs_chain)

print("\n" + "=" * 50)
print("🤖 ASISTENTE FARMACÉUTICO RAG ACTIVO 🤖")
print("=" * 50)

SALUDOS = ["hola", "buenas", "buen dia", "buen día", "buenas tardes", "buenas noches", "que tal", "qué tal", "como estas", "cómo estás"]
def es_saludo(texto: str) -> bool:
    return any(s in texto.lower().strip() for s in SALUDOS) and len(texto.split()) <= 4


historial = []

while True:
    pregunta = input("\n🗣️ Vos: ").strip()

    if pregunta.lower() in ["salir", "chau"]:
        print("🤖 Boris: ¡Nos vemos! Apagando sistemas.")
        break

    if es_saludo(pregunta):
        print("\n🤖 Boris: ¡Hola! ¿Cómo estás? Soy Boris, el asistente de la farmacia. ¿En qué te puedo ayudar hoy?")
        continue

    # Le pasamos la pregunta Y el historial de la charla
    respuesta = qa_chain.invoke({
        "input": pregunta,
        "historial_chat": historial
    })

    print(f"\n🤖 Boris: {respuesta['answer']}")

    historial.append(HumanMessage(content=pregunta))
    historial.append(AIMessage(content=respuesta["answer"]))