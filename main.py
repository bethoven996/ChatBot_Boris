import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# ¡ATENCIÓN! Tu clave secreta (cuidala mucho)
os.environ["GROQ_API_KEY"] = "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"

print("🚀 INICIANDO ARQUITECTURA RAG (Nivel 3 con Groq) 🚀")
print("-" * 50)

# 1. INGESTA DE DATOS (Lectura de tu Excel)
print("1️⃣ Leyendo la base de datos de medicamentos...")
df = pd.read_excel('medicamentos_argentina.xlsx')

# Creamos una columna unificada de texto para que la IA lea toda la info junta
df['contexto_ia'] = df.apply(
    lambda row: f"El medicamento {row['Nombre Comercial']} está compuesto por {row['Droga / Principio Activo']}. "
                f"Su presentación es {row['Presentación']} fabricado por el laboratorio {row['Laboratorio']}. "
                f"El precio de venta al público actual es de ${row['Precio de Venta al Público (PVP)']}.", 
    axis=1
)

# Convertimos el DataFrame en "Documentos" que LangChain pueda entender
loader = DataFrameLoader(df, page_content_column="contexto_ia")
documentos = loader.load()

# 2. MOTOR DE BÚSQUEDA VECTORIAL (Embeddings + ChromaDB)
print("2️⃣ Creando el espacio vectorial (esto convierte texto en matemáticas)...")
# Usamos un modelo open-source ligero para crear los vectores
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Guardamos los documentos en nuestra base de datos vectorial local (Chroma)
vector_db = Chroma.from_documents(documentos, embeddings)

# 3. CONECTANDO EL CEREBRO ONLINE (Groq - ACTUALIZADO)
print("3️⃣ Conectando al cerebro en la nube ultrarrápida (Groq - Llama 3.1)...")
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant"  # ¡ESTE ES EL CAMBIO CLAVE!
)

# 4. EL ORQUESTADOR (RetrievalQA Chain)
# Esto une la base de datos con el cerebro en la nube
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}), # Trae los 3 medicamentos más relevantes
    return_source_documents=True
)

print("\n" + "="*50)
print("🤖 ASISTENTE FARMACÉUTICO RAG ACTIVO 🤖")
print("Podés hacer preguntas complejas como: 'Me duele la cabeza y no quiero gastar más de 4000 pesos, ¿qué me recomendás?'")
print("="*50)

# 5. EL CHATBOT
while True:
    pregunta = input("\n🗣️ Vos: ").strip()
    
    if pregunta.lower() in ['salir', 'chau']:
        print("🤖 Boris: ¡Nos vemos! Apagando sistemas.")
        break
    
    print("🤖 Boris: (Pensando, buscando en la base vectorial y redactando...)")
    
    # Aquí ocurre la magia: busca los vectores y le pasa el contexto al LLM
    respuesta = qa_chain.invoke({"query": pregunta})
    
    print(f"\n🤖 Boris: {respuesta['result']}")