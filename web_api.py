import os
import streamlit as st
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


st.set_page_config(page_title="Boris - Farmacia", page_icon="💊")
st.title("💊 Farmacia Asistente - Boris")


@st.cache_resource
def iniciar_sistema():
    # 1. Cargar datos
    loader_csv = CSVLoader(file_path="datos.csv", encoding="utf-8", csv_args={"delimiter": ";"})
    docs_csv = loader_csv.load()
    loader_txt = TextLoader(file_path="obras_sociales.txt", encoding="utf-8")
    docs_txt = loader_txt.load()
    documentos = docs_csv + docs_txt

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documentos_split = splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documentos_split, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 3. LLM
       # 3. LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=0.4,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    # 4. Prompts y Cadenas
    prompt_memoria = ChatPromptTemplate.from_messages([
        ("system", "Teniendo en cuenta el historial de la conversación y la nueva pregunta del usuario, reformula la pregunta para que se entienda por sí sola. NO la respondas, solo reescribila."),
        MessagesPlaceholder("historial_chat"),
        ("human", "{input}")
    ])
    buscador_con_memoria = create_history_aware_retriever(llm, retriever, prompt_memoria)

    prompt_personalizado = ChatPromptTemplate.from_messages([
        ("system", """Sos Boris, el empleado de mostrador de una farmacia. Sos cálido y amable, pero muy directo y conciso. Hablás como un argentino real.
        REGLAS DE ORO:
        1. Sé breve y respondé exactamente lo que te preguntan.
        2. No repitas horarios ni dirección salvo que te lo pregunten específicamente.
        3. Si hacés cuentas con obras sociales, da el resultado final y el descuento, sin mostrar la fórmula.
        4. Si hay varias opciones, preguntale al cliente cuál prefiere.
        Contexto de la base de datos:
        {context}"""),
        MessagesPlaceholder("historial_chat"),
        ("human", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt_personalizado)
    qa_chain = create_retrieval_chain(buscador_con_memoria, combine_docs_chain)
    
    return qa_chain

# Arrancamos el cerebro de Boris (¡Esto demora unos segundos solo la primera vez!)
qa_chain = iniciar_sistema()


# Si es la primera vez que entramos, creamos las memorias vacías
if "historial_langchain" not in st.session_state:
    st.session_state.historial_langchain = []
    st.session_state.mensajes_pantalla = [{"role": "assistant", "content": "¡Hola! ¿Cómo estás? Soy Boris, el asistente de la farmacia. ¿En qué te puedo ayudar hoy?"}]

# Dibujar los mensajes anteriores en la pantalla
for msg in st.session_state.mensajes_pantalla:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if pregunta := st.chat_input("Escribí tu pregunta acá..."):
    # 1. Mostrar lo que el usuario escribió
    with st.chat_message("user"):
        st.markdown(pregunta)
    # Guardarlo en la pantalla
    st.session_state.mensajes_pantalla.append({"role": "user", "content": pregunta})

    # 2. Mostrar a Boris "Pensando" y procesar
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los sistemas de la farmacia..."):
            respuesta = qa_chain.invoke({
                "input": pregunta,
                "historial_chat": st.session_state.historial_langchain
            })
            texto_respuesta = respuesta['answer']
            st.markdown(texto_respuesta)
    
    # 3. Guardar la respuesta de Boris en la pantalla
    st.session_state.mensajes_pantalla.append({"role": "assistant", "content": texto_respuesta})
    
    # 4. Guardar los datos en el historial de memoria real (para LangChain)
    st.session_state.historial_langchain.append(HumanMessage(content=pregunta))
    st.session_state.historial_langchain.append(AIMessage(content=texto_respuesta))