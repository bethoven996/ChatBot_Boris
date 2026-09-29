import os
import unicodedata
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


st.set_page_config(page_title="Boris - Farmacia", page_icon="💊")
st.title("💊 Farmacia Asistente - Boris")


def normalizar(texto: str) -> str:
    """minúsculas y sin tildes, para comparar nombres de forma robusta
    (así 'ibupirac' encuentra 'Ibupirac' y 'diclofenac sodico' encuentra
    'Diclofenac sódico')."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


@st.cache_resource
def iniciar_sistema():
    # 1. Cargar datos.
    # metadata_columns saca "Nombre Comercial" y "Droga / Principio Activo" a
    # doc.metadata para poder filtrar por ellos directamente (sin depender de
    # similitud semántica). content_columns mantiene TODAS las columnas
    # también en el texto, así el LLM sigue viendo el nombre del producto.
    columnas_csv = [
        "Droga / Principio Activo", "Nombre Comercial", "Presentación",
        "Laboratorio", "Precio de Venta al Público (PVP)",
        "Fecha de extracción", "Fuente",
    ]
    loader_csv = CSVLoader(
        file_path="datos.csv",
        encoding="utf-8",
        csv_args={"delimiter": ";"},
        metadata_columns=["Nombre Comercial", "Droga / Principio Activo"],
        content_columns=columnas_csv,
    )
    docs_csv = loader_csv.load()
    loader_txt = TextLoader(file_path="obras_sociales.txt", encoding="utf-8")
    docs_txt = loader_txt.load()

    # Filtro de filas duplicadas por las dudas (inofensivo si no hay ninguna)
    vistos = set()
    docs_csv_unicos = []
    for doc in docs_csv:
        clave = doc.page_content.strip()
        if clave not in vistos:
            vistos.add(clave)
            docs_csv_unicos.append(doc)

    # Trocear solo el texto libre (obras_sociales.txt); las filas del CSV ya
    # son documentos completos y cortos, no hace falta trocearlas.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_txt_split = splitter.split_documents(docs_txt)

    # 2. Vector DB (para preguntas que NO nombran un producto puntual,
    # ej. "algo para el dolor de cabeza", o preguntas sobre obras sociales)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs_csv_unicos + docs_txt_split, embeddings)
    retriever_vectorial = vector_db.as_retriever(search_kwargs={"k": 6})

    # Nombres de producto conocidos, ordenados de más largo a más corto para
    # priorizar coincidencias específicas (ej. que "Ibupirac 600" se detecte
    # antes que el genérico "Ibupirac" cuando ambos aparecen en la pregunta).
    nombres_comerciales = sorted(
        {d.metadata["Nombre Comercial"] for d in docs_csv_unicos if d.metadata.get("Nombre Comercial")},
        key=len,
        reverse=True,
    )

    def buscar_hibrido(query: str):
        """Si la pregunta nombra un producto conocido, trae TODAS sus
        presentaciones (búsqueda exacta, 100% confiable). Además siempre
        suma los resultados de búsqueda semántica, para no perder contexto
        de obras sociales u otras preguntas más abiertas."""
        query_norm = normalizar(query)
        texto_restante = query_norm
        encontrados = []
        for nombre in nombres_comerciales:
            if normalizar(nombre) in texto_restante:
                encontrados.append(nombre)
                # lo "tapamos" para que nombres más cortos que ya quedaron
                # incluidos en uno más largo no vuelvan a matchear aparte
                texto_restante = texto_restante.replace(normalizar(nombre), " ")

        docs_resultado = []
        if encontrados:
            docs_resultado = [
                d for d in docs_csv_unicos
                if d.metadata.get("Nombre Comercial") in encontrados
            ]

        docs_semanticos = retriever_vectorial.invoke(query)
        ya_incluidos = {d.page_content for d in docs_resultado}
        for d in docs_semanticos:
            if d.page_content not in ya_incluidos:
                docs_resultado.append(d)
                ya_incluidos.add(d.page_content)

        return docs_resultado

    retriever_hibrido = RunnableLambda(buscar_hibrido)

    # 3. LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.4,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # 4. Prompts y Cadenas.
    # OJO: la clave del historial tiene que llamarse "chat_history" a secas.
    # create_history_aware_retriever revisa por dentro esa clave literal para
    # decidir si reformula la pregunta con el historial; con cualquier otro
    # nombre (como "historial_chat"), nunca la reformula y manda la pregunta
    # pelada al buscador, perdiendo todo el contexto de la conversación.
    prompt_memoria = ChatPromptTemplate.from_messages([
        ("system", """Teniendo en cuenta el historial de la conversación y la nueva pregunta del usuario, reformulá la pregunta para que se entienda por sí sola.
IMPORTANTE: si en el historial ya se mencionó una marca, principio activo o miligramos específicos (por ejemplo "Ibupirac 600"), conservalos textualmente en la pregunta reformulada.
NO respondas la pregunta, solo reescribila."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    buscador_con_memoria = create_history_aware_retriever(llm, retriever_hibrido, prompt_memoria)

    prompt_personalizado = ChatPromptTemplate.from_messages([
        ("system", """Sos Boris, el empleado de mostrador de una farmacia. Sos cálido y amable, pero muy directo y conciso. Hablás como un argentino real.
        REGLAS DE ORO:
        1. Sé breve y respondé exactamente lo que te preguntan.
        2. No repitas horarios ni dirección salvo que te lo pregunten específicamente.
        3. Si hacés cuentas con obras sociales, da el resultado final y el descuento, sin mostrar la fórmula.
        4. Si un producto tiene varias presentaciones o tamaños en el contexto, listalas brevemente (con precio de cada una) en vez de decir que no tenés la información.
        5. Nunca digas que no tenés información si el producto mencionado aparece en el contexto, aunque sea en una presentación distinta a la exacta que pidieron.
        Contexto de la base de datos:
        {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt_personalizado)
    qa_chain = create_retrieval_chain(buscador_con_memoria, combine_docs_chain)

    return qa_chain

# Arrancamos el cerebro de Boris (¡Esto demora unos segundos solo la primera vez!)
qa_chain = iniciar_sistema()


# Si es la primera vez que entramos, creamos las memorias vacías
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.mensajes_pantalla = [{"role": "assistant", "content": "¡Hola! ¿Cómo estás? Soy Boris, el asistente de la farmacia. ¿En qué te puedo ayudar hoy?"}]

# Dibujar los mensajes anteriores en la pantalla
for msg in st.session_state.mensajes_pantalla:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if pregunta := st.chat_input("Escribí tu pregunta acá..."):
    # 1. Mostrar lo que el usuario escribió
    with st.chat_message("user"):
        st.markdown(pregunta)
    st.session_state.mensajes_pantalla.append({"role": "user", "content": pregunta})

    # 2. Mostrar a Boris "Pensando" y procesar
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los sistemas de la farmacia..."):
            respuesta = qa_chain.invoke({
                "input": pregunta,
                "chat_history": st.session_state.chat_history
            })
            texto_respuesta = respuesta['answer']
            st.markdown(texto_respuesta)

    # 3. Guardar la respuesta de Boris en la pantalla
    st.session_state.mensajes_pantalla.append({"role": "assistant", "content": texto_respuesta})

    # 4. Guardar los datos en el historial de memoria real (para LangChain)
    st.session_state.chat_history.append(HumanMessage(content=pregunta))
    st.session_state.chat_history.append(AIMessage(content=texto_respuesta))