#Servidor Streamlit
import streamlit as st
#Parsing respuestas bot/humano
from htmlTemplates import css, bot_template, user_template

#Capturar variables de entorno
# from dotenv import dotenv_values
#Parsing PDF
from PyPDF2 import PdfReader
#Crear chunks válidos para el LLM de OpenAI
from langchain.text_splitter import CharacterTextSplitter

#Creación de embeddings a partir de los chunks. Requiere instalar tiktoken
from langchain.embeddings import OpenAIEmbeddings
#Modelo LLM basado en chats
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

#Conexión con PineCone client (LangChain)
from langchain.vectorstores import Pinecone

#Descargar de HTML de BGG
from urllib.request import urlopen

#Parsing HTML de BGG. Requiere installar html5lib para parser de HTML
from bs4 import BeautifulSoup

#Libreria PineCone Client
import pinecone
#Libreria OpenAI Client
import openai
#Libreria Sistema
import sys

# Función que parsea objeto PDF (subido por el usuario)
#    - Return: bggID: Extracto a partir del nombre generado por el usuario <BGG_ID>_<Type>.pdf. Ej: bggID de 342942_FAQ.pdf o 342942.pdf es 342942
#    - gameTitle: Se conecta a la URL https://boardgamegeek.com/boardgame/'+bggID y parsea el título (primer string antes de |). Página de error empiezan por BoardGameGeek
#    - text: Extracción de texto sin formato a partir del PDF creado por el usuario
# En el caso de error (juego no encontrado en BGG, error parseando PDF...) devuelve 0,'',''

def get_pdf_text(pdf):
    text = ""
    try:
        # Vamos a obtener los metadatos de la BGG del fichero generado: 167791_FAQ.pdf
        # Primero vemos si el nombre contiene un _
        split = pdf.name.find('_')
        if split > 0:
            breakpoint = split
        else:
            # Si no tiene _, cogemos el nombre completo del fichero (sin .pdf)
            breakpoint = pdf.name.find('.')

        # Obtemeos el ID de BGG a partir del nombre del fichero
        bggID = pdf.name[0:breakpoint]
        url = 'https://boardgamegeek.com/boardgame/'+bggID
        # Vamos a confirmar que el ID existe. 
        # Usamos BeautifulSoup module para leer el HTML de la página de BGG asociada al nombre
        soup = BeautifulSoup(urlopen(url), features="html5lib")
        # Extraemos el título
        BGGtitle = soup.title.get_text()
        # Y nos quedamos con la primera parte antes de |
        guion = BGGtitle.find(' |')
        gameTitle = BGGtitle[0:guion]
        if gameTitle == "BoardGameGeek":
            # BGG ha devuelto una pagina de error generica. El foramto del fichero debe ser BGGID.pdf
            with st.sidebar:
                st.warning("No se ha encontrado el ID:"+bggID+" en la BGG.")
        else:
            # Todo correcto, vamos a leer el PDF y guardamos el texto en la variable text 
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
            return bggID, gameTitle, text
    except Exception:
        type, value, traceback = sys.exc_info()
        with st.sidebar:
            st.warning("Error procesando "+pdf.name)
    return 0,'',''

# Funcion que a partir del texto completo del pdf, lo divide en token válidos para luego procesarlos en OpenAI
# Usamos chunks de 1000 caracteres con un overlap de 100
# Return: Chunks en los que se divide el texto original
def get_text_chunks(text):
    # Usamos la libreria de LangChain CharacterTextSplitter. Easy
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Función que CREA embeddings en PineCone con chunks de texto que vienen del PDF
# Usa variables de entorno para conectarse a PineCone. Los embeddings los genera en OpenAI a través del Open AI KEY del formulario web
# El Namespace tiene la estructura de <prefix-><nombre del juego> donde <nombre del juego> es el nombre en la BGG asociado al ID del PDF
# Return: vectorstore creado para usar búsquedas con LLM o False si existe un error 
def create_vectorstore(text_chunks, namespace):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openAI_user_key)
        vectorstore =  Pinecone.from_texts(api_key=st.secrets["PINECONE_API_KEY"],
                                           environment=st.secrets["PINECONE_ENVIROMENT"],
                                           texts=text_chunks, 
                                           embedding=embeddings, 
                                           index_name=st.secrets["PINECONE_INDEXNAME"],
                                           namespace=namespace)
        return vectorstore
    except Exception:
        type, value, traceback = sys.exc_info()
        with st.sidebar:
            st.error("Error procesando el fichero", icon="🚨")
        return False

# Función que retorno un vectorstore con los embeddings del juego namespace
# El Namespace tiene la estructura de <prefix-><nombre del juego> donde <nombre del juego> es el nombre en la BGG asociado al ID del PDF
# Return: vectorstore asociado al juego para usar búsquedas con LLM o False si existe un error 
def get_vectorstore(namespace):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openAI_user_key)
        vectorstore =  Pinecone.from_existing_index(embedding=embeddings,
                                                    index_name=st.secrets["PINECONE_INDEXNAME"],
                                                    namespace=namespace)
        return vectorstore
    except Exception:
        type, value, traceback = sys.exc_info()
        with st.sidebar:
            st.error("Error accediendo a la base de datos", icon="🚨")
        return False

# Función que crea objeto ConversationalRetrievalChain de LangChain para hacer preguntas sobre los embeddings del VectorStore
# ConversationalRetrievalChain tiene almacenado historial de búsquedas para dar contexto sobre nuevas búsquedas
# Return: conversation_chain la conversación o False si existe un error 
def get_conversation_chain(vectorstore):
    try:
        # Creamos el objeto LLM con el OpenAI UserKey del formulario web
        llm = ChatOpenAI(openai_api_key=st.session_state.openAI_user_key)
        # Creamos un Buffer con el historico de consultas del usuario (vacio)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        # Creamos objeto de coneversación basado en OpenAI
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception:
        type, value, traceback = sys.exc_info()
        with st.sidebar:
            st.error("Error realizando la consulta a OpenAI", icon="🚨")
        return False
    
# Función que invoca el objeto ConversationalRetrievalChain para hacer preguntas. Las preguntas y respuestas, se las envia a Streamlit para pintarlas en pantalla
# Return: False si ha habido error invocando ConversationalRetrievalChain
def handle_userinput(user_question):
    try:
        # Lanzamos la pregunta al objeto ConversationalRetrievalChain con la pregunta del usuario
        response = st.session_state.conversation({'question': user_question})
        # Añadimos la respuesta al histórico de chats
        st.session_state.chat_history = response['chat_history']
                
        # Por defecto, el orden es primero el primer mensaje, para pintarlo en orden inverso, hacemos copia del chat_history y le damos la vuelta
        reverseChatHistory = st.session_state.chat_history[::-1]
        userMessage = ""
        for i, message in enumerate(reverseChatHistory):
            # Los mensajes pares son los mensajes del usaurio, los impares las respuesta. Queremos que se muestre por parejas, pero primero la pregunta (arriba) y despues la respuesta (abajo)
            if i % 2 == 0:
                userMessage = message
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
                st.write(user_template.replace(
                    "{{MSG}}", userMessage.content), unsafe_allow_html=True)
    except Exception:
        type, value, traceback = sys.exc_info()
        with st.sidebar:
            st.error("Error a la hora de consultar en OpenAI. Confirma el valor de OpenAI Key", icon="🚨")
        return False

# Funcion que valida si el OpenAI Key del usaurio es válido
#  Para ello, envia un comando "say hi!" a OpenAI y si tiene respuesta del LLM retorna True
#  Si no se ha podido invocar el "say hi!" o hay fallo en al respuesta, seguramente el OpenAI Key no sea válido. Se devuelve False
def checkOpenAIKey(openAI_user_key):
    st.session_state.openAI_user_key = openAI_user_key  

    # Comentado para no consumir tokens en OpenAI. Descomentar para producción
    # Lanzamos un Hello World a OpenAI con el ID del input
    #try:
    #    from langchain.llms import OpenAI
    #    llm = OpenAI(openai_api_key=st.session_state.openAI_user_key)
    #    answer = llm.predict("say hi!")
    #except Exception:
    #    st.warning("OpenAI UserKey no es válido")
    #    st.session_state.openAI_user_key = None
    #    checkOpenAIKey = False

    checkOpenAIKey = True
    openai.api_key = openAI_user_key

# Función que cargar todos los juegos almacenados en PineCone para devolver la lista de juegos disponibles
# Return: Lista de juegos (array) o False si ha encontrado algun error. Puede devolver una lista vacia ([]) si no se encuetra en Pinecone ningun namespace que empiece por el prefix configurado
def load_games():
    try:
        # Cargamos las variables de entorno relacionadas con PineCone

        #Configuracion gestionado por Streamlit. Para debugging en local crear fichero en .streamlit/secrests.toml
        pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], 
                      environment=st.secrets["PINECONE_ENVIROMENT"])
        
        pineconeIndexName = st.secrets["PINECONE_INDEXNAME"]
        if pineconeIndexName:
            # Si no esta el indice creado, se crea en PineCone. En este caso, la lista de juegos estará vacia
            if pineconeIndexName not in pinecone.list_indexes():
                    pinecone.create_index(pineconeIndexName, 
                                        dimension=1536, 
                                        metric='cosine', 
                                        pods=1, 
                                        replicas=1, 
                                        pod_type='p1.x1')
            
            pineconeIndex = pinecone.Index(index_name=pineconeIndexName)
            # Retorna el listado de Namespace asocaidos al indice y cuenta configurado de Pinecone
            index_stats_response = pineconeIndex.describe_index_stats()
            pineconeNamespaces = index_stats_response.get("namespaces")

            # Registramos en gamesNamepsaces los gamespaces asociados al prefijo de juegos da2 (configurable en .env)
            gamesNamepsaces = []
            pineConePrefixLen = len(st.secrets["PINECONE_PREFIX"])
            for namespace in pineconeNamespaces:
                # Si el nombre del namespace (ej: 'gptda2-Deep Sea Adventure') empieza por el prefix configurado ('gptda2-')
                if namespace.startswith(st.secrets["PINECONE_PREFIX"]):
                    # Se añade a la lista de juegos con embeddings generado
                    gameName = namespace[pineConePrefixLen:]
                    gamesNamepsaces.append(gameName)

            # Antes de devolver la lista de juegos, se ordena alfabeticamente
            gamesNamepsaces.sort()
            return gamesNamepsaces
        else:
            return False
    except Exception:
        type, value, traceback = sys.exc_info()
        with st.sidebar:
            st.error("Error al cargar la base de datos. Recarga la página", icon="🚨")
        return False


# Funcion principal de StreamLit. 
def main():    
    # Vamos a crear la pagina donde presentar toda la UI
    st.set_page_config(page_title="GPTDa2",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Las variables de sesion se inicializan a vacia
    # Registra el objeto ConversationalRetrievalChain asociado al OpenAI Key del formulario de busqueda
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    # Almancena el historico de busqueas realizado
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    # Almacena el OpenAI Key del input web validado con la funcion checkOpenAIKey
    if "openAI_user_key" not in st.session_state:
        st.session_state.openAI_user_key = None
    # Retorna el listado de juegos encontrado en Pinecone ya disponibles para busqueda
    if "gameList" not in st.session_state:
        st.session_state.gameList = None
    # Devuelve el indice de la lista de juegos activo
    if "gameListIndex" not in st.session_state:
        st.session_state.gameListIndex = 0
    # Devuelve el juego sobre el que seleccionado sobre el que se harán las consultas
    if "selectedGame" not in st.session_state:
        st.session_state.selectedGame = None

    with st.sidebar:
        # En el sidebar del HTML, se crea un input para que el usuario inserte su propio OpenAI Key
        openAI_user_key = st.text_input("Pon aqui tu OpenAI API Key:",
                                        help="Crea una cuenta de OpenAI e inserta el valor de KEY: https://platform.openai.com/account/api-keys")
        # Si el usuario inserta un valor, se valida con la función checkOpenAIKey a ver si el OpenAI Key es valido. 
        if openAI_user_key:
            checkOpenAIKey(openAI_user_key)

    # El resto de la interfaz no se muestra hasta que no se haya detectado un OpenAI Key válido      
    if st.session_state.openAI_user_key:
        st.header("GPT-Da2:")

        #Configuracion gestionado por Streamlit. Para debugging en local crear fichero en .streamlit/secrests.toml
        pineConePrefix = st.secrets["PINECONE_PREFIX"]

        # Si es la primera vez que se carga la pagina (gamelist vacia), se buscan los juegos disponibles en Pinecone
        if not st.session_state.gameList:
            games = load_games()
            # Se inicializa la lista de juegos, ya no deberia volver a actualizarse otra vez en esta sesión de usuario
            st.session_state.gameList = games

        # Si hay juegos en la lista de juegos
        if st.session_state.gameList:
            with st.sidebar:        
                # Se muestra la lista de juegos en un selector con la lista de juegos almacenado en el objeto de streamlit de sesion gameList
                st.selectbox("Elige un juego", 
                    options=st.session_state.gameList,
                    help="Nombre del juego en BGG sobre el que hacer las preeguntas",
                    key="GameSelector",
                    index=st.session_state.gameListIndex)
                
        # Se da la opción de subir nuevos ficheros de reglas/FAQ/..., tanto si la lista de juegos original esta vacia como si ya habia elementos
        with st.sidebar:           
            # Se crea el objeto de Streamlit File_Uploader para subir PDF y se da información al usuario de los formatos válidos soportados para el nombre del fichero
            pdf_docs = st.file_uploader("O Si no está el juego que buscas, sube nuevas reglas en PDF y pulsa el botón 'Procesar'", 
                                    accept_multiple_files=False, 
                                    help="El fichero debe ser un pdf con el formato <ID BGG>_<tipo>.pdf. Ej: 167791_FAQ.pdf")

            # Si se detecta que el usuario ha pulsado sobre el botón "Process"
            if st.button("Process"):
                # Si el usuario ha insertado previamente un fichero de texto
                if pdf_docs:
                    # Feedback con spinner
                    with st.spinner("Processing"):
                        # Se extraen el id, name, raw_text del fichero PDF con la funcion get_pdf_text
                        id, name, raw_text = get_pdf_text(pdf_docs)

                        # Si el fichero tenia información de texto
                        if raw_text:
                            # Se muestra un feedback al usaurio de que se está procesando el fichero PDF con el nombre del juego asocaido a BGG
                            with st.sidebar:
                                st.info("Procesando el juego: '"+name+"'")
                           
                            # Se crean los chunks a partir del texto plano del PDF subido
                            text_chunks = get_text_chunks(raw_text)

                            # Creamos en PineCone los embeddings asociados a los chunks con el namespace <Prefix-><Nombre del juego en BGG>
                            vectorstore = create_vectorstore(text_chunks, pineConePrefix+name)

                            # Si se ha creado correctamente en Pinecone los embeddings del texto
                            if vectorstore:
                                # Creamos el objetivo ConversationalRetrievalChain y lo almacenamos en la variable de seion de Streamlit conversation
                                st.session_state.conversation = get_conversation_chain(vectorstore)

                                # Si se ha creado correctamente el objeto ConversationalRetrievalChain
                                if st.session_state.conversation:  
                                    # Miramos si el juego ya existia en la lista de juegos creados
                                    gameList = st.session_state.gameList 
                                    # Si no existe, se crea nueva entrada y se actualiza la lista de juegos
                                    if not name in gameList:
                                        gameList.append(name)
                                    # La lista de juegos de ordena
                                    gameList.sort()
                                    st.session_state.gameList = gameList
                                    # Y se devuelve el indice donde debe posicionarse el select para marcar el juego asociado al fichero subido
                                    # Esta logica aplica tanto si el juego es nuevo o si ya existia
                                    st.session_state.gameListIndex = gameList.index(name)

                                    with st.sidebar:
                                        st.success('Fichero subido correctamente!', icon="✅")
                                    st.experimental_rerun()
                                else:
                                    st.session_state.conversation = None

                else:
                    with st.sidebar:
                        st.warning("Sube primero las reglas/ayuda/faq del juego en pdf con el nombre BGGID_<tipo>.pdf")

        # Si existen juegos en la lista de juegos, se crea un input para la conversacion con el usuario
        if st.session_state.gameList:
            user_question = st.text_input("Preguntame sobre las reglas de: '"+st.session_state.GameSelector+"'")
            # Si el usuario ha introducido un criterio de búsqueda
            if user_question:
                # Si ha modificado el juego sobre el que realizar la busqueda, creamos unas nuevas variables de entorno de StreamList asociados a la conversacion
                # En el caso que se haya iniciado la sesion en StreamList, selectedGame es None y por lo tanto se crea un setup con el primer juego seleccionado de la lista
                if not(st.session_state.selectedGame == st.session_state.GameSelector):
                    # Se crea el VectorStore asociado al juego de la lista de juegos
                    vectorstore = get_vectorstore(pineConePrefix+st.session_state.GameSelector)
                    # Si se ha creado correctmente, se inicializan todas las variables de sesion de StreamLit asociados con el juego seleccionado
                    if vectorstore:
          #              st.session_state.chat_history = None
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.session_state.selectedGame = st.session_state.GameSelector
                    else:
                        st.session_state.selectedGame = None

                # Tanto si se ha detectado una seleccion de juego nuevo, como si la consulta es del mismo juego de la conversacion existente
                # Se comprueba si hay un juego seleccionado (puede estar vacio si ha habido un error en la conexion con Pinecone)
                if st.session_state.selectedGame:
                    # La pregunta del usuario se manda a OpenAI y tanto la pregunta como respuesta se muestran en pantalla
                    handle_userinput(user_question)
        else:
            with st.sidebar:        
                st.warning("Base de datos de juegos vacia")

if __name__ == '__main__':
    main()
