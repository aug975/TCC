import os
import json
import warnings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
from langchain.schema import SystemMessage, AIMessage, HumanMessage  
from langchain_community.vectorstores import FAISS  
from langchain_community.document_loaders import DirectoryLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.memory import ConversationSummaryBufferMemory

# Evita conflitos do OpenMP!!
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Desativa avisos de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)

# !!
OPENAIKEY = [key]

# Configuração do modelo
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=OPENAIKEY)  

# Diretório contendo os arquivos personalizados
directory_path = "c:/projects/mackbot/base"

# Criando um vetor de documentos para priorizar respostas
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  

# Criando embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAIKEY)  

# Carregando documentos do diretório
document_loader = DirectoryLoader(directory_path, glob="**/*.*")  
documents = document_loader.load()

if not documents:
    raise ValueError("Nenhum documento foi encontrado no diretório especificado.")

# Processando e indexando os documentos
texts = text_splitter.split_documents(documents)  

# Evita criar múltiplas instâncias de FAISS
if "vectorstore" not in globals():
    vectorstore = FAISS.from_documents(texts, embeddings)

# Inicializando a memória para armazenar contexto da conversa
memory = ConversationSummaryBufferMemory(
    memory_key="chat_history",
    llm=llm,
    max_token_limit=1000,
    return_messages=True
)

# Loop de interação com o usuário
print("Chatbot inicializado. Digite 'sair' para encerrar.")
while True:
    user_input = input("Você: ")  
    if user_input.lower() == "sair":  
        break
    
    # Buscando informações relevantes do índice vetorial
    relevant_docs = vectorstore.similarity_search(user_input, k=3)  
    context = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
    
    # Criando um prompt incluindo o contexto
    messages = memory.load_memory_variables({})["chat_history"] + [  
        SystemMessage(content="Use o seguinte contexto ao responder: " + context),  
        HumanMessage(content=user_input)  
    ]
    
    # Gerando resposta
    response = llm.invoke(messages)  
    
    # Armazenando contexto
    memory.save_context({"input": user_input}, {"output": response.content})  
    
    print("Chatbot:", response.content)
