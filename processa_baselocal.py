import os
from langchain_community.vectorstores import FAISS # Biblioteca para armazenamento e busca eficiente de embeddings  
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Geração de embeddings para comparação semântica
from langchain_community.document_loaders import DirectoryLoader # Carregamento de documentos de vários formatos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Fragmentação de textos para otimizar indexaç

OPENAIKEY = [key]

# Diretório contendo os arquivos personalizados
directory_path = "c:/projects/mackbot/base"
faiss_path = "faiss_index"  # Caminho para armazenar o índice FAISS
embeddings = OpenAIEmbeddings(openai_api_key=OPENAIKEY) 

# Criando um vetor de documentos para priorizar respostas
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Carregando documentos do diretório

document_loader = DirectoryLoader(directory_path, glob="**/*.*")  
documents = document_loader.load()

# Processando e indexando os documentos
texts = text_splitter.split_documents(documents)
if "vectorstore" not in globals():
    vectorstore = FAISS.from_documents(texts, embeddings)

# Salvando a base de conhecimento localmente para futuras execuções
vectorstore.save_local(faiss_path)
print("Treinamento concluído! Os documentos foram processados e indexados para futuras consultas.")
