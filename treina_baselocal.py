# pip install python-magic-bin

import os
import magic
from langchain_community.vectorstores import FAISS  # Biblioteca para armazenamento e busca eficiente de embeddings  
from langchain_openai import OpenAIEmbeddings  # Geração de embeddings para comparação semântica
from langchain_community.document_loaders import DirectoryLoader  # Carregamento de documentos de vários formatos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Fragmentação de textos para otimizar indexação

OPENAIKEY = [key]

# Diretório contendo os arquivos personalizados
directory_path = "c:/projects/mackbot/base"
faiss_path = "faiss_index"  # Caminho para armazenar o índice FAISS

embeddings = OpenAIEmbeddings(openai_api_key=OPENAIKEY)

# Criando um vetor de documentos para priorizar respostas
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Inicializando a detecção de tipo de arquivo
mime = magic.Magic(mime=True)

# Lista de tipos de arquivos suportados
supported_types = {"application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/html", "text/plain"}

# Carregando documentos do diretório
document_loader = DirectoryLoader(directory_path, glob="**/*.*")
documents = document_loader.load()
num_files = len(documents)

# Filtrando arquivos suportados antes de processar
filtered_documents = []
for doc in documents:
    file_type = mime.from_file(doc.metadata['source'])
    if file_type in supported_types:
        filtered_documents.append(doc)
    else:
        print(f"Ignorando arquivo não suportado: {doc.metadata['source']} (Tipo: {file_type})")

# Processando e indexando os documentos
texts = text_splitter.split_documents(filtered_documents)
if "vectorstore" not in globals():
    vectorstore = FAISS.from_documents(texts, embeddings)

# Salvando a base de conhecimento localmente para futuras execuções
vectorstore.save_local(faiss_path)
print(f"Treinamento concluído! {len(filtered_documents)} documentos foram processados e indexados para futuras consultas.")
