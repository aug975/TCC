import os
import json
import hashlib
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Definição da chave da OpenAI
OPENAIKEY = [key]

# Diretório contendo os arquivos personalizados
directory_path = "c:/projects/mackbot/base"
faiss_path = "faiss_index"  # Caminho para armazenar o índice FAISS
hashes_file = "indexed_hashes.json"  # Arquivo para registrar os hashes dos arquivos indexados

# Inicializa os embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAIKEY)

# Criando um vetor de documentos para priorizar respostas
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Carregando documentos do diretório
document_loader = DirectoryLoader(directory_path, glob="**/*.*")
documents = document_loader.load()

# Carrega o arquivo de hashes existentes (se existir)
if os.path.exists(hashes_file):
    with open(hashes_file, "r") as f:
        indexed_hashes = set(json.load(f))
else:
    indexed_hashes = set()

# Função para gerar um hash SHA-256 do conteúdo de um documento
def hash_document(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

# Identifica documentos novos comparando os hashes
new_docs = []
new_hashes = set()

for doc in documents:
    doc_hash = hash_document(doc.page_content)
    if doc_hash not in indexed_hashes:
        new_docs.append(doc)
        new_hashes.add(doc_hash)

# Exibe apenas a quantidade de novos arquivos indexados
num_new_files = len(new_docs)
if num_new_files > 0:
    print(f"{num_new_files} novos arquivos foram indexados nesta execução.")
else:
    print("Nenhum novo documento foi indexado nesta execução.")

# Se houver novos arquivos, processa e adiciona ao índice FAISS
if new_docs:
    texts = text_splitter.split_documents(new_docs)

    # Carregar índice existente ou criar um novo
    if os.path.exists(faiss_path):
        print("Carregando índice FAISS existente...")
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(texts)
    else:
        print("Criando novo índice FAISS...")
        vectorstore = FAISS.from_documents(texts, embeddings)

    # Salva a base de conhecimento atualizada
    vectorstore.save_local(faiss_path)

    # Atualiza a lista de hashes indexados
    indexed_hashes.update(new_hashes)
    with open(hashes_file, "w") as f:
        json.dump(list(indexed_hashes), f)

    print("Base de conhecimento atualizada com sucesso!")

# Aguarda o usuário pressionar qualquer tecla para encerrar
input("Pressione qualquer tecla para sair...")
