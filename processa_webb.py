import os
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS  # Biblioteca para armazenamento e busca eficiente de embeddings  
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Geração de embeddings para comparação semântica
from langchain_community.document_loaders import DirectoryLoader  # Carregamento de documentos de vários formatos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Fragmentação de textos para otimizar indexação
from langchain.schema import Document  # Importando a classe Document
from pathlib import Path
from urllib.parse import urlparse, urljoin  # Adicionando a importação correta

OPENAIKEY = [key]

# Endereço de pesquisa inicial
start_url = "https://blog.mackenzie.br/mercado-carreira/"
# Diretório contendo os arquivos de índice
faiss_path = "faiss_index"  # Caminho para armazenar o índice FAISS
embeddings = OpenAIEmbeddings(openai_api_key=OPENAIKEY) 

# Criando um vetor de documentos para priorizar respostas
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Função para buscar conteúdo de uma página web
def fetch_web_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove scripts e estilos
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        content = "\n".join(line for line in lines if line)

        return content[:3000]  # Limita para evitar excesso de informação
    except Exception as e:
        print(f"Erro ao buscar conteúdo da URL {url}: {e}")
        return ""

# Função para rastrear recursivamente links internos
def crawl_website(start_url, max_depth=2):
    visited = set()
    to_visit = [(start_url, 0)]  # (URL, profundidade)
    all_content = ""
    
    # Extraindo o caminho da URL base
    parsed_url = urlparse(start_url)
    base_domain = parsed_url.netloc
    base_path = parsed_url.path.rstrip('/')  # Caminho base (sem barra final)
    
    print(f"Caminho base da URL: {base_path}")
    
    while to_visit:
        url, depth = to_visit.pop(0)
        if url in visited or depth > max_depth:
            continue
        
        visited.add(url)
        print(f"Visitando: {url}")
        page_content = fetch_web_content(url)
        all_content += page_content + "\n\n"
        
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Encontrar todos os links na página
            for link in soup.find_all("a", href=True):
                href = link.get("href")
                full_url = urljoin(url, href)
                
                # Verifica se o link está dentro do caminho base ou no mesmo nível
                parsed_full_url = urlparse(full_url)
                if parsed_full_url.netloc == base_domain and parsed_full_url.path.startswith(base_path):
                    if full_url not in visited:
                        to_visit.append((full_url, depth + 1))
        except Exception as e:
            print(f"Erro ao processar links em {url}: {e}")

    return all_content

# Obtém todo o conteúdo do site
web_content = crawl_website(start_url)

# Criar um objeto Document com o conteúdo da página
document = Document(page_content=web_content)

# Processando e indexando os documentos
texts = text_splitter.split_documents([document])  # Passar uma lista de documentos

# Verifique se o índice FAISS já foi criado
if "vectorstore" not in globals():
    print("Criando o índice FAISS...")
    vectorstore = FAISS.from_documents(texts, embeddings)

# Certifique-se de que o diretório de destino existe
os.makedirs(faiss_path, exist_ok=True)

# Salvar o vetor de documentos no índice
try:
    vectorstore.save_local(faiss_path)
    print(f"Índice FAISS salvo no diretório: {os.path.abspath(faiss_path)}")
except Exception as e:
    print(f"Erro ao salvar o índice FAISS: {e}")

# Verifique se o conteúdo foi adicionado corretamente ao faiss_context
faiss_context = f"{web_content}"

# Confirme que o índice foi salvo corretamente
print("Treinamento concluído! Os documentos foram processados e indexados para futuras consultas.")

