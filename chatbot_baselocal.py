import os
import json
import warnings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.schema import SystemMessage, AIMessage, HumanMessage  
from langchain_community.vectorstores import FAISS  
from langchain_community.document_loaders import DirectoryLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.memory import ConversationBufferMemory

# Evita conflitos do OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Desativa avisos de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)

OPENAIKEY = [key]

# Caminho para o índice FAISS salvo durante o treinamento
faiss_path = "faiss_index"

# Verifica se existe um índice FAISS previamente salvo
if os.path.exists(faiss_path):
    vectorstore = FAISS.load_local(faiss_path, OpenAIEmbeddings(openai_api_key=OPENAIKEY),allow_dangerous_deserialization = True)
    print("Índice do treinamento carregado com sucesso!")
else:
    print("Nenhum índice de base de treinamento encontrado. Execute primeiro o treinamento.")
    exit()

# Configuração do modelo
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=OPENAIKEY)

# Criando memória da conversa (APENAS UMA VEZ)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    max_token_limit=1000,
    return_messages=True
)

# Criando o template do prompt
prompt_template = PromptTemplate(
    input_variables=["chat_history", "user_input"],  # Usa o histórico e a entrada do usuário
    template="""
    Você é um assistente útil e responde perguntas de maneira clara e informativa,
    desde que relacionado ao conteúdo das informações específicas fornecidas e relacionadas a instituição de ensino Mackenzie
    e demais informações contidas nas fontes de dados fornecidas.
    Se a pergunta estiver fora do conteúdo fornecido, por favor, diga:
        "Não estou treinado para dar essa resposta."
    O histórico da conversa até agora é:
    
    {chat_history}
    Agora o usuário fez a seguinte pergunta:

    {user_input}

    Responda de forma precisa e útil.
    """ 
)

# Criando a cadeia de conversação
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt_template,
    input_key="user_input"  # Define o nome correto da entrada esperada
)

# Loop de interação com o usuário
print("Chatbot experimetal da Universidade Mackenzie inicializado (por Augusto Rassi Scrideli).")
print("Digite 'sair' para encerrar.")

while True:
    print()
    user_input = input("Você: ")  
    if user_input.lower() == "sair":
        print("Chatbot encerrado.")
        break

    # Buscando informações relevantes do índice vetorial
    relevant_docs = vectorstore.similarity_search(user_input, k=3)  
    context = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""

    # Gera a resposta considerando o histórico da conversa
    response = conversation.predict(user_input=user_input)  # Agora o input tem o nome correto

    print("Chatbot:", response)  # Removido `.content` pois `response` já é uma string

    # Armazenando contexto
    memory.save_context({"user_input": user_input}, {"output": response})  

