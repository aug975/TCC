# Como rodar o bot
Processo para instalar e rodar o assistente
## Pré-requisitos
As seguintes bibliotecas devem ser instaladas com pip:
- langchain_openai
- langchain
- langchain_community
- unstructured
- unstructured[pdf]
- unstructured[pptx]
- unstructured[docx]
- python-dotenv
- openai
- faiss-cpu
## Diretório
O diretório com os documentos deve estar em algum lugar na máquina local. Simplesmente modifique a variável "directory_path" com o caminho do diretório.
## Chave de API
Substitua o valor placeholder da variável OPENAIKEY com sua chave de API.
## Ajuste do modelo
Modifique os valores na declaração da variável llm para ajustar o modelo e a temperatura.
## Execução
Rode o arquivo processa_data.py ou processa_web (a partir de URLs desejadas que devem ser adicionadas dentro do arquivo) primeiro com Python 3.12 e aguarde finalização do processo. Depois, execute o chatbot e siga as instruções no terminal.
