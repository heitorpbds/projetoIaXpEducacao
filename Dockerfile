# Usa uma imagem base Python 3.10 leve
FROM python:3.10-slim-buster

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Atualiza o pip para a versão mais recente
RUN pip install --no-cache-dir --upgrade pip

# Copia o arquivo de requisitos e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baixa os corpora do TextBlob, essencial para a análise de sentimento
RUN python -m textblob.download_corpora

# Copia o script da aplicação
COPY log_analyzer_autoencoders.py .

# Comando padrão para executar a aplicação (pode ser sobrescrito pelo docker-compose)
CMD ["python", "log_analyzer_autoencoders.py"]