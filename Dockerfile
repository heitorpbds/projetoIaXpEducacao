# --- Estágio 1: Build ---
# Este estágio usa uma imagem Python completa para compilar as dependências em "wheels".
# Isso evita a necessidade de incluir compiladores e ferramentas de build na imagem final.
FROM python:3.11 as builder

WORKDIR /app

# Instala a ferramenta 'wheel' para criar os pacotes
RUN pip install --upgrade pip wheel

# Copia o arquivo de dependências. É copiado separadamente para aproveitar o cache do Docker.
COPY requirements.txt .

# Baixa e compila todas as dependências em formato .whl (wheel)
# --wheel-dir especifica onde salvar os pacotes compilados.
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt


# --- Estágio 2: Final ---
# Este estágio cria a imagem final, que é otimizada para ser leve e segura.
FROM python:3.11-slim

# Garante que a saída do Python seja enviada diretamente para o terminal,
# o que é essencial para visualizar logs em tempo real em contêineres.
ENV PYTHONUNBUFFERED=1

# Cria um usuário não-root ('appuser') para executar a aplicação.
# Rodar como não-root é uma prática de segurança fundamental.
RUN useradd --create-home appuser
WORKDIR /home/appuser

# Copia as dependências pré-compiladas (wheels) do 'build stage'
COPY --from=builder /app/wheels /wheels

# Instala as dependências a partir dos wheels locais e, em seguida, remove a pasta de wheels.
# --no-index --find-links força o pip a usar apenas os arquivos locais, tornando o build mais rápido e determinístico.
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copia apenas o código da aplicação. O modelo e os dados serão montados via volumes.
COPY --chown=appuser:appuser log_analyzer_autoencoders.py .

# Muda para o usuário não-root
USER appuser

# Define o comando padrão para executar a aplicação quando o contêiner iniciar.
CMD ["python", "log_analyzer_autoencoders.py"]