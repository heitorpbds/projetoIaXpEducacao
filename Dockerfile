# =========================================================================
# BUILD STAGE
# Onde instalamos as dependências e preparamos o ambiente.
# =========================================================================
FROM python:3.9-slim as builder

# Define o diretório de trabalho
WORKDIR /app

# Instala as dependências de forma otimizada
# Primeiro, copia apenas o arquivo de dependências
COPY requirements.txt .

# Instala as dependências usando a cache do pip para acelerar futuras builds
# A flag --no-cache-dir aqui se refere ao cache do HTTP, não ao cache de wheels do pip
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# =========================================================================
# FINAL STAGE
# A imagem final, que será muito menor e mais segura.
# =========================================================================
FROM python:3.9-slim

# Cria um usuário não-root para executar a aplicação
# É uma boa prática de segurança para não rodar como root
RUN useradd --create-home appuser
WORKDIR /home/appuser

# Copia as dependências pré-compiladas (wheels) do 'build stage'
# E instala elas sem precisar de ferramentas de compilação na imagem final
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copia o código da aplicação e define as permissões
COPY --chown=appuser:appuser log_analyzer.py .
COPY --chown=appuser:appuser model/ ./model/
COPY --chown=appuser:appuser data/ ./data/

# Define o usuário que irá rodar o comando
USER appuser

# Comando padrão para rodar o analisador de logs
CMD ["python", "log_analyzer.py"]