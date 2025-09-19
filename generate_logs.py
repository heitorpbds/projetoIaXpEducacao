import os
import random
from datetime import datetime, timedelta

# Define o caminho do diretório e do arquivo
LOG_DIR = "data"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Garante que o diretório 'data' exista
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Tipos de logs para simulação
LOG_LEVELS = ["INFO", "INFO", "INFO", "DEBUG", "WARN", "ERROR"]

# Mensagens de log para simulação
MESSAGES = [
    "User admin logged in",
    "Failed to connect to database",
    "Query executed in 120ms",
    "User guest accessed dashboard",
    "Connection pool running low",
    "Failed to authenticate user",
    "Data upload successful",
    "API endpoint /status checked",
    "Internal server error"
]

def generate_logs(num_lines=1_000_000):
    """
    Gera um arquivo de log com o número especificado de linhas.
    """
    start_time = datetime.now()
    
    with open(LOG_FILE, "w") as f:
        for i in range(num_lines):
            # Gera um timestamp sequencial
            current_time = start_time + timedelta(seconds=i)
            timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

            # Seleciona um nível de log e uma mensagem aleatórios
            log_level = random.choice(LOG_LEVELS)
            message = random.choice(MESSAGES)

            # Adiciona variação para a mensagem de "query lenta"
            if "Query executed in" in message:
                message = f"Query executed in {random.randint(50, 200)}ms"
            
            # Formata a linha de log
            log_line = f"{timestamp_str} {log_level} {message}\n"
            f.write(log_line)

            # Imprime o progresso a cada 100.000 linhas
            if (i + 1) % 100_000 == 0:
                print(f"Progresso: {(i + 1) / 1_000_000 * 100:.0f}%")
    
    print(f"\nArquivo '{LOG_FILE}' com {num_lines} linhas gerado com sucesso!")

if __name__ == "__main__":
    generate_logs()