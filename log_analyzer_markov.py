# log_analyzer.py

import pandas as pd
import re
import numpy as np
import os
import pickle
from scipy import sparse # Importa a biblioteca scipy

# Define o caminho do arquivo de log e do modelo salvo
LOG_FILE = os.path.join("data", "app.log")
MODEL_FILE = os.path.join("model", "model_markov.pkl")

def parse_logs(log_string):
    """
    Analisa uma string de logs de acesso (formato Nginx/Apache) e a converte em um DataFrame do pandas.
    """
    # Regex para o formato de log de acesso comum
    log_pattern = re.compile(
        r'(\S+) \S+ \S+ \[([^\]]+)\] "([^"]+)" (\d{3}) \S+ "([^"]*)" "([^"]*)"'
    )
    
    parsed_data = []
    for line in log_string.strip().split('\n'):
        match = log_pattern.match(line)
        if match:
            # ip, timestamp, request, status, referer, user_agent
            parsed_data.append(match.groups())
            
    df = pd.DataFrame(parsed_data, columns=['ip', 'timestamp_str', 'request', 'status', 'referer', 'user_agent'])
    
    # Converte timestamp e status para os tipos corretos
    df['timestamp'] = pd.to_datetime(df['timestamp_str'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True) # Remove linhas com timestamp inválido
    df['status'] = df['status'].astype(int)

    # Cria a coluna 'level' com base no status HTTP
    def status_to_level(status):
        if status >= 500: return 'ERROR'
        if status >= 400: return 'WARN'
        return 'INFO'
    df['level'] = df['status'].apply(status_to_level)

    # Usa a requisição como a 'message' principal
    df['message'] = df['request']

    # Pré-processamento para agrupar URLs semelhantes
    def clean_message(msg):
        parts = msg.split()
        if len(parts) > 1:
            method = parts[0]
            path = parts[1]
            
            # Remove query strings primeiro
            path = re.sub(r'\?.*$', '', path)
            
            # Divide o caminho em segmentos
            path_segments = path.strip('/').split('/')
            
            cleaned_segments = []
            for segment in path_segments:
                # Regex para detectar UUIDs
                if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', segment):
                    cleaned_segments.append('<uuid>')
                # Se o segmento for numérico, substitui por <id>
                elif segment.isdigit():
                    cleaned_segments.append('<id>')
                # Generaliza dimensões de imagem (ex: 200x200)
                elif re.match(r'^\d+x\d+$', segment):
                    cleaned_segments.append('<dimensions>')
                # Se o segmento parece um slug (contém caracteres codificados ou é muito longo)
                elif '%' in segment or len(segment) > 30:
                     cleaned_segments.append('<slug>')
                else:
                    cleaned_segments.append(segment)
            
            cleaned_path = '/' + '/'.join(cleaned_segments)
            return f"{method} {cleaned_path}"
        return msg

    df['clean_message'] = df['message'].apply(clean_message)
    return df

def train_markov_model(log_df):
    """
    Treina um modelo de Cadeia de Markov com base na sequência de logs.
    """
    print("🚀 Treinando o modelo de Cadeia de Markov...")
    
    # Define os estados únicos a partir das mensagens limpas
    states = log_df['clean_message'].unique()
    state_map = {state: i for i, state in enumerate(states)}
    inv_state_map = {i: state for state, i in state_map.items()}
    num_states = len(states)

    # Cria a matriz de transição como uma matriz esparsa (muito mais eficiente em memória)
    transition_matrix = sparse.dok_matrix((num_states, num_states), dtype=np.float64)
    
    # Converte a sequência de logs em uma sequência de índices de estado
    log_sequence = log_df['clean_message'].map(state_map).tolist()

    # Preenche a matriz de transição com as contagens
    for i in range(len(log_sequence) - 1):
        current_state_idx = log_sequence[i]
        next_state_idx = log_sequence[i+1]
        transition_matrix[current_state_idx, next_state_idx] += 1

    # Normaliza as linhas para obter probabilidades
    # Converte para o formato CSR para operações de linha eficientes
    transition_matrix = transition_matrix.tocsr()
    row_sums = transition_matrix.sum(axis=1)
    non_zero_rows = row_sums.nonzero()[0]

    for i in non_zero_rows:
        # Normaliza a linha dividindo pelo seu somatório
        start = transition_matrix.indptr[i]
        end = transition_matrix.indptr[i+1]
        transition_matrix.data[start:end] /= row_sums[i, 0]

    model = {
        'transition_matrix': transition_matrix,
        'state_map': state_map,
        'inv_state_map': inv_state_map
    }
    print("✅ Modelo treinado com sucesso!")
    return model

def detect_anomalies(log_df, model):
    """
    Calcula o score de anomalia para cada transição de log usando o modelo de Markov.
    """
    print("🔍  Detectando anomalias com base no modelo treinado...")
    transition_matrix = model['transition_matrix']
    state_map = model['state_map']

    # Mapeia logs para índices, tratando logs não vistos no treino
    log_sequence = log_df['clean_message'].map(state_map).fillna(-1).astype(int).tolist()
    
    anomaly_scores = [0.0]  # O primeiro log não tem transição anterior

    for i in range(len(log_sequence) - 1):
        current_state_idx = log_sequence[i]
        next_state_idx = log_sequence[i+1]
        
        # Se um dos estados não foi visto no treino, a transição é anômala
        if current_state_idx == -1 or next_state_idx == -1:
            prob = 0
        else:
            prob = transition_matrix[current_state_idx, next_state_idx]
        
        # Calcula o score de anomalia (probabilidades menores geram scores maiores)
        if prob == 0:
            score = np.inf  # Transição impossível (anomalia máxima)
        else:
            score = -np.log(prob)
        
        anomaly_scores.append(score)
    
    log_df['anomaly_score'] = anomaly_scores
    print("✅ Análise de anomalias concluída.")
    return log_df

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            log_data = f.read()
    except FileNotFoundError:
        print(f"Erro: O arquivo de log '{LOG_FILE}' não foi encontrado.")
        print("Por favor, execute o script 'generate_logs.py' primeiro.")
        exit(1)

    logs_df = parse_logs(log_data)

    if os.path.exists(MODEL_FILE):
        print(f"\n🔄 Carregando modelo de Markov existente de '{MODEL_FILE}'...")
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print("✅ Modelo carregado com sucesso!")
    else:
        print(f"\n🚫 Modelo não encontrado. Treinando um novo modelo de Markov...")
        model = train_markov_model(logs_df)

        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 Modelo salvo com sucesso em '{MODEL_FILE}'!")

    # Detecta anomalias no conjunto de logs
    logs_df = detect_anomalies(logs_df, model)

    # Exibe as transições mais anômalas
    print("\n--- 📊 Top 10 Transições de Log Mais Anômalas ---\n")
    
    # Adiciona a mensagem do log anterior para dar contexto à transição
    logs_df['previous_message'] = logs_df['message'].shift(1)
    anomalous_transitions = logs_df.sort_values(by='anomaly_score', ascending=False).head(10)

    if anomalous_transitions.empty:
        print("Nenhuma anomalia encontrada.")
    else:
        for _, row in anomalous_transitions.iterrows():
            score = row['anomaly_score']
            score_str = f"{score:.2f}" if score != np.inf else "INF (Nunca vista)"
            
            print(f"Score de Anomalia: {score_str}")
            print(f"  DE:  {row['previous_message']}")
            print(f"  PARA: {row['message']}\n")