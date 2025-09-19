# log_analyzer.py

import pandas as pd
import re
import numpy as np
import os
import pickle
# NOVAS IMPORTAÇÕES
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from tqdm import tqdm

# --- NOTA DE DEPENDÊNCIAS ---
# Este script requer bibliotecas adicionais. Instale-as com:
# pip install torch transformers scikit-learn tqdm

# Define o caminho do arquivo de log e do modelo salvo
LOG_FILE = os.path.join("data", "app_treino.log")
# Altera o nome do arquivo do modelo para refletir a nova abordagem
MODEL_FILE = os.path.join("model", "model_bert_kmeans.pkl")
EMBEDDINGS_FILE = os.path.join("model", "log_embeddings.pkl") # Para cachear os embeddings

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

# --- NOVAS FUNÇÕES PARA BERT E K-MEANS ---

def get_bert_embeddings(messages, model_name='sentence-transformers/all-MiniLM-L6-v2', batch_size=32):
    """
    Gera embeddings para uma lista de mensagens de log usando um modelo BERT.
    """
    print(f"🤖 Carregando modelo BERT '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"🚀 Gerando embeddings em lotes de {batch_size} no dispositivo: {device}")

    all_embeddings = []
    # Usando tqdm para uma barra de progresso
    for i in tqdm(range(0, len(messages), batch_size), desc="Processando Lotes"):
        batch = messages[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean Pooling para obter um embedding de sentença fixo
        attention_mask = inputs['attention_mask']
        embedding = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(embedding.size()).float()
        sum_embeddings = torch.sum(embedding * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        all_embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(all_embeddings)

def train_clustering_model(embeddings, n_clusters=15):
    """
    Treina um modelo K-Means com os embeddings de log.
    """
    print(f"\n🔄 Treinando modelo K-Means com {n_clusters} clusters...")
    # O parâmetro n_init suprime um FutureWarning
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    print("✅ Modelo K-Means treinado com sucesso!")
    return kmeans

def detect_anomalies_kmeans(log_df, embeddings, kmeans_model):
    """
    Atribui logs a clusters e calcula um score de anomalia.
    """
    print("🔍  Calculando scores de anomalia...")
    # Atribui cada log a um cluster
    clusters = kmeans_model.predict(embeddings)
    log_df['cluster'] = clusters

    # Calcula a distância de cada ponto ao centro do seu cluster (score de anomalia)
    distances = kmeans_model.transform(embeddings)
    min_distances = np.min(distances, axis=1)
    log_df['anomaly_score'] = min_distances
    
    print("✅ Análise de anomalias concluída.")
    return log_df

# --- Bloco de Execução Principal (MODIFICADO) ---
if __name__ == "__main__":
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            log_data = f.read()
    except FileNotFoundError:
        print(f"Erro: O arquivo de log '{LOG_FILE}' não foi encontrado.")
        print("Por favor, execute o script 'generate_logs.py' primeiro.")
        exit(1)

    logs_df = parse_logs(log_data)
    
    # --- Lógica de Embeddings e Clusterização ---
    
    # Gera ou carrega embeddings para evitar reprocessamento
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"\n🔄 Carregando embeddings cacheados de '{EMBEDDINGS_FILE}'...")
        with open(EMBEDDINGS_FILE, 'rb') as f:
            log_embeddings = pickle.load(f)
        print("✅ Embeddings carregados.")
    else:
        print(f"\n🧠 Gerando novos embeddings BERT para {len(logs_df)} logs...")
        log_embeddings = get_bert_embeddings(logs_df['clean_message'].tolist())
        os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(log_embeddings, f)
        print(f"💾 Embeddings salvos em cache em '{EMBEDDINGS_FILE}'!")

    # Treina ou carrega o modelo K-Means
    if os.path.exists(MODEL_FILE):
        print(f"\n🔄 Carregando modelo K-Means existente de '{MODEL_FILE}'...")
        with open(MODEL_FILE, 'rb') as f:
            kmeans_model = pickle.load(f)
        print("✅ Modelo carregado com sucesso!")
    else:
        # O número de clusters é um hiperparâmetro importante. 15 é um ponto de partida.
        NUM_CLUSTERS = 15 
        kmeans_model = train_clustering_model(log_embeddings, n_clusters=NUM_CLUSTERS)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(kmeans_model, f)
        print(f"💾 Modelo K-Means salvo com sucesso em '{MODEL_FILE}'!")

    # Detecta anomalias usando o modelo
    logs_df = detect_anomalies_kmeans(logs_df, log_embeddings, kmeans_model)

    # Exibe os logs mais anômalos
    print("\n--- 📊 Top 10 Logs Mais Anômalos (maior distância do centro do cluster) ---\n")
    
    anomalous_logs = logs_df.sort_values(by='anomaly_score', ascending=False).head(10)

    if anomalous_logs.empty:
        print("Nenhuma anomalia encontrada.")
    else:
        for _, row in anomalous_logs.iterrows():
            print(f"Score de Anomalia: {row['anomaly_score']:.4f} | Cluster: {row['cluster']}")
            print(f"  Log: {row['message']}\n")

    # Exibe um resumo dos clusters
    print("\n--- 📦 Resumo dos Clusters ---\n")
    cluster_summary = logs_df['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_summary.items():
        print(f"Cluster {cluster_id}: {count} logs")
        # Mostra um exemplo de log do cluster
        example_log = logs_df[logs_df['cluster'] == cluster_id].iloc[0]['clean_message']
        print(f"  Exemplo: {example_log}\n")