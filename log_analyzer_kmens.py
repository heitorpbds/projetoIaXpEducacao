# log_analyzer.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import numpy as np
import os
import pickle

# Define o caminho do arquivo de log e do modelo salvo
LOG_FILE = os.path.join("data", "app_treino.log")
MODEL_FILE = os.path.join("model", "model_kmeans.pkl")

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
                # Se o segmento for numérico, substitui por <id>
                if segment.isdigit():
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

def analyze_sentiment(row):
    """
    Analisa o sentimento de uma linha de log com base no status HTTP.
    """
    status = row['status']

    if status >= 500: return -0.9  # Erro de Servidor (muito negativo)
    if status >= 400: return -0.7  # Erro de Cliente (negativo)
    if status >= 300: return 0.1   # Redirecionamento (neutro)
    if status >= 200: return 0.5   # Sucesso (positivo)
    
    return 0.0 # Padrão

def train_log_cluster_model(log_df, n_clusters=6):
    """
    Treina um modelo de clusterização K-Means com base nos dados de log processados.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    message_vectors = vectorizer.fit_transform(log_df['clean_message'])

    label_encoder = LabelEncoder()
    level_vectors = label_encoder.fit_transform(log_df['level']).reshape(-1, 1)

    combined_features = hstack([message_vectors, level_vectors])

    print(f"🚀 Treinando o modelo K-Means com {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(combined_features)
    
    cluster_labels = kmeans.labels_
    print("✅ Modelo treinado com sucesso!")
    return kmeans, vectorizer, label_encoder, cluster_labels

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
    print("\n🔍  Realizando análise de sentimento...")
    logs_df['sentiment'] = logs_df.apply(analyze_sentiment, axis=1)
    print("✅ Análise de sentimento concluída.")

    if os.path.exists(MODEL_FILE):
        print(f"\n🔄 Carregando modelo existente de '{MODEL_FILE}'...")
        with open(MODEL_FILE, 'rb') as f:
            model_data = pickle.load(f)
        
        model, vec, encoder = model_data['model'], model_data['vectorizer'], model_data['encoder']
        num_clusters = model.n_clusters
        print("✅ Modelo carregado com sucesso!")

        message_vectors = vec.transform(logs_df['clean_message'])
        level_vectors = encoder.transform(logs_df['level']).reshape(-1, 1)
        combined_features = hstack([message_vectors, level_vectors])
        labels = model.predict(combined_features)
    else:
        print(f"\n🚫 Modelo não encontrado. Treinando um novo modelo...")
        num_clusters = 6
        model, vec, encoder, labels = train_log_cluster_model(logs_df, n_clusters=num_clusters)

        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        model_data = {'model': model, 'vectorizer': vec, 'encoder': encoder}
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Modelo salvo com sucesso em '{MODEL_FILE}'!")

    logs_df['cluster'] = labels

    print("\n--- 📊 Análise de Clusters de Log com Sentimento ---\n")
    for i in range(num_clusters):
        cluster_df = logs_df[logs_df['cluster'] == i]
        if cluster_df.empty:
            continue
        
        avg_sentiment = cluster_df['sentiment'].mean()
        
        if avg_sentiment < -0.3: sentiment_label = "🔴 Negativo"
        elif avg_sentiment > 0.3: sentiment_label = "🟢 Positivo"
        else: sentiment_label = "⚪️ Neutro"

        print(f"--- Cluster {i} | Sentimento Médio: {avg_sentiment:.2f} ({sentiment_label}) ---")
        
        # Adiciona a coluna 'status' na amostragem para mais contexto
        cluster_samples = cluster_df.sample(min(10, len(cluster_df)))
        for _, row in cluster_samples.iterrows():
            print(f"  (Sent: {row['sentiment']:.2f}) [{row['status']}] {row['message']}")
        print("-" * (40 + len(str(i)) + len(sentiment_label)) + "\n")