# log_analyzer.py

import pandas as pd
import re
import numpy as np
import os
import pickle
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from scipy import sparse
import logging
from textblob import TextBlob

# --- NOTA DE DEPENDÊNCIAS ---
# Este script requer bibliotecas adicionais. Instale-as com:
# pip install torch scikit-learn scipy textblob
# Para TextBlob, você também pode precisar baixar os corpora:
# python -m textblob.download_corpora

# --- SEED PARA REPRODUZIBILIDADE ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONSTANTES GLOBAIS ---
# Regex para o formato de log de acesso comum (Nginx/Apache)
LOG_PATTERN = re.compile(
    r'(\S+) \S+ \S+ \[([^\]]+)\] "([^"]+)" (\d{3}) \S+ "([^"]*)" "([^"]*)"'
)

# --- FUNÇÕES AUXILIARES ---

def get_sentiment(text):
    """
    Analisa o sentimento de uma string de texto usando TextBlob.
    Retorna 'Positivo', 'Negativo' ou 'Neutro' com base na polaridade.
    """
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return 'Positivo'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negativo'
    else:
        return 'Neutro'

def status_to_level(status):
    """
    Converte um código de status HTTP em um nível de log (ERROR, WARN, INFO).
    """
    if status >= 500: return 'ERROR'
    if status >= 400: return 'WARN'
    return 'INFO'

def clean_message(msg):
    """
    Pré-processa uma mensagem de log (requisição HTTP) para agrupar URLs semelhantes,
    substituindo IDs, UUIDs, dimensões e slugs por placeholders.
    """
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

def parse_logs(log_string):
    """
    Analisa uma string de logs de acesso (formato Nginx/Apache) e a converte em um DataFrame do pandas.
    """
    parsed_data = []
    for line in log_string.strip().split('\n'):
        match = LOG_PATTERN.match(line) # Usa a constante global
        if match:
            parsed_data.append(match.groups())
            
    df = pd.DataFrame(parsed_data, columns=['ip', 'timestamp_str', 'request', 'status', 'referer', 'user_agent'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp_str'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df['status'] = df['status'].astype(int)

    df['level'] = df['status'].apply(status_to_level) # Usa a função auxiliar
    df['message'] = df['request']
    df['clean_message'] = df['message'].apply(clean_message) # Usa a função auxiliar
    return df

# --- CLASSES DO MODELO AUTOENCODER ---

class SparseLogDataset(Dataset):
    """
    Dataset customizado para lidar com matrizes esparsas de forma eficiente no PyTorch.
    Converte apenas uma linha para denso no momento do acesso.
    """
    def __init__(self, sparse_matrix):
        self.sparse_matrix = sparse_matrix

    def __len__(self):
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sparse_matrix[idx].toarray().squeeze(0))

class Autoencoder(nn.Module):
    """
    Define a arquitetura da rede neural Autoencoder.
    """
    def __init__(self, input_dim, encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- FUNÇÕES DE TREINAMENTO E DETECÇÃO ---

def train_autoencoder_model(log_df, epochs=10, batch_size=3072):
    """
    Treina o modelo Autoencoder com os dados de log.
    """
    logger.info("🚀 Vetorizando e preparando os dados para o Autoencoder...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(log_df['clean_message'])
    
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🧠 Treinando no dispositivo: {device}")
    
    try:
        num_workers = os.cpu_count()
    except NotImplementedError:
        num_workers = 2
    logger.info(f"Usando {num_workers} workers para o carregamento de dados.")

    dataset = SparseLogDataset(X_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    input_dim = X.shape[1]
    model = Autoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logger.info(f"🔥 Iniciando o treinamento por {epochs} épocas...")
    for epoch in range(epochs):
        total_loss = 0
        for inputs in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f'Época [{epoch+1}/{epochs}], Perda (Erro de Reconstrução): {total_loss/len(dataloader):.6f}')

    trained_model = {
        'model_state': model.state_dict(),
        'input_dim': input_dim,
        'vectorizer': vectorizer,
        'scaler': scaler
    }
    logger.info("✅ Modelo treinado com sucesso!")
    return trained_model

def detect_anomalies_autoencoder(log_df, trained_model, batch_size=256):
    """
    Calcula o score de anomalia para cada log usando o Autoencoder treinado.
    Processa em lotes para economizar memória.
    """
    logger.info("🔍  Detectando anomalias com base no erro de reconstrução...")
    
    vectorizer = trained_model['vectorizer']
    scaler = trained_model['scaler']
    input_dim = trained_model['input_dim']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder(input_dim).to(device)
    model.load_state_dict(trained_model['model_state'])
    model.eval()

    X = vectorizer.transform(log_df['clean_message'])
    X_scaled = scaler.transform(X)

    all_losses = []
    with torch.no_grad():
        for i in range(0, X_scaled.shape[0], batch_size):
            batch = X_scaled[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch.toarray()).to(device)
            
            reconstructed = model(batch_tensor)
            mse_loss = nn.MSELoss(reduction='none')
            loss = mse_loss(reconstructed, batch_tensor).mean(axis=1)
            all_losses.append(loss.cpu().numpy())
    
    log_df['anomaly_score'] = np.concatenate(all_losses)
    logger.info("✅ Análise de anomalias concluída.")
    return log_df

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analisa um arquivo de log para detectar padrões anômalos usando um Autoencoder.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("log_file", help="O caminho para o arquivo de log a ser analisado.")
    parser.add_argument("--model_file", default=os.path.join("model", "model_autoencoder.pkl"), help="O caminho para salvar ou carregar o modelo treinado.")
    parser.add_argument("--threshold_quantile", type=float, default=0.98, help="Quantil para definir o que é considerado uma anomalia (ex: 0.98 para os 2%% mais anômalos).")
    # output_format agora é apenas informativo, pois 'table' é o padrão e único formato.
    parser.add_argument("--output_format", default='table',
                        help="Formato da saída: 'table' (legível por humanos).")
    args = parser.parse_args()

    try:
        with open(args.log_file, "r", encoding="utf-8") as f:
            log_data = f.read()
    except FileNotFoundError:
        logger.error(f"Erro: O arquivo de log '{args.log_file}' não foi encontrado.")
        exit(1)

    logs_df = parse_logs(log_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"INFO: Executando no dispositivo: {device}")

    model = None
    if os.path.exists(args.model_file):
        logger.info(f"\n🔄 Carregando modelo Autoencoder existente de '{args.model_file}'...")
        try:
            with open(args.model_file, 'rb') as f:
                model = torch.load(f, map_location=device)
            logger.info("✅ Modelo carregado com sucesso!")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar o modelo: {e}. Treinando um novo modelo.")
            model = None

    if model is None:
        logger.info(f"\n🚫 Modelo não encontrado ou falhou ao carregar. Treinando um novo modelo Autoencoder...")
        model = train_autoencoder_model(logs_df)
        os.makedirs(os.path.dirname(args.model_file), exist_ok=True)
        with open(args.model_file, 'wb') as f:
            torch.save(model, f)
        logger.info(f"💾 Modelo salvo com sucesso em '{args.model_file}'!")

    logs_df = detect_anomalies_autoencoder(logs_df, model)
    
    anomaly_threshold = logs_df['anomaly_score'].quantile(args.threshold_quantile)
    anomalous_logs = logs_df[logs_df['anomaly_score'] >= anomaly_threshold]

    logger.info(f"\n--- 📊 Sumário de Padrões Anômalos (Acima do quantil {args.threshold_quantile:.2f}, Score > {anomaly_threshold:.6f}) ---\n")

    if anomalous_logs.empty:
        logger.info("Nenhum padrão anômalo significativo encontrado com o limiar atual.")
    else:
        anomaly_summary = anomalous_logs.groupby('clean_message').agg(
            count=('ip', 'size'),
            max_score=('anomaly_score', 'max'),
            unique_ips=('ip', lambda x: x.nunique()),
            first_seen=('timestamp', 'min'),
            last_seen=('timestamp', 'max'),
            example_log=('message', 'first'),
            sentiment=('clean_message', lambda x: get_sentiment(x.iloc[0]))
        ).sort_values(by='count', ascending=False).reset_index()

        logger.info(f"\n{'Contagem':<10} | {'Score Máx':<12} | {'IPs Únicos':<12} | {'Sentimento':<10} | {'Padrão Anômalo Detectado'}")
        logger.info(f"{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*60}")
        for _, row in anomaly_summary.head(15).iterrows():
            pattern = row['clean_message']
            if len(pattern) > 58:
                pattern = pattern[:55] + "..."
            logger.info(f"{row['count']:<10} | {row['max_score']:.6f}   | {row['unique_ips']:<12} | {row['sentiment']:<10} | {pattern}")
        logger.info("\n* Padrões ordenados por contagem. 'Score Máx' é o maior score de anomalia para um log desse padrão.")
        logger.info("* 'Sentimento' é uma análise básica do padrão de log.")