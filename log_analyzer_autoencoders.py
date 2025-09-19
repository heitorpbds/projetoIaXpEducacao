# log_analyzer.py

import pandas as pd
import re
import numpy as np
import os
import pickle
# NOVAS IMPORTAÇÕES PARA O AUTOENCODER
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset # Modificado de TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse # Importa a biblioteca de matrizes esparsas

# --- NOTA DE DEPENDÊNCIAS ---
# Este script requer bibliotecas adicionais. Instale-as com:
# pip install torch scikit-learn scipy

# Define o caminho do arquivo de log e do modelo salvo
LOG_FILE = os.path.join("data", "app_treino.log")
MODEL_FILE = os.path.join("model", "model_autoencoder.pkl") # Nome do modelo alterado

def parse_logs(log_string):
    """
    Analisa uma string de logs de acesso (formato Nginx/Apache) e a converte em um DataFrame do pandas.
    (Esta função permanece inalterada)
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

# --- NOVAS FUNÇÕES E CLASSE PARA O MODELO AUTOENCODER ---

# Dataset customizado para lidar com matrizes esparsas de forma eficiente
class SparseLogDataset(Dataset):
    def __init__(self, sparse_matrix):
        self.sparse_matrix = sparse_matrix

    def __len__(self):
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx):
        # Converte apenas uma linha para denso no momento do acesso
        return torch.FloatTensor(self.sparse_matrix[idx].toarray().squeeze(0))

class Autoencoder(nn.Module):
    """
    Define a arquitetura da rede neural Autoencoder.
    """
    def __init__(self, input_dim, encoding_dim=32):
        super(Autoencoder, self).__init__()
        # Codificador: comprime a entrada para uma representação menor
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, encoding_dim), nn.ReLU()
        )
        # Decodificador: tenta reconstruir a entrada original a partir da representação comprimida
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim), nn.Sigmoid() # Sigmoid para saídas entre 0 e 1
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder_model(log_df, epochs=10, batch_size=64):
    """
    Treina o modelo Autoencoder com os dados de log.
    """
    print("🚀 Vetorizando e preparando os dados para o Autoencoder...")
    # 1. Vetorizar: Manter como matriz esparsa, sem chamar .toarray()
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(log_df['clean_message']) # Retorna matriz esparsa
    
    # 2. Escalar: MinMaxScaler suporta matrizes esparsas
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X) # Retorna matriz esparsa
    
    # 3. Preparar para o PyTorch usando o Dataset customizado
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Treinando no dispositivo: {device}")
    
    dataset = SparseLogDataset(X_scaled) # Usa o novo Dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Inicializar e treinar o modelo
    input_dim = X.shape[1]
    model = Autoencoder(input_dim).to(device)
    criterion = nn.MSELoss() # A perda é o erro de reconstrução
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"🔥 Iniciando o treinamento por {epochs} épocas...")
    for epoch in range(epochs):
        total_loss = 0
        for inputs in dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Época [{epoch+1}/{epochs}], Perda (Erro de Reconstrução): {total_loss/len(dataloader):.6f}')

    # Empacota tudo o que é necessário para a detecção
    trained_model = {
        'model_state': model.state_dict(),
        'input_dim': input_dim,
        'vectorizer': vectorizer,
        'scaler': scaler
    }
    print("✅ Modelo treinado com sucesso!")
    return trained_model

def detect_anomalies_autoencoder(log_df, trained_model, batch_size=256):
    """
    Calcula o score de anomalia para cada log usando o Autoencoder treinado.
    MODIFICADO PARA PROCESSAR EM LOTES E ECONOMIZAR MEMÓRIA.
    """
    print("🔍  Detectando anomalias com base no erro de reconstrução...")
    
    # 1. Carregar componentes do modelo
    vectorizer = trained_model['vectorizer']
    scaler = trained_model['scaler']
    input_dim = trained_model['input_dim']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder(input_dim).to(device)
    model.load_state_dict(trained_model['model_state'])
    model.eval()

    # 2. Transformar os dados, mantendo-os esparsos
    X = vectorizer.transform(log_df['clean_message'])
    X_scaled = scaler.transform(X)

    # 3. Calcular o erro de reconstrução em lotes (batches)
    all_losses = []
    with torch.no_grad():
        for i in range(0, X_scaled.shape[0], batch_size):
            # Pega um lote e converte apenas ele para denso
            batch = X_scaled[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch.toarray()).to(device)
            
            reconstructed = model(batch_tensor)
            mse_loss = nn.MSELoss(reduction='none')
            loss = mse_loss(reconstructed, batch_tensor).mean(axis=1)
            all_losses.append(loss.cpu().numpy())
    
    log_df['anomaly_score'] = np.concatenate(all_losses)
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

    if os.path.exists(MODEL_FILE):
        print(f"\n🔄 Carregando modelo Autoencoder existente de '{MODEL_FILE}'...")
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print("✅ Modelo carregado com sucesso!")
    else:
        print(f"\n🚫 Modelo não encontrado. Treinando um novo modelo Autoencoder...")
        model = train_autoencoder_model(logs_df)

        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 Modelo salvo com sucesso em '{MODEL_FILE}'!")

    # Detecta anomalias no conjunto de logs
    logs_df = detect_anomalies_autoencoder(logs_df, model)

    # Exibe os logs mais anômalos (maior erro de reconstrução)
    print("\n--- 📊 Top 10 Logs Mais Anômalos ---\n")
    
    anomalous_logs = logs_df.sort_values(by='anomaly_score', ascending=False).head(10)

    if anomalous_logs.empty:
        print("Nenhuma anomalia encontrada.")
    else:
        for _, row in anomalous_logs.iterrows():
            print(f"Score (Erro): {row['anomaly_score']:.6f}")
            print(f"  Log Original: {row['message']}\n")