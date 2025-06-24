import os
import streamlit as st
import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

# --- Configuração Inicial e Conexão com MongoDB ---

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

st.set_page_config(page_title="Model Cache IA", layout="wide")

st.title("💡 Cache Inteligente de Modelos com MongoDB")
st.markdown("""
Esta é uma demonstração interativa de um sistema de cache de modelos preditivos com MongoDB. 
A aplicação permite selecionar um perfil de modelo, inserir dados e receber uma predição em tempo real, 
carregando os parâmetros do modelo diretamente do banco de dados.
""")

# --- Conexão com o Banco de Dados (com cache para performance) ---

@st.cache_resource
def get_mongo_client():
    """Cria e gerencia a conexão com o MongoDB."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        st.error("A variável de ambiente MONGODB_URI não foi encontrada!")
        st.info("Por favor, crie um arquivo `.env` na raiz do projeto e adicione `MONGODB_URI='your_connection_string'`.")
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping') # Verifica a conexão
        return client
    except ConnectionFailure:
        st.error("Falha ao conectar ao MongoDB. Verifique sua URI de conexão.")
        return None

client = get_mongo_client()

# --- Funções do Banco de Dados ---

def init_db(collection):
    """
    Verifica se a coleção está vazia e, se estiver, treina um modelo
    simples e o insere no DB para demonstração.
    """
    try:
        if collection.count_documents({}) == 0:
            with st.spinner("Nenhum modelo encontrado. Treinando e inserindo um modelo de exemplo ('modelo_ml')..."):
                # Treina um modelo de regressão linear simples
                X = np.array([[1, 2, 3], [2, 1, 0], [3, 3, 3]])
                y = np.array([10, 5, 15])
                model = LinearRegression().fit(X, y)

                # Insere os parâmetros do modelo no MongoDB
                collection.insert_one({
                    "_id": "modelo_ml",
                    "description": "Modelo de Regressão Linear treinado com dados de exemplo.",
                    "weights": model.coef_.tolist(),
                    "bias": model.intercept_.item()
                })
                st.success("✅ Modelo de exemplo ('modelo_ml') inserido com sucesso no MongoDB!")
    except Exception as e:
        st.warning(f"Ocorreu um erro ao inicializar o banco de dados: {e}")

@st.cache_data(ttl=60) # Cache para não consultar o DB a cada interação
def get_available_profiles(_collection):
    """Busca todos os perfis (_id) disponíveis na coleção."""
    try:
        profiles = _collection.distinct("_id")
        return profiles
    except OperationFailure as e:
        st.error(f"Não foi possível buscar os perfis do MongoDB: {e}")
        return []

def get_model_from_db(collection, profile_id):
    """Busca os parâmetros de um modelo específico no MongoDB."""
    return collection.find_one({"_id": profile_id})

# --- Interface Principal ---

if not client:
    st.stop()

# Define os nomes do banco e da coleção, usando os valores do .env ou padrões
DB_NAME = os.getenv("DB_NAME", "model_cache_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "models")
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Inicializa o banco de dados se necessário
init_db(collection)

st.divider()

# Colunas para organizar a interface
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Seleção do Modelo")
    
    available_profiles = get_available_profiles(collection)
    if not available_profiles:
        st.warning("Nenhum perfil de modelo encontrado no banco de dados.")
        st.stop()

    # Widget para selecionar o perfil do modelo
    profile = st.selectbox(
        "Selecione o Perfil do Modelo:",
        options=available_profiles,
        help="Cada perfil corresponde a um documento no MongoDB que armazena os pesos de um modelo."
    )

    st.header("2. Input dos Dados")
    st.write("Insira os valores das features para a predição.")
    
    # Inputs numéricos para as features (baseado no modelo de exemplo com 3 features)
    feature1 = st.number_input("Feature 1", value=1.0)
    feature2 = st.number_input("Feature 2", value=2.0)
    feature3 = st.number_input("Feature 3", value=3.0)
    
    features = [feature1, feature2, feature3]

with col2:
    st.header("3. Predição e Resultados")
    
    if st.button("🚀 Realizar Predição", use_container_width=True, type="primary"):
        model_data = get_model_from_db(collection, profile)
        
        if not model_data:
            st.error(f"Perfil '{profile}' não foi encontrado no banco de dados.")
        else:
            st.success(f"Modelo '{profile}' carregado com sucesso do MongoDB!")
            
            # Extrai pesos e bias
            weights = np.array(model_data["weights"])
            bias = model_data["bias"]
            
            # Realiza a predição
            prediction = float(np.dot(features, weights) + bias)
            
            # Exibe o resultado
            st.metric(label=f"Predição do Modelo '{profile}'", value=f"{prediction:.4f}")
            
            # Expander para mostrar os detalhes do modelo carregado
            with st.expander("🔍 Ver detalhes do modelo carregado do MongoDB"):
                st.write(f"**Descrição:** {model_data.get('description', 'N/A')}")
                st.write("**Pesos (Weights):**")
                st.json(model_data["weights"])
                st.write(f"**Viés (Bias):** `{model_data['bias']}`")
                
                st.markdown("---")
                st.write("A predição foi calculada usando a fórmula: `(features • weights) + bias`")