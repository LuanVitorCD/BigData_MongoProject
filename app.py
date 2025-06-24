import os
import streamlit as st
import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

# --- Configuração Inicial e Conexão com MongoDB ---

load_dotenv()
st.set_page_config(page_title="Model Cache IA", layout="wide")

st.title("💡 Cache Inteligente de Modelos com MongoDB")
st.markdown("""
Esta é uma demonstração interativa de um sistema de cache de modelos preditivos com MongoDB. 
A aplicação permite selecionar um perfil de modelo, inserir dados e receber uma predição em tempo real, 
carregando os parâmetros do modelo diretamente do banco de dados.
""")

@st.cache_resource
def get_mongo_client():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        st.error("A variável de ambiente MONGODB_URI não foi encontrada!")
        st.info("Por favor, crie um arquivo `.env` e adicione `MONGODB_URI='your_connection_string'`.")
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"Erro ao conectar ao MongoDB: {e}")
        return None

client = get_mongo_client()

# --- Funções do Banco de Dados ---

def init_db(collection):
    try:
        if collection.count_documents({}) == 0:
            with st.spinner("Nenhum modelo encontrado. Inserindo modelo de exemplo..."):
                X = np.array([[1, 2, 3], [2, 1, 0], [3, 3, 3]])
                y = np.array([10, 5, 15])
                model = LinearRegression().fit(X, y)
                collection.insert_one({
                    "_id": "modelo_ml",
                    "description": "Modelo de Regressão Linear com dados de exemplo.",
                    "weights": model.coef_.tolist(),
                    "bias": model.intercept_.item()
                })
                st.success("✅ Modelo 'modelo_ml' inserido com sucesso!")
    except Exception as e:
        st.warning(f"Erro ao inicializar o banco de dados: {e}")

@st.cache_data(ttl=60)
def get_available_profiles(col):
    try:
        return col.distinct("_id")
    except Exception as e:
        st.error(f"Erro ao buscar perfis: {e}")
        return []

def get_model_from_db(col, profile_id):
    return col.find_one({"_id": profile_id})

if not client:
    st.stop()

DB_NAME = os.getenv("DB_NAME", "model_cache_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "models")
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

init_db(collection)
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Seleção do Modelo")
    available_profiles = get_available_profiles(collection)
    if not available_profiles:
        st.warning("Nenhum perfil de modelo encontrado.")
        st.stop()

    profile = st.selectbox("Perfil do Modelo:", options=available_profiles)

    st.header("2. Input dos Dados")
    feature1 = st.number_input("Feature 1", value=1.0)
    feature2 = st.number_input("Feature 2", value=2.0)
    feature3 = st.number_input("Feature 3", value=3.0)
    features = [feature1, feature2, feature3]

with col2:
    st.header("3. Predição e Resultados")
    if st.button("🚀 Realizar Predição", use_container_width=True, type="primary"):
        model_data = get_model_from_db(collection, profile)
        if not model_data:
            st.error(f"Perfil '{profile}' não encontrado no banco.")
        else:
            st.success(f"Modelo '{profile}' carregado com sucesso!")
            weights = np.array(model_data["weights"])
            bias = model_data["bias"]
            prediction = float(np.dot(features, weights) + bias)
            st.metric(label=f"Predição do Modelo '{profile}'", value=f"{prediction:.4f}")
            with st.expander("🔍 Ver detalhes do modelo"):
                st.write(f"**Descrição:** {model_data.get('description', 'N/A')}")
                st.write("**Pesos:**")
                st.json(model_data["weights"])
                st.write(f"**Bias:** `{bias}`")
                st.markdown("---")
                st.write("Predição calculada como: `(features • weights) + bias`")