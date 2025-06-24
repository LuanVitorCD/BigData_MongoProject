import os
import streamlit as st
import numpy as np
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

# --- Configuração Inicial e Título ---
st.set_page_config(page_title="Model Cache IA", layout="wide")
st.title("💡 Cache Inteligente de Modelos com MongoDB")
st.markdown("""
Esta é uma demonstração interativa de um sistema de cache de modelos preditivos com MongoDB. 
A aplicação permite selecionar um perfil de modelo, inserir dados e receber uma predição em tempo real.
""")

# --- Lógica de Conexão e Banco de Dados (Estilo FastAPI) ---

# Usamos o cache do Streamlit para evitar reconexões, é uma boa prática.
@st.cache_resource
def get_db_collection():
    """
    Carrega variáveis de ambiente, conecta ao MongoDB e retorna a coleção.
    Se a conexão falhar, exibe o erro detalhado na tela.
    """
    try:
        # 1. Carrega as variáveis de ambiente (do .env local ou do Render)
        load_dotenv()
        mongodb_uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("DB_NAME", "model_cache_db")
        collection_name = os.getenv("COLLECTION_NAME", "models")

        if not mongodb_uri:
            st.error("Erro Crítico: A variável de ambiente MONGODB_URI não foi encontrada!")
            return None

        # 2. Conecta ao cliente (mesma lógica do seu código original)
        client = MongoClient(mongodb_uri)
        
        # 3. Força a conexão para verificar se é bem-sucedida (passo importante para depuração)
        client.admin.command('ping')
        
        # 4. Seleciona o banco e a coleção
        db = client[db_name]
        collection = db[collection_name]
        return collection

    except PyMongoError as e:
        st.error("Falha na Conexão com o MongoDB!")
        st.error(f"Erro Detalhado: {e}")
        st.info("""
            **Possíveis Soluções:**
            1.  **No MongoDB Atlas:** Verifique se o acesso de IP está liberado para "qualquer lugar" (0.0.0.0/0) em Network Access.
            2.  **No Render:** Verifique se a variável de ambiente MONGODB_URI está copiada corretamente, incluindo usuário e senha.
            3.  **Senha:** Se sua senha contém caracteres especiais (@, :, /), tente trocá-la por uma senha com apenas letras e números.
        """)
        return None

def init_db(collection):
    """
    Popula o banco de dados com um modelo de exemplo se a coleção estiver vazia.
    Lógica idêntica à sua versão FastAPI.
    """
    try:
        if collection.count_documents({}) == 0:
            with st.spinner("Nenhum modelo encontrado. Treinando e inserindo um modelo de exemplo..."):
                X = np.array([[1, 2, 3], [2, 1, 0]])
                y = np.array([1.8, 3.2])
                model = LinearRegression().fit(X, y)

                collection.insert_one({
                    "_id": "modelo_ml_gerado",
                    "description": "Modelo de Regressão Linear gerado na inicialização.",
                    "weights": model.coef_.tolist(),
                    "bias": model.intercept_.item()
                })
                st.success("✅ Modelo de exemplo ('modelo_ml_gerado') inserido no MongoDB!")
                # Força um rerun para o selectbox ser populado com o novo modelo
                st.experimental_rerun()
    except Exception as e:
        st.warning(f"Ocorreu um erro não fatal ao inicializar o banco de dados: {e}")

# --- Interface Principal do Streamlit ---

# Tenta obter a coleção do banco de dados
collection = get_db_collection()

# Se a conexão falhou, 'collection' será None e a aplicação para aqui.
if collection is None:
    st.stop()

# Se a conexão foi bem-sucedida, continua e inicializa o DB se necessário.
init_db(collection)

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Seleção do Modelo")
    
    try:
        available_profiles = collection.distinct("_id")
    except PyMongoError as e:
        st.error(f"Erro ao buscar perfis do DB: {e}")
        available_profiles = []

    if not available_profiles:
        st.warning("Nenhum perfil de modelo encontrado no banco de dados.")
        st.info("O DB pode estar vazio ou a conexão pode ter problemas de permissão de leitura.")
        st.stop()

    profile = st.selectbox(
        "Selecione o Perfil do Modelo:",
        options=available_profiles,
        help="Cada perfil corresponde a um documento no MongoDB."
    )

    st.header("2. Input dos Dados")
    st.write("Insira os valores das features para a predição.")
    
    feature1 = st.number_input("Feature 1", value=1.0)
    feature2 = st.number_input("Feature 2", value=2.0)
    feature3 = st.number_input("Feature 3", value=3.0)
    
    features = [feature1, feature2, feature3]

with col2:
    st.header("3. Predição e Resultados")
    
    if st.button("🚀 Realizar Predição", use_container_width=True, type="primary"):
        model_data = collection.find_one({"_id": profile})
        
        if not model_data:
            st.error(f"Perfil '{profile}' não foi encontrado.")
        else:
            st.success(f"Modelo '{profile}' carregado com sucesso do MongoDB!")
            
            weights = np.array(model_data["weights"])
            bias = model_data["bias"]
            prediction = float(np.dot(features, weights) + bias)
            
            st.metric(label=f"Predição do Modelo '{profile}'", value=f"{prediction:.4f}")
            
            with st.expander("🔍 Ver detalhes do modelo carregado"):
                # Remove o campo _id para não poluir a visualização
                model_data.pop("_id", None)
                st.json(model_data)