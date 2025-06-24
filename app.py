# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pymongo
import os
import io
import joblib
from dotenv import load_dotenv

# --- Configuração da Página do Streamlit ---
st.set_page_config(
    page_title="MongoDB: Cache Inteligente de Modelos",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Estilo CSS Customizado (Tema Escuro com detalhes em Azul) ---
st.markdown("""
<style>
    /* Cor de fundo principal */
    .stApp {
        background-color: #0f1116;
    }
    /* Cor dos headers e títulos */
    h1, h2, h3 {
        color: #ffffff;
    }
    /* Cor do texto geral */
    .st-emotion-cache-16txtl3 {
        color: #e0e0e0;
    }
    /* Estilo dos botões */
    .stButton>button {
        color: #ffffff;
        background-color: #0068c9;
        border: 2px solid #0068c9;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0088ff;
        border-color: #0088ff;
        color: #ffffff;
    }
    /* Estilo dos containers e expanders */
    .st-emotion-cache-6q9sum {
        border: 1px solid #2c3e50;
        background-color: #1a1c24;
        border-radius: 10px;
    }
    /* Cor do texto dentro dos expanders */
    .st-emotion-cache-6q9sum p {
        color: #e0e0e0;
    }
    /* Estilo da Sidebar */
    .st-emotion-cache-163ttbj {
        background-color: #1a1c24;
    }
</style>
""", unsafe_allow_html=True)


# --- Conexão com o MongoDB Atlas ---
# Usamos o cache do Streamlit para evitar reconectar a cada interação do usuário.
@st.cache_resource
def get_mongo_client():
    """
    Carrega a URI de conexão do arquivo .env e conecta ao MongoDB.
    Retorna o cliente do banco de dados.
    """
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        st.error("A variável de ambiente 'MONGO_URI' não foi encontrada!")
        st.stop()
    try:
        client = pymongo.MongoClient(mongo_uri)
        # O comando ping é usado para verificar se a conexão foi bem-sucedida.
        client.admin.command('ping')
        print("Conexão com o MongoDB bem-sucedida!")
        return client
    except pymongo.errors.ConnectionFailure as e:
        st.error(f"Não foi possível conectar ao MongoDB: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado na conexão com o MongoDB: {e}")
        st.stop()

# --- Funções de Manipulação de Modelos ---

def train_and_save_models(db_client):
    """
    Treina dois modelos diferentes (um simples e um mais robusto) e os salva
    no MongoDB como objetos binários usando joblib.
    """
    with st.spinner("Carregando dados e treinando os modelos..."):
        # Usando o dataset Iris como exemplo
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Dicionário de modelos a serem treinados
        models = {
            "modelo_basico": DecisionTreeClassifier(max_depth=2, random_state=42),
            "modelo_premium": LogisticRegression(max_iter=200, random_state=42)
        }
        
        collection = db_client.big_data_trabalho.model_cache
        
        for name, model in models.items():
            # Treina o modelo
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            
            # Serializa o modelo para formato binário
            model_binary = io.BytesIO()
            joblib.dump(model, model_binary)
            model_binary.seek(0)
            
            # Prepara o documento para o MongoDB
            doc = {
                "model_id": name,
                "model_data": model_binary.read(),
                "accuracy": accuracy,
                "description": f"Modelo treinado para o perfil '{'Básico' if 'basico' in name else 'Premium'}'"
            }
            
            # Insere ou atualiza o modelo no banco de dados (upsert)
            collection.update_one(
                {"model_id": name},
                {"$set": doc},
                upsert=True
            )
    st.success("Modelos treinados e salvos com sucesso no MongoDB!")


# Usamos o cache de dados para não precisar baixar o mesmo modelo do DB repetidamente.
@st.cache_data(ttl=600) # Cache por 10 minutos
def load_model_from_db(_db_client, model_id):
    """
    Carrega um modelo específico do MongoDB a partir do seu ID.
    O _db_client tem um underscore para indicar ao Streamlit para não hashear o objeto do cliente.
    """
    collection = _db_client.big_data_trabalho.model_cache
    
    with st.spinner(f"Carregando '{model_id}' do MongoDB..."):
        model_doc = collection.find_one({"model_id": model_id})
        
        if not model_doc:
            st.error(f"Modelo com ID '{model_id}' não encontrado no banco de dados.")
            st.warning("Por favor, clique em 'Treinar e Salvar Modelos' primeiro.")
            st.stop()
            
        # Desserializa o modelo a partir dos dados binários
        model_binary = io.BytesIO(model_doc["model_data"])
        model = joblib.load(model_binary)
        
    return model, model_doc.get("description", "Sem descrição"), model_doc.get("accuracy", 0)


# --- Interface do Streamlit ---

st.title("🧠 MongoDB como Cache Inteligente para Modelos Preditivos")
st.markdown("Este aplicativo demonstra como o MongoDB pode ser usado para armazenar diferentes versões de modelos de Machine Learning e carregá-los dinamicamente com base em um contexto, como o tipo de usuário.")

# Inicializa a conexão
client = get_mongo_client()

# --- Seção de Administração (Treinamento) ---
with st.expander("🔧 Painel de Administração: Treinamento dos Modelos"):
    st.info("Esta seção simula o processo de engenharia de ML, onde modelos são treinados e versionados. Clique no botão para treinar dois modelos distintos e salvá-los no MongoDB.")
    if st.button("Treinar e Salvar Modelos no MongoDB"):
        train_and_save_models(client)


st.divider()

# --- Seção de Predição (Aplicação) ---
st.header("🔮 Simulação de Predição em Tempo Real")
st.markdown("Selecione um tipo de usuário e insira os dados para ver qual modelo é carregado do MongoDB para fazer a predição.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Selecione o Contexto")
    # O contexto que define qual modelo carregar
    user_type = st.radio(
        "Tipo de Usuário:",
        ("Usuário Básico", "Usuário Premium"),
        captions=["Usa um modelo mais simples e rápido.", "Usa um modelo mais preciso e robusto."]
    )
    
    # Define qual model_id buscar no MongoDB com base no contexto
    model_id_to_load = "modelo_basico" if user_type == "Usuário Básico" else "modelo_premium"
    
    st.subheader("2. Insira os Dados da Flor")
    # Coleta de input do usuário (usando o dataset Iris como exemplo)
    sepal_length = st.slider("Comprimento da Sépala (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.slider("Largura da Sépala (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Comprimento da Pétala (cm)", 1.0, 7.0, 4.3)
    petal_width = st.slider("Largura da Pétala (cm)", 0.1, 2.5, 1.3)

with col2:
    st.subheader("3. Executar Predição")
    
    if st.button("Fazer Predição", type="primary"):
        # Carrega o modelo correto do MongoDB
        model, description, accuracy = load_model_from_db(client, model_id_to_load)
        
        st.info(f"**Modelo Carregado:** `{model_id_to_load}`")
        st.markdown(f"**Descrição:** *{description}*")
        st.markdown(f"**Acurácia do Modelo:** `{accuracy:.2%}`")
        
        # Prepara os dados de entrada para o modelo
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Faz a predição
        prediction_idx = model.predict(input_data)[0]
        prediction_name = load_iris().target_names[prediction_idx]
        
        st.success(f"**Resultado da Predição:** A flor é da espécie **`{prediction_name.capitalize()}`**.")
        
        with st.expander("Ver detalhes técnicos do modelo carregado"):
            st.write("Abaixo estão os hiperparâmetros do modelo que foi carregado do banco de dados e usado para a predição:")
            st.json(model.get_params())

st.divider()

# --- Seção de Explicação ---
st.header("O que está acontecendo aqui?")
st.markdown("""
1.  **Armazenamento Flexível:** Os modelos de Machine Learning (objetos Python) são serializados (convertidos em uma sequência de bytes) e armazenados como `Binary data` em documentos no MongoDB. Cada documento também contém metadados úteis, como um `model_id`, descrição e métricas de performance (acurácia).

2.  **Cache Inteligente:** Em vez de ter os arquivos de modelo (`.pkl` ou `.joblib`) no sistema de arquivos do servidor, nós os centralizamos no MongoDB. Isso permite:
    - **Versionamento:** Manter diferentes versões (`modelo_basico`, `modelo_premium`) no mesmo lugar.
    - **Carregamento Dinâmico:** A aplicação não precisa saber qual modelo carregar de antemão. Ela consulta o banco de dados com um ID (`model_id`) baseado no contexto (neste caso, o tipo de usuário) e carrega o modelo apropriado.
    - **Escalabilidade:** Em um sistema real com múltiplos servidores de aplicação, todos podem acessar o mesmo repositório centralizado de modelos no MongoDB Atlas.

3.  **Performance com Cache do Streamlit:** Para evitar consultas repetidas ao banco de dados para o mesmo modelo, usamos as funções `@st.cache_resource` (para a conexão) e `@st.cache_data` (para o modelo carregado). Isso garante que, se o mesmo tipo de usuário fizer várias predições, o modelo será baixado do MongoDB apenas uma vez.
""")
