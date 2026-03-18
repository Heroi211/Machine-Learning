import logging
import mlflow
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from graphs import Graphs as gr
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
from sklearn.pipeline import Pipeline as SkPipeline
load_dotenv()

#Enviroments
ppath_data = os.getenv('PATH_DATA')
ppath_data_preprocessed = os.getenv('PATH_DATA_PREPROCESSED')
ppath_model = os.getenv('PATH_MODEL')
debug = os.getenv('DEBUG')

logging.basicConfig(
    level=logging.INFO(
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
)
logger = logging.getLogger(__name__)

pmsg_raise = "Pipeline interrompido"
agora = datetime.now().strftime('%Y%m%d_%H%M%S')

class Baseline:
    """
    Pipeline para geração do baseline padrão
    """
    def __init__(self,pobjective):
        self.path_data = ppath_data
        self.path_data_preprocessed = ppath_data_preprocessed
        self.path_model = ppath_model
        self.msg_raise = pmsg_raise
        self.objective = pobjective
        self.data = None
        self.target = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.ratio = None
        self.model = None
        self.preprocessor = None
    
    def load_data(self):
        """
        Carregamento dos dados, passo 1
        """
        logger.info("Iniciando Dataset...")
        
        if not self.path_data:
            logger.error("Path não encontrado ou valor vazio")
            raise ValueError(self.msg_raise)
        
        self.data = pd.read_csv(self.path_data)
        self.target = self.data.columns.to_list().pop() 
        
        
        logger.debug(f"Dataset carregado {self.data.shape}")
        logger.debug(f"5 primeiras linhas {self.data.head(5)}")
        
    def summary_overview(self):
        """
        Visão geral do dataset inicial, passo 2
        """
        
        logger.info("Informações gerais do dataset:\n")
        logger.debug(self.data.info())
        logger.info("Colunas presentes no dataset:\n")
        logger.debug(self.data.columns)
        logger.info("Estatísticas inciiais do dataset:\n")
        logger.debug(self.data.describe())
        
    # ------------------------------------------------------
    # EDA - Análise exploratória de dados
    # ------------------------------------------------------
        
    def missing_identifier(self):
        """
        Inicialmente como terceiro passo, identificar os missings values
        """
        
        logger.info("Inicializando a identificação dos missings values")
        
        if self.target not in self.data.columns:
            logger.error("A Coluna target não esta presente na fonte de dados")
            raise ValueError(self.msg_raise)
        
        if self.data[self.target].isnull().any():
            null_count = self.data[self.target].isnull().sum()
            logger.error(f"Valores nulos ou faltantes foram encontrados na coluna target {null_count}")
            raise ValueError(self.msg_raise)
        
        missing = pd.DataFrame({
            "Coluna" : self.data.columns,
            "Missing_count" : self.data.isnull().sum(),
            "Missing_percentage" : ((self.data.isnull().sum() / len(self.data))*100).round(2)
        })
        
        missing = missing[missing["Missing_count"]>0].sort_values(by='Missing_percentage',ascending=False)
        
        if len(missing)==0:
            logger.info("Dataset não apresentou missing values")
        else:
            logger.debug(f"Missing values identificados na fonte de dados: {missing}")
            gr.build_report(
            g_type=1, # BARH
            x_data=missing['Coluna'],
            y_data=missing['Missing_percentage'],
            title="Distribuição de Missing Values por Coluna",
            xlabel="Porcentagem (%)",
            filename=f"missing_values_{agora}.png",
            color="skyblue"
            )
            
            
    def target_analysis(self):
        """
        Neste passo 4, analisamos a fonte de dados identificando a variável target
        """
        
        logger.info("Iniciando a análise da target")

        self.data.rename(columns={self.target:'target'}, inplace=True)
        logger.info("Variável target definida")
        
        logger.info("Convertendo variável target para binário")
        
        le = LabelEncoder()
        self.data['target'] = le.fit_transform(self.data['target'].astype(str))
        logger.info(f"Target codificada: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        if self.data['target'].isnull().any():
            null_count = self.data['target'].isnull().sum()
            logger.error(f"Valores nulos ou faltantes foram encontrados na coluna target {null_count}")
            raise ValueError(self.msg_raise)
        
        logger.info("Iniciando analise de balancemanto da variável target")
        target_counts = self.data['target'].value_counts()
        target_percentages = self.data['target'].value_counts(normalize=True) * 100
        
        logger.info(f"Contagem\n {target_counts}")
        logger.info("\nPercentual:")
        for idx, pct in target_percentages.items():
            label = "Sem churn" if idx==0 else "Churn"
            logger.info(f"{label} ({idx}): {pct:.2f}")
            
        gr.build_report(
            g_type=3, # PIE
            x_data=target_counts.values,
            labels=['Sem Churn (0)', 'Churn (1)'],
            title="Proporção da Variável Target",
            filename=f"target_distribution_pie_{agora}.png",
            color="coral"
        )
        gr.build_report(
            g_type=2, 
            x_data=['Sem Churn', 'Churn'],
            y_data=target_counts.values,    
            title="Distribuição Absoluta da Target",
            ylabel="Quantidade",
            filename=f"target_distribution_bar{agora}.png",
            color="skyblue"
        )
            
        self.ratio = target_counts.min()/target_counts.max()
        logger.info(f"\n Ratio de Balancemanto: {self.ratio:.2f}")
        
        if self.ratio < 0.5:
            logger.warning("Dataset desbalanceado!")
            # Precisa implementar alguma estratégia para tratamento de dataset desbalanceado, mas não sei qual usar.
        else:
            logger.info("✓ Dataset razoavelmente balanceado.")
            
            
    def outlier_analysis(self):
        """
        Passo 5, identificar outliers se houverem.
        """
        logger.info("Iniciando análise de outliers...")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
               
        gr.build_outliers_report(
            data=self.data, 
            numeric_cols=numeric_cols, 
            filename=f"outliers_boxplot_{agora}.png"
        )
            
       
    # ------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------   
       
    def prepare_and_train(self):
        """ Une Clean, Encoding e Train usando Sklearn Pipelines (Fase 2 e 3) """
        logger.info("Iniciando preparação e treino robusto...")

        num_cols = self.x_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.x_train.select_dtypes(exclude=[np.number]).columns.tolist()

        # Criando transformadores (Fase 2: Evita Leakage usando fit apenas no treino)
        num_transformer = SimpleImputer(strategy='median')
        cat_transformer = SkPipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])

        full_pipeline = SkPipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', LogisticRegression(random_state=42, class_weight='balanced'))
        ])

        with mlflow.start_run(run_name=f"baseline_{self.objective}"):

            full_pipeline.fit(self.x_train, self.y_train)
            self.model = full_pipeline

            y_pred = self.model.predict(self.x_test)
            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "f1": f1_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred),
                "recall": recall_score(self.y_test, y_pred)
            }
            
            mlflow.log_params(full_pipeline.named_steps['classifier'].get_params())
            mlflow.log_metrics(metrics)
            
            if os.path.exists("graphs"):
                mlflow.log_artifacts("graphs", artifact_path="plots")
            
            mlflow.sklearn.log_model(self.model, "model")
            logger.info("Pipeline de treino e logs concluído com sucesso.")   
    
    def split_data(self):
        """
        Divisão de treino e teste
        """
        
        logger.info("Iniciando split data...")
        
        x=self.data.drop(columns='target')
        y=self.data['target']
        
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
            
    def save(self):
        """
        Salva o pipeline completo (transformações + modelo) e os dados processados.
        """
        logger.info("Iniciando o salvamento dos artefatos...")

        for path in [self.path_data_preprocessed, self.path_model]:
            if not os.path.exists(path):
                os.makedirs(path)
                logger.info(f"Diretório criado: {path}")

        try:
            feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
            x_test_processed = pd.DataFrame(
                self.model.named_steps['preprocessor'].transform(self.x_test),
                columns=feature_names
            )
            
            csv_path = f"{self.path_data_preprocessed}{self.objective}_sample_{agora}.csv"
            x_test_processed.head(100).to_csv(csv_path, index=False)
            logger.info(f"Amostra processada salva em: {csv_path}")
        except Exception as e:
            logger.warning(f"Não foi possível salvar amostra CSV: {e}")

        model_name = f"baseline_model_{self.objective}_{agora}.joblib"
        joblib_path = os.path.join(self.path_model, model_name)
        
        joblib.dump(self.model, joblib_path)
        logger.info(f"Pipeline (Joblib) salvo em: {joblib_path}")

        try:
            mlflow.log_artifact(joblib_path, artifact_path="model_files")
            logger.info("Modelo registrado como artefato no MLflow.")
        except Exception as e:
            logger.error(f"Erro ao registrar no MLflow: {e}")
        
            
if __name__=="__main__":
            
    start_time = datetime.now()
    logger.info(f"Iniciando o pipeline: {start_time}")
    
    pipeline = Baseline(pobjective="churn")
    pipeline.load_data()
    logger.debug(f"Dados carregados: {start_time - datetime.now() }")
    
    pipeline.summary_overview()
    logger.debug(f"Overview gerado: {start_time - datetime.now() }")
    
    pipeline.missing_identifier()
    logger.debug(f"Identificados Missings: {start_time - datetime.now() }")
    
    pipeline.target_analysis()
    logger.debug(f"Análise da target: {start_time - datetime.now() }")
    
    pipeline.outlier_analysis()
    logger.debug(f"Análise de outliers: {start_time - datetime.now() }")
    
    pipeline.split_data()
    logger.debug(f"Split de dados: {start_time - datetime.now() }")
    
    pipeline.prepare_and_train()
    logger.debug(f"Preparando dados para limpeza e treinamento: {start_time - datetime.now()}")
    
    pipeline.save()
    logger.debug(f"Modelos salvos: {start_time - datetime.now() }")
    
    end_time = start_time - datetime.now()
    
    logger.debug(f"Baseline encerrado em : {end_time}")

        
        