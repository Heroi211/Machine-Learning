import logging
import mlflow
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from graphs import Graphs as gr
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

load_dotenv()

ppath = os.getenv('PATH')
debug = os.getenv('DEBUG')

logging.basicConfig(
    level=logging.INFO(
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
)
logger = logging.getLogger(__name__)

pmsg_raise = "Pipeline interrompido"
agora = datetime.now().strftime('%d/%m/%Y %H:%M:%S')

class Baseline:
    """
    Pipeline para geração do baseline padrão
    """
    def __init__(self,pobjective):
        self.path = ppath
        self.msg_raise = pmsg_raise
        self.data = None
        self.data_clean = None
        self.data_encoded = None
        self.target = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.ratio = None
        self.objective = None
        self.model = None
    
    def load_data(self):
        """
        Carregamento dos dados, passo 1
        """
        logger.info("Iniciando Dataset...")
        
        if not self.path:
            logger.error("Path não encontrado ou valor vazio")
            raise ValueError(self.msg_raise)
        
        self.data = pd.read_csv(self.path)
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
        #encontrar uma maneira de tornar isso padrão para todos os datasets
        mapping = {'yes':1,'no':0}
        self.data['target'] = (self.data['target']
                               .astype(str)
                               .str.lower()
                               .str.strip()
                               .map(mapping))
        
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
            
        self.ratio = target_counts.min/target_counts.max()
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
            
    def clean_data(self):
        """
            Tratamento de dados
        """
        logger.info("Iniciando limpeza de dados...")
        
        self.data_clean = pd.DataFrame()
        self.data_clean = self.data
        
        logger.info("Imputando moda em colunas missing não numericas")
        non_numeric_cols = self.data_clean.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            mode = self.data_clean[col].mode()[0]
            self.data_clean[col].fillna(mode,inplace=True)
            logger.debug(f"{col} : imputada com a moda: {mode}")
            
        numeric_cols = self.data_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median = self.data_clean[col].median()
            self.data_clean[col].fillna(median,inplace=True)
            logger.debug(f"{col} : imputada com a mediana : {median}")
            
        logger.info("\n✓ Tratamento de missing values concluído!")
        logger.debug(f"Shape após imputação: {self.data_clean.shape}")
        logger.debug(f"Valores ausentes restantes por coluna: {self.data_clean.isnull().sum()}")
        
    def encoding(self):
        """
        One hot encoding
        """
        
        logger.info("Iniciando encoding de variáveis categorias...")
        non_categorical_col = self.data_clean.select_dtypes(exclude=np.number).columns
        self.data_encoded = pd.get_dummies(self.data_clean,columns=non_categorical_col, drop_first=True)
        
        logger.info(f"Shape após one hot encoding: {self.data_encoded.shape}")
       
    def split_data(self):
        """
        Divisão de treino e teste
        """
        
        logger.info("Iniciando split data...")
        
        x=self.data_encoded.drop(columns='target')
        y=self.data_encoded['target']
        
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=0.2,random_state=42)
       
    def train(self):
        """
        Teste com MLflow
        """
        
        logger.info("Iniciando treino do modelo com mlflow...")
        with mlflow.start_run(run_name="logistic_regression_baseline_pipeline"):
            model=LogisticRegression(random_state=42)
            model.fit(self.x_train,self.y_train)
            
            y_pred_train = model.predict(self.x_train)
            y_pred_test  = model.predict(self.x_test)
            
            train_accuracy = accuracy_score(self.y_train,y_pred_train)
            
            test_accuracy  = accuracy_score(self.y_test,y_pred_test)
            test_f1        = f1_score(self.y_test,y_pred_test)
            test_precision = precision_score(self.y_test,y_pred_test)
            test_recall    = recall_score(self.y_test,y_pred_test)
            
            mlflow.log_metric("train_accuracy",train_accuracy)
            mlflow.log_metric("test_accuracy",test_accuracy)
            mlflow.log_metric("test_f1",test_f1)
            mlflow.log_metric("test_precision",test_precision)
            mlflow.log_metric("test_recall",test_recall)
                    
            overfitting = train_accuracy - test_accuracy
            
            logger.info(f"Logistic Regression \n")
            logger.debug(f"Train Accuracy: {train_accuracy:.4f}\n")
            logger.debug(f"Test Accuracy: {test_accuracy:.4f}\n")
            logger.debug(f"Test F1 Score: {test_f1:.4f}\n")
            logger.debug(f"Test Precision: {test_precision:.4f}\n")
            logger.debug(f"Test Recall: {test_recall:.4f}\n")
            logger.debug(f"Overfitting: {overfitting:.4f}\n")
            
    def save(self):
        """
        Salvando modelo para próxima step
        """
        
        logger.info("Iniciando save do modelo preprocessado")
        self.data_encoded.to_csv(f"data/processed/{self.objective}_preprocessed_{agora}.csv")
        
        logger.debug(f"Pre Processamento: data/processed/{self.objective}_preprocessed_{agora}.csv, salvo")
        logger.debug(f"salvando modelo treinado...")
        mlflow.sklearn.log_model(self.model, f"Logistic_regression_model_{self.objective}_{agora}")
        logger.info(f"Modelo salvo!")
        logger.info("Salvando modelo! com joblib")
        joblib.dump(self.model,f"models/baseline_model_{self.objective}_{agora}.joblib")
        
        
            
if __name__=="__main":
            
    start_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    logger.info(f"Iniciando o pipeline: {start_time}")
    
    pipeline = Baseline(pobjective="churn")
    pipeline.load_data()
    logger.debug(f"Dados carregados: {start_time - datetime.now().strftime('%d/%m/%Y %H:%M:%S') }")
    pipeline.summary_overview()
    logger.debug(f"Overview gerado: {start_time - datetime.now().strftime('%d/%m/%Y %H:%M:%S') }")
    pipeline.missing_identifier()
    logger.debug(f"Identificados Missings: {start_time - datetime.now().strftime('%d/%m/%Y %H:%M:%S') }")
    pipeline.target_analysis()
    logger.debug(f"Análise da target: {start_time - datetime.now().strftime('%d/%m/%Y %H:%M:%S') }")
    pipeline.outlier_analysis()
    logger.debug(f"Análise de outliers: {start_time - datetime.now().strftime('%d/%m/%Y %H:%M:%S') }")
    pipeline.clean_data()
    logger.debug(f"Dados limpos: {start_time - datetime.now().strftime('%d/%m/%Y %H:%M:%S') }")
    pipeline.encoding()
    logger.debug(f"Encoding : {start_time - datetime.now().strftime('%d/%m/%Y %H:%M:%S') }")
    pipeline.split_data()
    logger.debug(f"Split de dados: {start_time - datetime.now().strftime('%d/%m/%Y %H:%M:%S') }")
    pipeline.train()
    logger.debug(f"Treino feito: {start_time - datetime.now().strftime('%d/%m/%Y %H:%M:%S') }")
    pipeline.save()
    logger.debug(f"Modelos salvos: {start_time - datetime.now().strftime('%d/%m/%Y %H:%M:%S') }")
    
    end_time = start_time - datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    
    logger.debug(f"Baseline encerrado em : {end_time}")
        
    
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        
        
        
        