import logging
import pandas as pd
import os
from dotenv import load_dotenv
from graphs import Graphs as gr
from datetime import datetime
import numpy as np

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
    def __init__(self, ppath,pmsg_raise):
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
        
        
        self.load_data()
        self.summary_overview()
    
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
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        
        
        
        