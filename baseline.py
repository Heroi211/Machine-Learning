import logging
import pandas as pd
import os
from dotenv import load_dotenv
from graphs import Graphs as gr

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
        
        missing = missing[missing["Missing_count"]>0].sort_values(by='Missing_Percentage',ascending=False)
        
        if len(missing)==0:
            logger.info("Dataset não apresentou missing values")
        else:
            logger.debug(f"Missing values identificados na fonte de dados: {missing}")
            width   = 10
            height  = 6
            x_data  = missing['Coluna']
            y_data  = missing['Missing_Percentage']
            x_label = "Porcentagem de Missing Values (%)"
            title   = "Distribuição de Missing Values por Coluna"
            color   = "blue"
            graph  = 1
            
            gr.building_graphs(pgraph=graph,ptitle=title,pxlabel=x_label,px_data=x_data,py_data=y_data,pwidth=width,pheight=height,pcolor=color)
            gr.show_graph()
            
            
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
        
        
        
        
        
        
        
        
        
            
        
        
        
        