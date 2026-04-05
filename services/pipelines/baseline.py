import logging
import mlflow
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from core.graphs import Graphs as gr
from core.configs import settings
from core.custom_logger import setup_log
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import glob 

import shutil


load_dotenv()

os.makedirs(settings.mlflow_artifact_root, exist_ok=True)
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

#Enviroments
ppath_data = settings.path_data
ppath_data_preprocessed = settings.path_data_preprocessed
ppath_model = settings.path_model
ppath_graphs = settings.path_graphs

ptest_size = settings.test_size
prandom_state = settings.random_state

logger = logging.getLogger("ml.pipeline")
pmsg_raise = "Pipeline interrompido"
pagora = datetime.now().strftime('%Y%m%d_%H%M%S')
psnapshot_path = os.path.join(settings.path_data, settings.path_logs, pagora)

class Baseline:
    """
    Pipeline de baseline para problemas de **classificação binária tabulada**.

    Contrato de entrada do CSV
    --------------------------
    - A coluna alvo (target) deve ser sempre a **última coluna** do arquivo.
    - O arquivo **não deve conter coluna de ID** (remover antes do upload).
    - Valores ausentes são imputados automaticamente (mediana para numéricos,
      moda para categóricos).
    - O target é binarizado: qualquer valor > 0 vira 1, 0 permanece 0.

    Escopo e limitações
    --------------------
    - Suporta apenas **classificação binária**. Regressão e multiclasse
      estão fora do escopo deste pipeline.
    - Desbalanceamento de classes é tratado via `class_weight='balanced'`
      na Regressão Logística — estratégia de reamostragem não implementada.
    - Modelo gerado serve como referência mínima; comparar sempre com o
      pipeline de Feature Engineering antes de promover para produção.
    """
    def __init__(self, pobjective, run_timestamp: str | None = None, csv_path: str | None = None, class_labels: tuple[str, str] | None = None):
        """
        Parameters
        ----------
        pobjective : str
            Identificador do domínio (ex.: "heart_disease", "churn").
        run_timestamp : str, optional
            Timestamp do run para naming de artefatos. Gerado automaticamente se omitido.
        csv_path : str, optional
            Caminho explícito do CSV (upload via API). Se omitido, usa o mais recente em PATH_DATA.
        class_labels : tuple[str, str], optional
            Rótulos semânticos das classes (negativa, positiva).
            Padrão: ("Sem <objective>", "<objective>").
            Exemplo para churn: ("Não Churn", "Churn").
        """
        self.path_data = ppath_data
        self.path_data_preprocessed = ppath_data_preprocessed
        self.path_model = ppath_model
        self.path_graphs = ppath_graphs
        self.msg_raise = pmsg_raise
        self.objective = pobjective
        self.test_size = ptest_size
        self.random_state = prandom_state
        self._explicit_csv_path = os.path.abspath(csv_path) if csv_path else None
        
        if class_labels is not None:
            self.label_neg, self.label_pos = class_labels
        else:
            self.label_neg = f"Sem {self.objective}"
            self.label_pos = str(self.objective)
        if run_timestamp is not None:
            self.now = run_timestamp
            self.snapshot_path = os.path.join(settings.path_data, settings.path_logs, run_timestamp)
        else:
            self.now = pagora
            self.snapshot_path = psnapshot_path
        self.current_csv_path = None
        self.data = None
        self.data_encoded = None
        self.target = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.ratio = None
        self.model = None
    
    def load_data(self):
        """
        Carregamento dos dados, passo 1.
        Se `csv_path` foi passado no construtor (ex.: upload da API), usa esse ficheiro;
        caso contrário (CLI), escolhe o CSV mais recente em `path_data` por compatibilidade.
        """
        logger.info("Iniciando Dataset...")

        if self._explicit_csv_path:
            if not os.path.isfile(self._explicit_csv_path):
                logger.error(f"CSV não encontrado: {self._explicit_csv_path}")
                raise ValueError(self.msg_raise)
            self.current_csv_path = self._explicit_csv_path
            logger.info(f"Arquivo CSV (rota explícita): {self.current_csv_path}")
            self.data = pd.read_csv(self.current_csv_path)
        else:
            if not self.path_data:
                logger.error("Path não encontrado ou valor vazio")
                raise ValueError(self.msg_raise)

            path_data = self.path_data
            csv = os.path.join(path_data, "*.csv")
            file = glob.glob(csv)
            if not file:
                logger.error(f"Nenhum arquivo CSV encontrado no caminho: {path_data}")
                raise ValueError(self.msg_raise)

            file_must_modern = max(file, key=os.path.getctime)
            self.current_csv_path = file_must_modern
            logger.info(f"Arquivo CSV encontrado (modo legado — último ctime): {file_must_modern}")
            self.data = pd.read_csv(file_must_modern)
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
            filename=f"missing_values_{self.now}.png",
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
        
        self.data['target'] = self.data['target'].astype(int)
        self.data['target'] = np.where(self.data['target']>0,1,0)
        
        
        target_counts = self.data['target'].value_counts()
        target_percentages = self.data['target'].value_counts(normalize=True) * 100
        
        logger.info(f"Contagem\n {target_counts}")
        logger.info("\nPercentual:")
        for idx, pct in target_percentages.items():
            label = self.label_neg if idx == 0 else self.label_pos
            logger.info(f"{label} ({idx}): {pct:.2f}")

        gr.build_report(
            g_type=3,  # PIE
            x_data=target_counts.values,
            labels=[f"{self.label_neg} (0)", f"{self.label_pos} (1)"],
            title="Proporção da Variável Target",
            filename=f"target_distribution_pie_{self.now}.png",
            color="coral"
        )
        gr.build_report(
            g_type=2,
            x_data=[self.label_neg, self.label_pos],
            y_data=target_counts.values,
            title="Distribuição Absoluta da Target",
            ylabel="Quantidade",
            filename=f"target_distribution_bar_{self.now}.png",
            color="skyblue"
        )
            
        self.ratio = target_counts.min()/target_counts.max()
        logger.info(f"\n Ratio de Balancemanto: {self.ratio:.2f}")
        
        if self.ratio < 0.5:
            logger.warning(
                "Dataset desbalanceado (ratio=%.2f). "
                "Mitigação aplicada: class_weight='balanced' na Regressão Logística. "
                "Estratégias de reamostragem (SMOTE, undersampling) estão fora do escopo deste pipeline.",
                self.ratio,
            )
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
            filename=f"outliers_boxplot_{self.now}.png"
        )
            
       
    # ------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------   

    def clean_and_encode(self):
        """
        Limpeza e encoding diretamente no DataFrame.
        Gera self.data_encoded — dataset completo, limpo, pronto para split e CSV.
        """
        logger.info("Iniciando limpeza e encoding do DataFrame...")

        df_clean = self.data.copy()

        if 'dataset' in df_clean.columns:
            df_clean.drop(columns=['dataset'], inplace=True)
            logger.info("Coluna 'dataset' removida (metadado de origem).")

        non_numeric = df_clean.drop(columns='target').select_dtypes(exclude=[np.number]).columns.tolist()
        num_cols = df_clean.drop(columns='target').select_dtypes(include=[np.number]).columns.tolist()

        for col in non_numeric:
            if df_clean[col].isnull().any():
                mode = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode)
                logger.info(f"{col}: imputado com moda ('{mode}')")

        for col in num_cols:
            if df_clean[col].isnull().any():
                median = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median)
                logger.info(f"{col}: imputado com mediana ({median:.2f})")

        remaining_nulls = df_clean.isnull().sum().sum()
        if remaining_nulls > 0:
            logger.error(f"Ainda existem {remaining_nulls} valores nulos após imputação")
            raise ValueError(self.msg_raise)

        logger.info("Imputação concluída — zero valores nulos")

        cat_cols = []
        for col in non_numeric:
            unique_vals = set(df_clean[col].unique())
            if unique_vals <= {True, False}:
                df_clean[col] = df_clean[col].astype(bool)
                logger.info(f"{col}: binária — convertida para bool, sem encoding.")
            else:
                cat_cols.append(col)

        if cat_cols:
            df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
            logger.info(f"One-hot encoding aplicado em: {cat_cols}")

        self.data_encoded = df_clean
        logger.info(f"DataFrame limpo e codificado: {self.data_encoded.shape}")
        logger.debug(f"Colunas finais: {self.data_encoded.columns.tolist()}")

    def prepare_and_train(self):
        """Treino do modelo baseline sobre dados já limpos e codificados."""
        logger.info("Iniciando treino do baseline...")

        model = LogisticRegression(random_state=self.random_state, class_weight='balanced')

        experiment_name = f"{self.objective}_baseline"
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name, artifact_location=settings.mlflow_artifact_root)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"baseline_{self.objective}"):

            model.fit(self.x_train, self.y_train)
            self.model = model

            y_pred_train = self.model.predict(self.x_train)
            y_pred_test = self.model.predict(self.x_test)

            metrics = {
                "train_accuracy": accuracy_score(self.y_train, y_pred_train),
                "test_accuracy": accuracy_score(self.y_test, y_pred_test),
                "test_f1": f1_score(self.y_test, y_pred_test),
                "test_precision": precision_score(self.y_test, y_pred_test),
                "test_recall": recall_score(self.y_test, y_pred_test),
            }

            overfitting = metrics["train_accuracy"] - metrics["test_accuracy"]

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_metric("overfitting_gap", overfitting)

            if os.path.exists(self.path_graphs):
                mlflow.log_artifacts("graphs", artifact_path=f"{self.path_graphs}plots")

            mlflow.sklearn.log_model(self.model, "model")

            logger.info(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
            logger.info(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
            logger.info(f"Test F1:        {metrics['test_f1']:.4f}")
            logger.info(f"Test Precision: {metrics['test_precision']:.4f}")
            logger.info(f"Test Recall:    {metrics['test_recall']:.4f}")
            logger.info(f"Overfitting:    {overfitting:.4f}")
            logger.info("Treino e logs concluídos.")   
    
    def split_data(self):
        """
        Divisão de treino e teste a partir do DataFrame limpo e codificado.
        """
        
        logger.info("Iniciando split data...")
        
        x = self.data_encoded.drop(columns='target')
        y = self.data_encoded['target']
        
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=self.test_size,random_state=self.random_state,stratify=y)
            
    def save(self):
        """
        Salva o DataFrame pré-processado completo e o modelo treinado.
        """
        logger.info("Iniciando o salvamento dos artefatos...")

        for path in [self.path_data_preprocessed, self.path_model]:
            os.makedirs(path, exist_ok=True)

        csv_path = os.path.join(self.path_data_preprocessed, f"{self.objective}_sample_{self.now}.csv")
        self.data_encoded.to_csv(csv_path, index=False)
        logger.info(f"Dataset pré-processado salvo em: {csv_path} ({self.data_encoded.shape})")

        model_name = f"baseline_model_{self.objective}_{self.now}.joblib"
        joblib_path = os.path.join(self.path_model, model_name)
        joblib.dump(self.model, joblib_path)
        logger.info(f"Modelo salvo em: {joblib_path}")

        try:
            mlflow.log_artifact(joblib_path, artifact_path="model_files")
            logger.info("Modelo registrado como artefato no MLflow.")
        except Exception as e:
            logger.error(f"Erro ao registrar no MLflow: {e}")
            
    
    
    def run(self, start_time):
        self.load_data()
        logger.debug(f"Dados carregados: {datetime.now() - start_time}")
        self.summary_overview()
        logger.debug(f"Overview gerado: {datetime.now() - start_time}")
        self.missing_identifier()
        logger.debug(f"Identificados Missings: {datetime.now() - start_time}")
        self.target_analysis()
        logger.debug(f"Análise da target: {datetime.now() - start_time}")
        self.outlier_analysis()
        logger.debug(f"Análise de outliers: {datetime.now() - start_time}")
        self.clean_and_encode()
        logger.debug(f"Limpeza e encoding: {datetime.now() - start_time}")
        self.split_data()
        logger.debug(f"Split de dados: {datetime.now() - start_time}")
        self.prepare_and_train()
        logger.debug(f"Treino do baseline: {datetime.now() - start_time}")
        self.save()
        logger.debug(f"Artefatos salvos: {datetime.now() - start_time}")
    
    def save_artifacts(self):
        """
        Arquiva o dataset original e os gráficos gerados no diretório de snapshot.
        Limpa a pasta de entrada após o arquivamento.
        """
        logger.info(f"Arquivando artefatos em: {self.snapshot_path}")

        if self.current_csv_path and os.path.exists(self.current_csv_path):
            try:
                shutil.copy(self.current_csv_path, self.snapshot_path)
                logger.info(f"CSV original copiado para snapshot.")
                
                os.remove(self.current_csv_path)
                logger.info(f"Arquivo original removido da pasta de entrada: {self.current_csv_path}")
            except Exception as e:
                logger.error(f"Erro ao mover/deletar o CSV original: {e}")

        target_graphs_path = os.path.join(self.snapshot_path, "graphs")
        if os.path.exists(self.path_graphs):
            if not os.path.exists(target_graphs_path):
                os.makedirs(target_graphs_path)
            
            for file_name in os.listdir(self.path_graphs):
                full_file_name = os.path.join(self.path_graphs, file_name)
                if os.path.isfile(full_file_name):
                    shutil.move(full_file_name, target_graphs_path)
            
            logger.info(f"Gráficos movidos para: {target_graphs_path}")
                
        
        
            
if __name__=="__main__":
            
    logger = setup_log(psnapshot_path, pagora)
    start_time = datetime.now()
    logger.info(f"Iniciando o pipeline: {start_time}")
    pipeline = Baseline(pobjective="Heart_Disease")
    try:
        pipeline.run(start_time)
        pipeline.save_artifacts()
        logger.debug(f"Artefatos salvos: {datetime.now() - start_time }")
        end_time = datetime.now() - start_time
        logger.debug(f"Baseline encerrado em : {end_time}")
    except Exception as e:
        logger.error(f"Erro durante a execução do pipeline: {e}")
        raise ValueError("Pipeline interrompido devido a um erro.")
    
    

        
        