import matplotlib.pyplot as plt
import os
import logging
import seaborn as sns
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)
path_graphs = os.getenv('PATH_GRAPHS')

class Graphs:
    @staticmethod
    def build_report(g_type, x_data, y_data=None, title="", xlabel="", ylabel="", 
                     filename="graph.png", color="coral", labels=None):
        """
        Gera e salva gráficos (BAR, BARH, PIE).
        g_type: 1 (BARH), 2 (BAR), 3 (PIE)
        """
        # Garantir diretório na raiz
        if not os.path.exists(f"{path_graphs}"):
            os.makedirs(f"{path_graphs}")
            logger.info("Diretório /graphs criado.")

        fig, ax = plt.subplots(figsize=(10, 6))

        if g_type == 1:   # BARH
            ax.barh(x_data, y_data, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        elif g_type == 2: # BAR
            ax.bar(x_data, y_data, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        elif g_type == 3: # PIE
            ax.pie(x_data, labels=labels, autopct='%1.1f%%', colors=[color, 'lightblue'])
        
        ax.set_title(title)
        plt.tight_layout()
        
        save_path = os.path.join(f"{path_graphs}", filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico salvo em: {save_path}")
        
        plt.show()
        plt.close(fig) 
        
    @staticmethod
    def build_outliers_report(data, numeric_cols, filename="outliers_analysis.png"):
        """
        Gera um grid de Boxplots com contagem de outliers via Z-Score.
        """
        num_cols_count = len(numeric_cols)
        nrows = (num_cols_count + 1) // 2  
        ncols = 2 if num_cols_count > 1 else 1

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        
        if num_cols_count == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        for idx, col in enumerate(numeric_cols):
            
            sns.boxplot(x=data[col], ax=axes[idx], color='coral')
            
            col_data = data[col].dropna()
            z = np.abs(stats.zscore(col_data))
            outliers_count = (z > 3).sum()
            
            axes[idx].set_title(f"{col} | Outliers (Z>3): {outliers_count}")
            axes[idx].set_xlabel("")

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        
        if not os.path.exists(f"{path_graphs}"): os.makedirs(f"{path_graphs}")
        save_path = os.path.join(f"{path_graphs}", filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        logger.info(f"Relatório de outliers salvo: {save_path}")
        plt.show()
        plt.close(fig)