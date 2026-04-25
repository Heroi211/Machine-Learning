# Melhorar Recall de Churn

## Objetivo
- Aumentar o recall do modelo de churn sem degradar F1 de forma relevante.

## Contexto
- Dataset desbalanceado com target `churn`.
- Baseline atual apresenta baixa cobertura de positivos.

## Escopo e Restricoes
- Pode alterar feature engineering, estrategia de balanceamento e hiperparametros.
- Nao alterar contratos de API nem infraestrutura.

## Criterio de Pronto
- Recall superior ao baseline em validacao cruzada.
- Resultado reprodutivel com configuracao documentada.

## Metricas-Alvo
- Metrica principal: recall da classe positiva (churn).
- Metricas de suporte: F1, precision e PR-AUC.
- Limites minimos aceitaveis: F1 nao deve cair abaixo do baseline.

## Plano de Execucao
1. Medir baseline com split estratificado e registrar metricas.
2. Testar balanceamento (class_weight, undersampling ou oversampling).
3. Revisar features com foco em sinal preditivo para churn.
4. Rodar busca de hiperparametros e calibrar threshold.
5. Comparar experimentos e selecionar melhor trade-off.

## Riscos e Mitigacoes
- Risco: overfitting em minoria por oversampling agressivo.
- Mitigacao: validacao cruzada estratificada e controle de complexidade.