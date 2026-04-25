# Comparativo Tecnico de Scripts ML

## Objetivo
- Comparar dois ou mais scripts de ML para avaliar qualidade tecnica, aderencia a boas praticas e viabilidade de adocao em producao.

## Contexto
- Esta task deve ser usada em revisoes recorrentes de scripts (experimentos, pipelines, preprocessamento, treino, avaliacao).
- O foco e decidir o que incorporar, o que ajustar e o que descartar com base em criterios objetivos.

## Escopo e Restricoes
- Escopo: ingestao, limpeza, FE, modelagem, avaliacao, robustez, reproducibilidade.
- Fora de escopo: alteracoes de API, deploy e infraestrutura (salvo quando explicitamente solicitado).
- Nao implementar codigo durante a etapa de comparacao, exceto quando solicitado.

## Criterio de Pronto
- Diferencas tecnicas mapeadas com impacto e risco.
- Recomendacoes priorizadas por custo-beneficio.
- Decisao final clara: adotar, adaptar ou descartar cada abordagem.

## Metricas-Alvo
- Qualidade metodologica: ausencia de leakage, validacao adequada e baseline consistente.
- Qualidade de avaliacao: metrica principal bem definida + metricas de suporte.
- Qualidade operacional: reproducibilidade, rastreabilidade e facilidade de manutencao.

## Plano de Execucao
1. Mapear objetivos e escopo de cada script comparado.
2. Fazer comparativo por dimensoes tecnicas.
3. Classificar diferencas por impacto, risco e esforco de adocao.
4. Priorizar recomendacoes (quick wins vs mudancas estruturais).
5. Fechar com decisao orientada e proximo experimento.

## Dimensoes Minimas de Comparacao
- Ingestao e saneamento de dados
- Tratamento de tipos e missing
- FE e consistencia de transformacoes
- Split, validacao e controle de leakage
- Modelo, hiperparametros e calibracao
- Metricas e criterio de decisao
- Logging, rastreabilidade e reproducibilidade
- Complexidade de manutencao e aderencia ao mercado

## Formato Padrao de Saida
1. Resumo executivo (maximo 10 linhas)
2. Comparativo por dimensao (A vs B)
3. Tabela de diferencas (impacto, risco, recomendacao)
4. Lacunas frente a boas praticas de ML
5. Plano de acao priorizado
6. Decisao final (adotar/adaptar/descartar) + proximo experimento

## Prompt Base Reutilizavel
Use este prompt no chat ao anexar os scripts:

"""
@.cursor/agents/ml-agent.md
@.cursor/rules/ml-modeling.mdc
@.cursor/rules/ml-review-feedback.mdc
@<script_a>
@<script_b>

Modo: review_hibrido

Compare tecnicamente os scripts anexados e avalie:
- aderencia a conceitos de ML
- qualidade metodologica
- robustez para producao
- custo-beneficio de adocao

Nao implemente codigo.
Entregue no formato padrao da task comparativo-scripts-ml.
"""
