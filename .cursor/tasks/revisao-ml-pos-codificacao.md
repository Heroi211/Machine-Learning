# Revisao ML Pos-Codificacao

## Objetivo
- Receber feedback tecnico pedagogico e profissional sobre treino, FE e baseline.

## Contexto
- Alteracoes recentes feitas no pipeline de ML.
- Interesse em entender o por que das decisoes e o que tem maior valor de mercado.

## Escopo e Restricoes
- Escopo: qualidade de dados, FE, treino, avaliacao e baseline.
- Fora de escopo: mudancas de API, deploy e infraestrutura.

## Criterio de Pronto
- Feedback estruturado com prioridades claras.
- Explicacao didatica das principais recomendacoes.
- Decisao objetiva do que implementar agora vs depois.

## Metricas-Alvo
- Principal: recall da classe positiva (quando aplicavel).
- Suporte: F1, precision, PR-AUC e estabilidade entre folds.
- Operacional: simplicidade de manutencao e reprodutibilidade.

## Plano de Execucao
1. Revisar diffs e arquitetura atual do pipeline.
2. Avaliar qualidade metodologica (split, leakage, baseline, validacao).
3. Avaliar qualidade de features e custo operacional.
4. Priorizar melhorias por impacto x esforco.
5. Fechar com plano de evolucao em curto prazo.

## Riscos e Mitigacoes
- Risco: foco excessivo em tecnicas avancadas sem consolidar baseline.
- Mitigacao: priorizacao por retorno pratico e estabilidade.
