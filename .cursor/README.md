# Guia de Governanca de Agentes

Este diretorio centraliza contexto operacional para uso consistente de agentes no projeto.

## Estrutura

- `rules/`: regras persistentes e reutilizaveis de padrao tecnico.
- `agents/`: especializacao de papel por dominio.
- `tasks/`: objetivos de trabalho com contexto e criterio de pronto.
- `QUANDO_USAR.md`: guia rapido de decisao para evitar burocracia.

## Como usar

1. Defina primeiro a task em `tasks/`.
2. Selecione o agente por dominio:
   - `api-agent`: endpoints, contratos, validacao e performance de API.
   - `ml-agent`: features, modelagem, avaliacao e metricas de ML.
   - `mlops-agent`: deploy, Docker, CI/CD e operacao.
3. Mantenha escopo claro para evitar sobreposicao de responsabilidade.
4. Para revisoes de ML pos-codificacao, use `tasks/revisao-ml-pos-codificacao.md`.

## Convencoes de Regras

- Regras devem estar em formato `.mdc` com frontmatter.
- Uma regra por preocupacao principal.
- Evite duplicar instrucoes de `rules/` dentro de `agents/`.
- Priorize instrucoes acionaveis e verificaveis.

## Convencoes de Tasks

Use o template `tasks/_TEMPLATE.md` para novas demandas.
Campos obrigatorios:

- Objetivo
- Contexto
- Escopo e restricoes
- Criterio de pronto
- Metricas-alvo
- Plano de execucao

Template adicional:
- `tasks/revisao-ml-pos-codificacao.md`: review pedagogico + profissional para treino/FE/baseline.
- `tasks/comparativo-scripts-ml.md`: comparativo recorrente entre scripts de ML para decisao de adocao.

## Processo de Evolucao

- Ajuste regras quando houver padrao recorrente.
- Ajuste agentes quando houver mudanca de responsabilidade.
- Revise tasks finalizadas para capturar aprendizados reutilizaveis.
