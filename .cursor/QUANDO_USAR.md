# Quando Usar Rules, Agents e Tasks

## Tabela Rapida

| Item | O que e | Quando usar | Quando evitar |
|---|---|---|---|
| `rules/` | Regras persistentes do projeto | Quando uma orientacao se repete em varias tarefas | Em demandas pontuais ou unicas |
| `agents/` | Especializacao por dominio (ML/API/MLOps) | Quando precisa de profundidade tecnica por contexto | Em ajustes triviais sem analise |
| `tasks/` | Plano de execucao de uma demanda | Quando ha escopo, etapas e necessidade de acompanhamento | Em perguntas rapidas ou mudancas pequenas |

## Regra 80/20

- Pergunta rapida: use agente, sem task.
- Mudanca pequena e isolada: sem task formal.
- Mudanca media/grande: criar task.
- Problema recorrente: transformar em rule.
- Assunto especializado: usar agente do dominio.

## Heuristica de Esforco

- Ate 30 min e sem impacto sistemico: **sem task**.
- 30 min a 2h com 2+ etapas: **task curta recomendada**.
- Acima de 2h, risco de regressao, ou impacto em pipeline/rota critica: **task obrigatoria**.

## Fluxo de Decisao (Checklist)

1. O problema e recorrente?
   - Sim: criar/ajustar `rules/`.
2. Exige profundidade tecnica especifica?
   - Sim: acionar `agents/` do dominio.
3. Ha mais de uma etapa, dependencia, ou risco?
   - Sim: criar `tasks/`.
4. E algo simples e local?
   - Sim: resolver direto, sem task.

## Exemplos Praticos

- Ajustar threshold de churn apos experimento rapido:
  - Use `ml-agent`, sem task (se for ajuste simples).
- Refatorar pipeline de FE com varias transformacoes:
  - Use `ml-agent` + criar task.
- Nova rota `predict` com contrato e validacao:
  - Use `api-agent` + criar task.
- Padrao repetido de metricas em reviews:
  - Registrar como nova regra em `rules/`.

## Convencao Recomendada

- Antes de criar artefato novo, pergunte:
  - "Isso vai se repetir?"
  - "Isso precisa ser rastreado?"
  - "Isso depende de mais de uma etapa?"
- Se duas respostas forem "sim", formalize em `task` ou `rule`.
