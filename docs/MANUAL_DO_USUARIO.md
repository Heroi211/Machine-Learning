# Manual do utilizador — fluxo até à predição

Este guia descreve, em linguagem simples, **o que fazer na aplicação** desde o primeiro acesso até obter uma **predição** (resultado do modelo). Os caminhos técnicos (URLs e nomes de campos) correspondem à API do projeto; o prefixo **`{versão}`** é o valor configurado no ambiente (ex.: `/v1`) — substitua pelo que a sua equipa indicar.

---

## 1. Duas funções no sistema

| Papel | O que faz |
|--------|-----------|
| **Administrador** | Treina modelos com os dados, escolhe qual modelo fica **em uso** para cada tema (domínio) e gere utilizadores/perfis, quando aplicável. |
| **Utilizador consumidor** | Usa o modelo **já** preparado pelo administrador para pedir **predições** (não escolhe ficheiros de treino nem treina modelos). |

Quem entra na aplicação **só** como consumidor precisa de **conta** e do **modelo já ativo** para o tema em que vai prever.

---

## 2. Primeiro acesso: registo e entrada

1. **Registo** (criar conta) — quem ainda não tem utilizador.  
   - O sistema regista nome, e-mail e palavra-passe.  
   - Por defeito, a conta nova é de **utilizador normal** (não administrador).

2. **Entrar (login)** — depois de ter conta.  
   - O sistema devolve um **token de acesso** (chave temporária) que deve ser usado nas **próximas ações** até expirar.

3. **Quem é administrador?**  
   - Na base de dados, o perfil de administrador está associado ao papel **“Administrador”**.  
   - Só um administrador existente (ou a equipa técnica) pode alterar o perfil de um utilizador para administrador. **Não** é automático no registo.

---

## 3. Fluxo do **administrador** (preparar o modelo)

Objetivo: ter um **modelo treinado** e **ativo** para um **domínio** (tema), por exemplo `heart_disease`. O **domínio** deve ser **o mesmo nome** que usam no treino (objetivo) e depois na predição.

### Passo A — Treinar (opcional: baseline; depois ou direto: engenharia de features)

O administrador envia dados em ficheiro **CSV** e indica o **objetivo** (domínio).  
A ordem típica académica é:

1. **Treino baseline** (opcional) — primeiro processamento a partir dos dados brutos.  
2. **Treino de engenharia de features (FE)** — trabalho mais completo sobre dados já adequados; pode usar o CSV pré-processado.  
   - Pode indicar a **métrica** que importa para escolher o melhor modelo (ex.: recall em vez de acurácia).

O sistema guarda o resultado no servidor e **devolve** também um ficheiro CSV na resposta, com informação de metadados nos cabeçalhos da resposta.

### Passo B — Promover o modelo a “em produção”

Quando um treino termina **com sucesso**, o sistema regista um **identificador do treino** (run).  
O administrador usa a ação **Promover** e indica:

- o **domínio** (tema) — igual ao **objetivo** usado nesse treino;  
- o **identificador do treino** que quer tornar o modelo oficial.

Só assim esse modelo passa a ser o usado nas **predições** para aquele domínio. Sem este passo, quem tentar prever recebe mensagem de que **não há modelo ativo**.

---

## 4. Fluxo do **utilizador consumidor** (pedir uma predição)

**Pré-requisito:** existir conta e **token** válido; e o administrador já ter **promovido** um modelo para o **domínio** em que se quer prever.

1. O utilizador chama a ação de **predição**.  
2. Envia **em JSON** (dados estruturados):  
   - **domínio** — o mesmo tema do modelo (ex.: `heart_disease`);  
   - **características** — os valores que descrevem o caso (ex.: idade, medidas clínicas), no formato que o modelo espera.

3. O sistema responde com a **predição** (e, se aplicável, uma **probabilidade**), e regista o pedido para auditoria.

Se o domínio **não** tiver modelo ativo, a resposta indica que não há modelo disponível — o consumidor deve contactar o administrador.

---

## 5. Resumo visual do fluxo

```
[Administrador]  Registo/login → Treina (baseline/FE) → Promove modelo para um domínio
                                              ↓
[Consumidor]       Registo/login → Predição (domínio + características) → Resultado
```

---

## 6. O que o consumidor **não** faz

- Não escolhe qual “treino interno” usar — isso fica definido pela **promoção** feita pelo administrador.  
- Não envia ficheiros de treino na predição — só os **dados** da instância a classificar.

---

## 7. Dúvidas frequentes (em linguagem simples)

- **“Porque dá erro de modelo não encontrado?”**  
  O administrador ainda não **promoveu** um treino concluído para esse domínio, ou o domínio está escrito de forma diferente (maiúsculas/minúsculas são normalizadas, mas o nome deve ser o correto).

- **“O que é domínio?”**  
  O **tema** do problema (ex.: doença cardíaca) — tem de ser **consistente** entre treino, promoção e predição.

- **“Preciso de ser administrador para prever?”**  
  Não. Basta **conta** e **modelo ativo** para o domínio.

---

## 8. Manutenção, logs e observabilidade

O manual acima cobre o **fluxo funcional**. Para **onde estão os ficheiros de log**, como é medida a **latência** dos pedidos e **quando** se analisa **drift** de dados (não é automático em cada predição), consulte **[Observabilidade e manutenção](OBSERVABILIDADE_E_MANUTENCAO.md)**. Em resumo: não existe um painel de administrador só para ver logs na própria API — o registo está no **servidor** (JSONL de acesso, ficheiros de pipeline) e relatórios opcionais vêm de **scripts** na pasta `scripts/maintenance/`.

---

*Manual alinhado ao comportamento da API (rotas de autenticação, processador e papéis). Para detalhes técnicos de cada campo, consulte a documentação interativa da API (Swagger/OpenAPI), se estiver ativa no ambiente.*
