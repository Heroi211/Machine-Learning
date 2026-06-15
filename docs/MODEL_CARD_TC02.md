# Model Card — Recomendação E-commerce (TC02)

> Domínio: `recommendation` · Dataset: MovieLens ml-latest-small

## Identificação

| Campo | Valor |
|-------|--------|
| Nome | `tc02_recommender` |
| Tipo | Recomendação user-item (embedding PyTorch) |
| Baselines | Popularity, NMF (sklearn) |
| Métricas | Hit Rate@K, Precision@K, Recall@K, NDCG@K, MAP@K |

## Uso pretendido

Ranquear produtos (itens) para um `user_id` com base em interações históricas de rating/clique.

## Limitações

- Cold-start: utilizadores/itens novos caem no fallback de popularidade.
- Split temporal por utilizador (última interação em teste) — não simula catálogo dinâmico completo.
- MovieLens substitui catálogo e-commerce real (aceito pelo enunciado como alternativa).

## Vieses

- Popularidade domina utilizadores com pouco histórico.
- Distribuição geográfica/temporal do MovieLens ≠ e-commerce atual.

## Reprodução

```bash
pip install -e ".[dev,tc02]"
dvc repro
python scripts/ml/promote_registry.py
```
