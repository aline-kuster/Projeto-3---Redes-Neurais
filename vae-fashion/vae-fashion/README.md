# Projeto 3 — VAE convolucional no Fashion-MNIST

Autora: Aline Breda Kuster

Treina autoencoders variacionais **convolucionais** com espaço latente de 2 e de 8
dimensões, escolhe a topologia de cada um por busca aleatória de hiperparâmetros
(critério: −ELBO no conjunto de validação) e analisa o espaço latente.

## Estrutura

```
src/
  data.py      # download, normalização e divisão estratificada 54k / 6k / 10k
  model.py     # ConvVAE e a função de custo (-ELBO = BCE + KL)
  search.py    # busca aleatória de hiperparâmetros + treino final (z = 2 e z = 8)
  analyze.py   # métricas, gráficos do latente, PCA, K-means, classificador linear
results/
  search.json  # todas as configurações testadas e as curvas de treino
  metrics.json # métricas finais usadas no relatório
  figs/        # figuras do relatório
report/
  relatorio.tex / relatorio.pdf
```

## Como rodar

```bash
pip install torch torchvision scikit-learn scipy matplotlib
python src/search.py --trials 8 --search-epochs 10 --final-epochs 60 --patience 6
python src/analyze.py
```

Roda em CPU (cerca de 1 h no total com 2 núcleos). Se o download automático do
torchvision falhar, coloque os quatro arquivos `.gz` do Fashion-MNIST
(repositório zalandoresearch/fashion-mnist) em `data/FashionMNIST/raw/`.

## Espaço de busca

| Hiperparâmetro | Valores |
|---|---|
| Filtros das convoluções (camada 1 / camada 2) | 16/32, 32/64 |
| Largura da camada densa | 128, 256, 512 |
| Taxa de aprendizado (Adam) | 5e-4, 1e-3, 2e-3 |
| Tamanho do lote | 64, 128 |

São sorteadas 8 combinações para cada dimensão latente (sorteios independentes),
cada uma treinada por 10 épocas. A melhor de cada dimensão é treinada de novo por
até 60 épocas, com parada antecipada (paciência 6) na −ELBO de validação.
