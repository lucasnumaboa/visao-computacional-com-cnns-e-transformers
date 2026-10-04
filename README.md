# Visão Computacional com CNNs e Transformers: Projeto de Disciplina

**INFNET · Visão Computacional com CNNs e Transformers [26E3_3]**
**Aluno:** Lucas de Oliveira Ferreira

Este repositório tem os quatro notebooks do projeto, o relatório técnico em PDF e os scripts auxiliares. Todos os notebooks rodam **do início ao fim no Google Colab com GPU T4**, sem montar o Drive e sem `kaggle.json`: os datasets são baixados automaticamente do endpoint público do Kaggle.

| Atividade | Notebook | Abrir no Colab | Tempo (T4) |
|---|---|---|---|
| **A1:** Vision Transformers em inspeção de qualidade (peças fundidas) | [`A1_vision_transformers.ipynb`](notebooks/A1_vision_transformers.ipynb) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucasnumaboa/visao-computacional-com-cnns-e-transformers/blob/main/notebooks/A1_vision_transformers.ipynb) | ~20–30 min |
| **A2:** CLIP no ADS-16 (ranking de conceitos e busca semântica) | [`A2_clip_ads16.ipynb`](notebooks/A2_clip_ads16.ipynb) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucasnumaboa/visao-computacional-com-cnns-e-transformers/blob/main/notebooks/A2_clip_ads16.ipynb) | ~5–8 min |
| **A3:** CNN pré-treinada por feature extraction | [`A3_cnn_kaggle.ipynb`](notebooks/A3_cnn_kaggle.ipynb) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucasnumaboa/visao-computacional-com-cnns-e-transformers/blob/main/notebooks/A3_cnn_kaggle.ipynb) | ~4–6 min |
| **A4.1:** Estudo de caso COVID-19 em raio-X + GAN condicional | [`A4_estudo_caso_raio_x.ipynb`](notebooks/A4_estudo_caso_raio_x.ipynb) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucasnumaboa/visao-computacional-com-cnns-e-transformers/blob/main/notebooks/A4_estudo_caso_raio_x.ipynb) | ~45–55 min |
| **A4.2:** Transfer learning em tráfego urbano | só no relatório (análise escrita, sem código) | — | — |

📄 **Relatório técnico:** [`relatorio/lucas_ferreira_deep-learning-and-vision_computer-vision.pdf`](relatorio/lucas_ferreira_deep-learning-and-vision_computer-vision.pdf)
✅ **Rubrica × evidência:** [`rubrica_checklist.md`](rubrica_checklist.md)

## Como rodar

### No Google Colab (recomendado)
1. Clique em **Abrir no Colab** na tabela acima.
2. `Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU`.
3. `Ambiente de execução → Executar tudo`.

Não é preciso montar o Drive nem ter credencial do Kaggle. Os notebooks já estão salvos com as saídas da execução de referência, então dá para ler os resultados sem rodar. O cabeçalho de cada notebook informa memória e tempo estimados.

### Localmente
```bash
python -m venv .venv
.venv/Scripts/activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
jupyter notebook notebooks/
```
Os dados vão para `./data` (configurável com a variável de ambiente `DATA_ROOT`) e as figuras para `./figs/<atividade>` (`FIG_DIR`). Em GPUs com menos de 10 GB, o A1 mantém o batch efetivo de 32 por acumulação de gradiente. No Windows, os `DataLoader` rodam sem workers.

## Principais resultados

| Atividade | Resultado |
|---|---|
| A1 | Atenção e ViT implementados do zero (6 testes passando). Fine-tuning do ViT-B/16 pré-treinado: **100% no teste** (195 imagens). ViT do zero: 78,5%. Swin-T: 99,0%. ResNet-50: 99,5%. Os mapas de atenção da camada 6 localizam os defeitos. |
| A2 | Zero-shot de 59,8% (top-1) em 20 categorias de anúncio. Busca legenda→foto com **R@1 de 39,9%** (acaso: 0,08%). Threshold calibrado em τ = 0,234 pelo P95 de pares negativos. |
| A3 | ResNet-50 congelada + `Linear(2048, 7)`: **99,62%** no teste. O teste de estresse revela atalhos de aquisição (`bike` cai a 38,5% em baixa resolução). |
| A4.1 | Baseline reproduzido: accuracy de 89,6%, mas **recall de COVID-19 de 41,7%**. Pipeline corrigido: 82,6–92,6%. As sintéticas da cGAN **não** melhoraram o recall (ΔRecall com IC). A ablação de estabilidade mostra divergência na versão de livro e minibatch-std contra o *mode collapse* (KID −24%). |

## Estrutura

```
notebooks/            os 4 notebooks entregues (com saídas)
notebooks/src/        fonte de cada notebook em formato "percent" (.py), mais fácil de revisar e versionar
tools/py2nb.py        converte notebooks/src/*.py → .ipynb
tools/run_nb.py       executa um notebook célula a célula, com log, salvando após cada célula
tools/sync_md.py      atualiza só o texto (markdown) de um notebook já executado
tools/nb_tail.py      mostra as últimas saídas de um notebook (acompanhar execuções longas)
tools/build_report.py gera o PDF do relatório (Markdown → HTML → PDF via Edge headless)
relatorio/            PDF do relatório e as seções em Markdown
figs/                 figuras geradas pelos notebooks (usadas no relatório)
spec.md               plano do projeto e decisões técnicas
resumo_aulas.md       resumo das 8 aulas usado para alinhar as decisões ao conteúdo da disciplina
rubrica_checklist.md  onde cada item da rubrica é demonstrado
```

## Datasets (baixados em tempo de execução, não redistribuídos aqui)

| Atividade | Dataset | Licença |
|---|---|---|
| A1 | [Casting Product Image Data for Quality Inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) (subconjunto `casting_512x512`) | CC BY-NC-ND 4.0 |
| A2 | [ADS-16 Computational Advertising Dataset](https://www.kaggle.com/datasets/groffo/ads16-dataset) (Roffo & Vinciarelli, 2016) | conforme a descrição no Kaggle |
| A3 | [Images Dataset](https://www.kaggle.com/datasets/pavansanagapati/images-dataset) | CC0 |
| A4.1 | [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (Chowdhury et al., 2020; Rahman et al., 2021) | © autores originais |

As figuras e saídas dos notebooks mostram amostras desses datasets, apenas para fins acadêmicos.

## Uso de IA

Projeto desenvolvido com apoio do **Claude (Anthropic)**, via Claude Code, conforme a política "Sinal Verde" da disciplina. A IA foi usada para resumir as aulas, planejar, escrever e depurar o código e redigir as análises. Todos os notebooks foram executados de ponta a ponta, e os números do relatório são as saídas reais dessas execuções. A declaração completa está na seção 8 do relatório.

Os slides das aulas não estão incluídos, porque são material da INFNET.
