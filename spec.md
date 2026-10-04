# Spec: Projeto de Disciplina, Visão Computacional com CNNs e Transformers

> **Status em 03/10/2026:**
> - **Concluídos:**
>   - os 4 notebooks executados de ponta a ponta, localmente, numa RTX 3050 6 GB;
>   - o relatório PDF (36 páginas);
>   - o ZIP em `entrega/`.
> - **Pendentes, que dependem do aluno:**
>   1. Rodar cada notebook uma vez no Colab T4 ("Executar tudo") para confirmar.
>   2. Confirmar no Moodle o nome da disciplina usado no nome do ZIP.
>   3. Postar o ZIP no Moodle.
> - **Desvios em relação ao plano:**
>   - A2: o corpus final tem 1.501 imagens. Cada foto vinha duplicada como miniatura `_th_`. O τ foi calibrado com as legendas dos participantes.
>   - A4: o DOI citado no enunciado corresponde a Dumakude & Ezugwu (2023).

Lista do que precisa ser feito para a entrega, com as decisões técnicas já tomadas e alinhadas ao que foi ensinado nas aulas. O resumo das aulas está em [resumo_aulas.md](resumo_aulas.md). A referência `[Aula N]` indica de onde vem cada técnica.

Legenda:
- **(obrigatório)**: exigido literalmente pelo enunciado.
- **(decisão)**: escolha técnica tomada aqui, com justificativa.
- **(recomendado)**: reforça a análise, que é o que a avaliação pondera ("executar o código é pré-requisito, não objetivo").

Princípio: usar as ferramentas das aulas (TorchVision Weights API, pacote `clip` da OpenAI, DCGAN/cGAN, `torch-fidelity`). Tudo o que fugir disso entra com justificativa explícita.

---

## 0. Entregáveis

| Entregável | Arquivo |
|---|---|
| Notebook A1 | `notebooks/A1_vision_transformers.ipynb` |
| Notebook A2 | `notebooks/A2_clip_ads16.ipynb` |
| Notebook A3 | `notebooks/A3_cnn_kaggle.ipynb` |
| Notebook A4.1 | `notebooks/A4_estudo_caso_raio_x.ipynb` |
| Relatório (PDF único: A1, A2, A3, A4.1 e A4.2) | `relatorio/lucas_ferreira_deep-learning-and-vision_computer-vision.pdf` |
| Pacote Moodle | `entrega/LucasdeOliveiraFerreira_VisaoComputacionalcomCNNseTransformers_pd.zip` |

A4.2 (tráfego) só entra no relatório, sem notebook.

---

## 1. Decisões fechadas

| # | Decisão | Escolha | Justificativa |
|---|---|---|---|
| D1 | Dataset da A1 | **Casting Product Image Data for Quality Inspection** (Kaggle `ravirajsinh45/real-life-industrial-dataset-of-casting-product`, CC BY-NC-ND 4.0). Usar **só o subconjunto `casting_512x512`**: 1.300 imagens sem augmentation (781 `def_front`, 519 `ok_front`). | Domínio industrial (inspeção de qualidade em manufatura), ligado à atuação em ERP para indústria. O subconjunto 300×300 vem pré-aumentado, o que gera risco de vazamento entre cópias no split; o 512×512 é cru. Dataset pequeno sustenta o argumento de pré-treino + fine-tuning [Aula 04]. Defeitos são locais, o que torna a atenção interpretável. |
| D2 | Corpus da A2 | ADS-16 (Kaggle `groffo/ads16-dataset`). Usar **o corpus completo de imagens**: 300 anúncios (20 categorias × 15) mais as fotos dos 120 usuários (`IM-POS`/`IM-NEG`). Total bem acima de 500, sem subamostragem. Deduplicar por MD5 e ignorar os `.zip` duplicados. | Atende "≥ 500 imagens" sem precisar de amostragem. A análise é feita **separada por origem** (anúncios vs fotos de usuários). **Divergência com o enunciado:** ele fala em 16 categorias, mas o dataset publicado tem 20. Registrar isso no notebook e no relatório. |
| D3 | Download de dados sem editar o notebook | Endpoint público do Kaggle, que funciona **sem credencial**: `curl -L https://www.kaggle.com/api/v1/datasets/download/<owner>/<slug> -o x.zip`. Fallback: `kagglehub.dataset_download`. Cache opcional no Drive. | Testado: o endpoint devolve 302 para uma URL assinada do GCS sem `kaggle.json`. Atende "sem modificações além de montar o Drive". |
| D4 | Dataset da A4.1 | **COVID-19 Radiography Database** (Kaggle `tawsifurrahman/covid19-radiography-database`). Amostrar com seed fixa o cenário do enunciado: **Normal 840, Pneumonia 240 (Viral Pneumonia), COVID-19 120**. | Público, rotulado e com as três classes. Limitação a documentar: não tem ID de paciente, então `GroupKFold` por paciente não é possível [Aula 08]. As fontes heterogêneas da classe COVID abrem risco de shortcut. |
| D5 | Nomes de arquivo | PDF: `lucas_ferreira_deep-learning-and-vision_computer-vision.pdf`. ZIP: `LucasdeOliveiraFerreira_VisaoComputacionalcomCNNseTransformers_pd.zip`. | Formato do enunciado. Disciplina: Visão Computacional com CNNs e Transformers [26E3_3]. |
| D6 | Abordagem generativa da A4.1 | **GAN condicional no estilo DCGAN (cGAN)** condicionada nas 3 classes, imagens 128×128 em tons de cinza. | Arquitetura ensinada [Aula 07]. Condicionar nas 3 classes deixa G aprender a anatomia pulmonar com as ~840 imagens de treino e especializar em COVID pelo rótulo; só 84 imagens COVID não sustentariam uma GAN própria. CycleGAN fica como alternativa discutida no texto. 128×128 é o mínimo razoável para raio-X e cabe no T4. |
| D7 | Biblioteca de modelos | **TorchVision** (ViT-B/16, ResNet-50, ResNet-18) e pacote **`clip` da OpenAI** (ViT-B/32). Não usar `timm`. | É o que as aulas usam [Aula 01, 06, 08]. O hook de atenção da Aula 08 é para o ViT do TorchVision. |
| D8 | Ambiente | Colab, GPU T4, PyTorch com AMP (`torch.autocast`). | Exigência do enunciado. |

---

## 2. Requisitos globais (todos os notebooks)

- [ ] **(obrigatório)** Rodar do início ao fim no Colab T4, sem modificações além de montar o Drive.
- [ ] **(obrigatório)** Primeira célula (markdown) com requisitos de memória (RAM, VRAM, disco) e tempo estimado de execução.
- [ ] **(obrigatório)** Citar as fontes, inclusive as ferramentas de IA (Claude, Anthropic), no notebook e no relatório.
- [ ] Célula de setup:
  - [ ] `pip install` só do que falta no Colab: `git+https://github.com/openai/CLIP.git`, `torch-fidelity`;
  - [ ] checagem de GPU;
  - [ ] seeds fixas (`random`, `numpy`, `torch`, `cudnn.deterministic`).
- [ ] Célula `CONFIG` com todos os hiperparâmetros.
- [ ] Download automático (D3) com verificação de integridade (contagem de arquivos por classe).
- [ ] Markdown explicando **e justificando** cada decisão, com referência à aula.
- [ ] Figuras salvas em `figs/` (PNG, 150 dpi) para reaproveitar no relatório.
- [ ] Notebook entregue **com outputs salvos**.
- [ ] Teste final em runtime T4 limpo ("Executar tudo"). Medir o tempo real e atualizar o cabeçalho.

---

## 3. A1: Vision Transformers em inspeção de qualidade industrial

### 3.1 Dados
- [ ] Baixar o dataset de casting (D3) e usar só `casting_512x512/{def_front, ok_front}`.
- [ ] Exploração:
  - [ ] contagem por classe (781/519, desbalanceio de 60/40);
  - [ ] grade de exemplos de cada classe;
  - [ ] tipos de defeito visíveis (blow holes, rebarbas, encolhimento).
- [ ] Split estratificado 70/15/15 (910/195/195) com seed fixa [Aula 08]. O teste só é usado uma vez, no final.
- [ ] Pré-processamento:
  - [ ] cinza → 3 canais;
  - [ ] resize para 224;
  - [ ] `Normalize` com média e desvio do ImageNet (pré-treino) [Aula 01/08].
- [ ] **(decisão)** Augmentation no treino:
  - [ ] rotação de 0–360° e flips: a peça é vista de cima, é circular, e o defeito pode estar em qualquer posição, então o rótulo é invariante;
  - [ ] `ColorJitter` leve de brilho/contraste (iluminação).
  - Justificar com os riscos discutidos na Aula 08.

### 3.2 Modelo construído sobre os módulos de atenção
- [ ] **(obrigatório)** Implementar do zero, em PyTorch:
  - [ ] `MultiHeadSelfAttention`: Q, K e V, `softmax(QKᵀ/√d_k)·V`, concatenação dos heads, `W_O`, devolvendo também os pesos de atenção [Aula 02];
  - [ ] `PatchEmbedding` com `nn.Conv2d(3, 768, 16, stride=16)` [Aula 04];
  - [ ] `EncoderBlock` Pre-LN (LN → MHSA → residual → LN → MLP 4× GELU → residual) [Aula 02/04];
  - [ ] `ViT`: token [CLS] + position embeddings aprendidos (197×768), 12 blocos, `LayerNorm(z_cls)` → `Linear(768, C)` [Aula 04/05].
- [ ] **(decisão)** Carregar os pesos pré-treinados `ViT_B_16_Weights.IMAGENET1K_V1` do TorchVision **dentro do ViT próprio**, mapeando o `state_dict`. Validar que os logits batem com os de `torchvision.models.vit_b_16` (diferença máxima < 1e-3). Isso mostra que a arquitetura implementada é o ViT-B/16 de fato e dá acesso nativo aos pesos de atenção.

### 3.3 Experimentos (pré-treino vs fine-tuning)
- [ ] **E1, ViT do zero:** ViT pequeno (por exemplo depth 6, dim 192, 3 heads, patch 16) com init aleatória, treinado no dataset. Mostra a "fome de dados" do ViT [Aula 04].
- [ ] **E2, linear probe:** ViT-B/16 pré-treinado congelado, treinando só a head [Aula 03/06].
- [ ] **E3, fine-tuning completo:**
  - [ ] AdamW, LR de 2e-5 a 5e-5 no backbone e 1e-3 na head (LR diferencial);
  - [ ] warmup linear + cosine;
  - [ ] weight decay 0.05;
  - [ ] ~10 épocas;
  - [ ] early stopping pela métrica de validação [Aula 03/08].
- [ ] **E4, ablação de CutMix:** E3 com CutMix [Aula 04].
  - **(decisão)** Hipótese a discutir: em detecção de defeito, o rótulo proporcional à área é semanticamente errado. Um recorte com defeito colado numa peça OK deixa a imagem defeituosa, não "30% defeituosa". Comparar os resultados e concluir.
- [ ] Tabela comparativa E1–E4 com métricas de validação, tempo de treino e número de parâmetros treináveis.
- [ ] Escolher o melhor pela validação e avaliar **uma vez** no teste.

### 3.4 Resultados
- [ ] **(obrigatório)** Métricas no teste:
  - [ ] accuracy;
  - [ ] precision, recall e F1 por classe;
  - [ ] **recall de `def_front`** (taxa de escape de peça defeituosa);
  - [ ] balanced accuracy;
  - [ ] ROC-AUC;
  - [ ] matriz de confusão [Aula 08].
- [ ] **(obrigatório)** Visualizações:
  - [ ] curvas de loss/accuracy por época (treino e validação) de cada experimento;
  - [ ] matriz de confusão;
  - [ ] exemplos de acertos e erros.
- [ ] (recomendado) Análise do threshold de decisão: trade-off entre escape e refugo falso.

### 3.5 Atenção
- [ ] **(obrigatório)** Mapa da atenção [CLS] → patches (`A[:, h, 0, 1:]` → 14×14 → upsampling bilinear para 224 → overlay com alpha 0.5) de **cabeças individuais** da última camada [Aula 05/08].
- [ ] Comparar camadas 1, 6 e 12.
- [ ] Attention rollout [Aula 04].
- [ ] Mapas em peças defeituosas vs peças OK, e antes vs depois do fine-tuning.
- [ ] (recomendado) Entropia da atenção por cabeça e número de patches acima de 2σ [Aula 08].
- [ ] **(obrigatório)** Interpretação escrita:
  - [ ] o que cada cabeça pondera (defeito, borda da peça, furo central, fundo);
  - [ ] se há atenção espúria ou shortcut, como fundo ou bordas [Aula 08].

### 3.6 Justificativa (texto)
- [ ] **(obrigatório)** Por que ViT-B/16 pré-treinado para esse domínio. Contrapor a "regra de ouro" da Aula 04 (CNN com poucos dados).
- [ ] **(obrigatório)** Por que esses hiperparâmetros.
- [ ] **(obrigatório)** O que os resultados revelam.
- [ ] **(obrigatório)** O que mudaria: mais dados reais, validação em linha de produção, DINOv2 como linear probe [Aula 05], resolução maior para defeitos pequenos.

---

## 4. A2: CLIP no ADS-16

**Restrição:** nenhum treino. Só embeddings pré-treinados e consultas em linguagem natural.

### 4.1 Preparação
- [ ] Baixar o ADS-16 (D3). Indexar as imagens com metadados:
  - [ ] origem: `ad` ou `user_pos` / `user_neg`;
  - [ ] categoria do anúncio (1–20);
  - [ ] usuário.
- [ ] Deduplicar por MD5, ignorar `.zip` e registrar o total final.
- [ ] Documentar a divergência 16 vs 20 categorias. Buscar os nomes das 20 categorias no PDF do paper que vem no dataset.
- [ ] **(decisão)** Usar o modelo **CLIP ViT-B/32** (pacote `clip`) [Aula 06/08], com embeddings de imagem normalizados em L2 e guardados em cache (`.npy`).

### 4.2 Item 2.1: ranking de objetos por frequência semântica
- [ ] **(obrigatório)** Definir ≥ 20 conceitos distintos. Proposta de 24:
  - pessoas: a person, a woman, a man, a child;
  - consumo e produto: food, a drink or beverage, a bottle, clothing, shoes, jewelry or a watch, cosmetics or makeup;
  - veículos e tecnologia: a car, a smartphone or electronic device, a computer;
  - lugares e cena: a building or city, outdoor scenery, a beach, home interior or furniture;
  - outros: an animal, sports, money or a credit card, an airplane or travel;
  - publicidade: text and logo, a product on a plain background.
- [ ] **(decisão)** Ensemble de templates por conceito ("a photo of {c}.", "an advertisement showing {c}.", "a picture containing {c}.", …), com `T̄ = Normalize(ΣT_k)` [Aula 06].
- [ ] **(obrigatório)** Matriz de cosine similarity imagens × conceitos.
- [ ] **(obrigatório)** Threshold τ definido e justificado [Aula 08]:
  - [ ] **calibração principal:** τ = percentil 95 da similaridade de pares negativos. Os negativos são pares (anúncio, prompt de categoria errada), usando os rótulos das 20 categorias;
  - [ ] comparar com as faixas da aula: ≥ 0,28 é forte; 0,22 privilegia recall, 0,30 privilegia precisão;
  - [ ] análise de sensibilidade do ranking com τ ∈ {0,22; 0,25; 0,28; P95};
  - [ ] (complementar) verificação manual da precisão numa amostra por conceito.
- [ ] **(obrigatório)** Ranking com frequência (% de imagens com cos ≥ τ) e score médio, no corpus completo e **separado por origem**.
- [ ] **(obrigatório)** Visualizar os 5 conceitos mais frequentes com 4–5 imagens de exemplo cada.
- [ ] (recomendado) Heatmap conceito × categoria de anúncio.
- [ ] Análise crítica:
  - [ ] conceitos genéricos que "ganham sempre" (text and logo em anúncios);
  - [ ] typographic bias;
  - [ ] cosseno não é probabilidade [Aula 08].

### 4.3 Item 2.2: busca semântica por texto
- [ ] **(obrigatório)** Busca texto → imagem retornando as top-5 com `torch.topk` [Aula 06].
- [ ] **(obrigatório)** 10 consultas, variando de genérico a específico e de concreto a abstrato:
  1. "a car"
  2. "a red sports car on the road"
  3. "food"
  4. "a cup of coffee"
  5. "a smiling woman"
  6. "a woman applying makeup"
  7. "a mobile phone advertisement"
  8. "luxury"
  9. "freedom and adventure"
  10. "family happiness"
- [ ] Grade top-5 com score, origem e categoria para cada consulta.
- [ ] **(obrigatório)** Variar o threshold por consulta: quantas imagens passam de τ e o que entra ou sai [Aula 08].
- [ ] **(obrigatório)** Para cada consulta, analisar se recupera o que a consulta descreve ou se interpreta de forma inesperada.
- [ ] (recomendado) Precision@5 anotada manualmente e tabela-resumo por tipo de consulta.
- [ ] Discussão:
  - [ ] o abstrato é resolvido por proxies visuais (luxo vira relógio ou carro);
  - [ ] typographic bias;
  - [ ] viés do pré-treino (WIT-400M).

---

## 5. A3: CNN pré-treinada por feature extraction

### 5.1 Dados
- [ ] Baixar `pavansanagapati/images-dataset` (D3). Estrutura confirmada: `data/{bike 365, cars 420, cats 202, dogs 202, flowers 210, horses 202, human 202}`, total de 1.803 imagens em BMP, JPG e PNG.
- [ ] **Ignorar a pasta duplicada `data/data/`.**
- [ ] Converter para RGB, checar arquivos corrompidos e deduplicar por MD5.
- [ ] Registrar o desbalanceio (cars é 2× cats).
- [ ] Split estratificado 70/15/15 com seed fixa [Aula 08].

### 5.2 Item 3.1: treino
- [ ] **(decisão)** Usar **ResNet-50** (`ResNet50_Weights.IMAGENET1K_V2`) [Aula 01/08]: 2048-d após GAP, cabe no T4 e é o exemplo das aulas. EfficientNet-B0 fica citada como alternativa leve.
- [ ] Pré-processamento: `weights.transforms()` (resize, crop 224, normalização ImageNet) [Aula 01].
- [ ] **(obrigatório)** Congelar todo o backbone (`requires_grad=False`). Manter o backbone em `eval()` para não atualizar as estatísticas do BatchNorm.
- [ ] **(obrigatório)** Trocar `fc` por `nn.Linear(2048, 7)` e treinar só ela, num **único treino**:
  - [ ] Adam com lr 1e-3 [Aula 01];
  - [ ] batch 32;
  - [ ] 15 épocas;
  - [ ] CrossEntropy.
- [ ] (decisão) Extrair as features uma vez e guardar em cache [Aula 03]. Registrar que isso impede augmentation online, o que se liga ao 3.2.
- [ ] **(obrigatório)** Reportar:
  - [ ] accuracy por classe e global;
  - [ ] balanced accuracy [Aula 08];
  - [ ] curvas de loss e accuracy por época (treino e validação);
  - [ ] matriz de confusão;
  - [ ] exemplos de erros.

### 5.3 Item 3.2: propostas de melhoria (texto, apoiado nos erros de 3.1)
Para cada opção: benéfica ou não para estas 7 classes, e **para quais classes distorce** [Aula 08]:
- [ ] **(obrigatório)** Geometric augmentation (flip, rotação leve, recorte):
  - [ ] flip é seguro para todas as classes;
  - [ ] rotação forte é irreal para `human` e `horses` (gravidade);
  - [ ] recorte agressivo pode remover a parte discriminante (rodas em `bike`/`cars`, rosto em `cats`/`dogs`).
- [ ] **(obrigatório)** Color augmentation (jitter, grayscale):
  - [ ] a cor é discriminante para `flowers`;
  - [ ] a pelagem ajuda em `cats` vs `dogs` vs `horses`;
  - [ ] grayscale e matiz forte prejudicam essas classes.
- [ ] **(obrigatório)** Scale variation (`RandomResizedCrop`, multi-scale):
  - [ ] ajuda na variação de enquadramento (carro inteiro vs detalhe);
  - [ ] objetos finos (`bike`) sofrem em escala pequena.
- [ ] **(obrigatório)** Normalização ImageNet:
  - [ ] já usada no 3.1 via `weights.transforms()`;
  - [ ] explicar por que é obrigatória com backbone congelado: o backbone espera a distribuição do pré-treino; sem ela, as ativações saturam e há "bug silencioso" [Aula 01/08].
- [ ] Priorizar as melhorias pelas confusões observadas (provavelmente cats ↔ dogs).
- [ ] Citar o fine-tuning parcial da layer4 com LR diferencial (1e-5 / 1e-3) como próximo passo [Aula 08].

---

## 6. A4.1: Estudo de caso, COVID-19 em raio-X

### 6.1 Dados
- [ ] Baixar a COVID-19 Radiography Database (D4). Amostrar Normal 840, Viral Pneumonia 240 e COVID 120 com seed fixa.
- [ ] Converter para cinza e reduzir para 128×128, para dar a mesma resolução a imagens reais e sintéticas e não criar um atalho de resolução.
- [ ] Ler e citar o artigo indicado (doi:10.1038/s41598-023-37743-4). Atenção: o enunciado o atribui a "Nour & Tariq (2023)", mas o DOI corresponde a Dumakude & Ezugwu (2023), "Automated COVID-19 detection with convolutional neural networks"; registrar a divergência.

### 6.2 Diagnóstico do projeto anterior (≥ 5 problemas)
- [ ] **(obrigatório)** Tabela com as colunas problema | evidência | impacto clínico | correção. Problemas:
  1. ResNet-18 **sem pré-treino** com 1.200 imagens [Aula 01].
  2. **Overfitting** (93% no treino vs 61% na validação).
  3. **Sem augmentation**.
  4. **Desbalanceamento não tratado** (70/20/10).
  5. **Split 80/20 sem estratificação**: ~24 COVID na validação, variando por seed [Aula 08].
  6. **Accuracy global como única métrica**: paradoxo da acurácia, recall de COVID escondido [Aula 07/08].
  7. **SGD com LR fixo de 0,01** sem scheduler nem early stopping; 15 épocas arbitrárias.
  8. Sem conjunto de teste separado.
  9. Possível vazamento por paciente (sem `GroupKFold`) [Aula 08].
  10. Viés de fonte (shortcut) e threshold não calibrado para sensibilidade.

### 6.3 Reprodução do baseline falho
- [ ] **(recomendado)** Reproduzir a configuração original:
  - [ ] `resnet18(weights=None)`;
  - [ ] SGD com lr 0,01 e 15 épocas;
  - [ ] sem augmentation;
  - [ ] 80/20 sem estratificação.
- [ ] Mostrar que a accuracy esconde o recall de COVID (matriz de confusão).

### 6.4 cGAN (DCGAN condicional) [Aula 07]
- [ ] **(obrigatório)** Implementar a cGAN:
  - [ ] G(z ∈ ℝ¹⁰⁰, embedding de y) → ConvTranspose + BN + ReLU → Tanh, saída 1×128×128;
  - [ ] D(x, mapa de y) → Conv stride 2 + LeakyReLU(0.2) + **Spectral Norm** → logit.
- [ ] Estabilização, conforme a aula:
  - [ ] perda não-saturante;
  - [ ] TTUR (η_D = 4e-4, η_G = 1e-4, Adam β = (0.5, 0.999));
  - [ ] one-sided label smoothing (0,9);
  - [ ] instance noise com annealing.
- [ ] Treinar **só com o split de treino**, sem validação nem teste.
- [ ] Monitorar:
  - [ ] curvas de loss de G e D;
  - [ ] grade por classe ao longo das épocas;
  - [ ] **checagem de mode collapse** ("batch clone syndrome") [Aula 07].
- [ ] Qualidade das sintéticas:
  - [ ] FID e KID com `torch-fidelity`, sintéticas COVID vs reais COVID de treino, **reportando N** e o viés de N pequeno [Aula 08];
  - [ ] grade lado a lado de reais vs sintéticas.
- [ ] Salvar o checkpoint de G no Drive (a sessão pode cair).

### 6.5 Experimento com e sem imagens geradas
- [ ] Split **estratificado** 70/15/15: COVID fica 84/18/18. Validação e teste 100% reais [Aula 08].
- [ ] Classificador corrigido, igual em todas as condições:
  - [ ] ResNet-18 **pré-treinada** (`IMAGENET1K_V1`);
  - [ ] entrada de 128 reamostrada para 224, normalização ImageNet;
  - [ ] AdamW com cosine;
  - [ ] early stopping por macro-F1 na validação;
  - [ ] augmentation **sem flip horizontal** (assimetria anatômica [Aula 08]): rotação de ±10°, `RandomResizedCrop` com scale (0,85; 1), brilho e contraste leves.
- [ ] **(obrigatório)** Condições:
  - **A:** só reais.
  - **B1:** reais + 252 COVID sintéticas (3× a classe).
  - **B2:** reais + ~500 COVID sintéticas (≈ balanceado).
  - (referência sem GAN) **C:** reais + loss com class weights.
- [ ] 3 seeds por condição, média ± desvio, mais IC por bootstrap do recall no teste.
- [ ] **(obrigatório)** Métrica principal: **recall de COVID** e **ΔRecall = Recall_B − Recall_A** [Aula 08].
- [ ] Também reportar:
  - [ ] precision, recall e F1 por classe;
  - [ ] especificidade;
  - [ ] balanced accuracy;
  - [ ] macro-F1;
  - [ ] matrizes de confusão;
  - [ ] curvas de treino base vs aumentado [Aula 08].
- [ ] Análise crítica:
  - [ ] o ganho é real ou ruído (teste com 18 COVID: cada caso vale 5,6 p.p.);
  - [ ] risco de as sintéticas virarem atalho;
  - [ ] FID vs ganho downstream.

### 6.6 Plano de melhoria integrado (texto)
- [ ] **(obrigatório)** Cobrir **todos** os problemas de 6.2:
  - [ ] pré-treino e transfer learning;
  - [ ] augmentation clinicamente válida;
  - [ ] balanceamento (GAN + pesos);
  - [ ] split estratificado e por paciente;
  - [ ] métricas clínicas (recall/sensibilidade como métrica principal, threshold calibrado para sensibilidade-alvo);
  - [ ] scheduler e early stopping;
  - [ ] teste externo multi-hospital;
  - [ ] auditoria de shortcut (mapas de atenção ou Grad-CAM);
  - [ ] validação com o time clínico.

---

## 7. A4.2: Transfer learning em tráfego urbano (só no relatório)

- [ ] Ler a referência da Meegle e citá-la.
- [ ] **(obrigatório)** ≥ 4 problemas, cada um com impacto operacional e com como abordar:
  1. **Covariate shift** (chuva, noite, névoa) ausente do treino. Abordagem:
     - coleta direcionada;
     - augmentation fotométrica;
     - **CycleGAN dia → noite / seco → chuva** [Aula 07];
     - teste de estresse OOD [Aula 08].
  2. **Vazamento no split**: frames da mesma câmera e próximos no tempo em treino e validação, então os 78% são otimistas. Abordagem: **split por câmera (`GroupKFold`)** [Aula 08].
  3. **Generalização de viewpoint**: só 12 câmeras, ângulos novos falham. Abordagem: mais câmeras, validação leave-camera-out, normalização de perspectiva.
  4. **Gap de tarefa**: o ImageNet é centrado em objeto, e congestionamento é propriedade da cena e do **tempo** (um único frame não mede fluxo). Abordagem:
     - detecção + tracking (YOLO + DeepSORT) para contar e medir velocidade [Aula 01];
     - modelos temporais;
     - backbones hierárquicos para veículos pequenos [Aula 05].
  5. **Dataset pequeno e rótulo ordinal ambíguo** (moderado vs congestionado). Abordagem:
     - protocolo de anotação e concordância entre anotadores;
     - formulação ordinal;
     - active learning.
  6. **Sem monitoramento em produção.** Abordagem: detecção de drift e OOD, alarme de confiança baixa, retreino periódico.
  7. **Fine-tuning completo com pouco dado degrada a robustez.** Abordagem: linear probe ou fine-tuning parcial com LR diferencial [Aula 06/08].
- [ ] Sem implementação.

---

## 8. Relatório técnico (PDF único)

- [ ] **(obrigatório)** Cobrir A1, A2, A3, A4.1 e A4.2. Para cada atividade:
  - [ ] definição do problema;
  - [ ] decisões técnicas e justificativa;
  - [ ] resultados com métricas e gráficos;
  - [ ] análise crítica.
- [ ] Estrutura: capa → sumário → introdução → A1 → A2 → A3 → A4.1 → A4.2 → conclusão → declaração de uso de IA → referências.
- [ ] Declaração de uso de IA: ferramentas, finalidade e como os outputs foram verificados.
- [ ] Referências:
  - [ ] papers: Vaswani 2017; Dosovitskiy 2021; Radford 2021 (CLIP); He 2016 (ResNet); Goodfellow 2014; Mirza & Osindero 2014; Radford 2016 (DCGAN); Zhu 2017 (CycleGAN); Heusel 2017 (FID); Yun 2019 (CutMix); Nour & Tariq 2023; Roffo & Vinciarelli 2016 (ADS-16); Chowdhury 2020 / Rahman 2021 (COVID-19 Radiography DB);
  - [ ] link da Meegle;
  - [ ] datasets do Kaggle com licença.
- [ ] Números do relatório idênticos aos outputs dos notebooks.

---

## 9. Empacotamento e entrega

- [ ] Rodar os 4 notebooks em runtime T4 limpo e salvar com outputs.
- [ ] Atualizar o cabeçalho de memória e tempo com os valores medidos.
- [ ] Gerar o PDF.
- [ ] Montar o ZIP com os 4 notebooks e o PDF. Conferir a listagem.
- [ ] Postar no Moodle.

---

## 10. Ordem de execução

1. A3: valida o download anônimo do Kaggle e o pipeline TorchVision.
2. A2: sem treino, rápido.
3. A1: experimentos E1–E4 e atenção.
4. A4.1: cGAN + experimento; maior risco de tempo.
5. A4.2: texto.
6. Relatório.
7. Execução limpa, ZIP e entrega.
