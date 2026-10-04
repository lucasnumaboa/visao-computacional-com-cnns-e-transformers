# 2. Atividade 1: Vision Transformers em inspeção visual de qualidade (fundição)

## 2.1 Definição do problema

**Domínio: indústria, controle de qualidade.** O objetivo é classificar rotores de bombas submersíveis produzidos por fundição (vista superior) em **defeituoso** (`def_front`: porosidade, rebarbas, falhas de borda) ou **OK** (`ok_front`). Hoje a inspeção é manual, lenta e sujeita a fadiga, e deixar passar uma peça defeituosa (falso negativo) pode levar à rejeição de um lote inteiro. O tema se liga à minha atuação com ERP para indústria, em que os resultados de inspeção alimentam o módulo de controle de qualidade.

**Dataset:** *Casting Product Image Data for Quality Inspection* (Kaggle, CC BY-NC-ND 4.0). Download: <https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product>. **Decisão:** usar só o subconjunto `casting_512x512`, com **1.300 fotos originais** (781 defeituosas e 519 OK). O conjunto 300×300, de 7.348 imagens, **já vem com augmentation aplicada** pelo autor, então dividi-lo colocaria versões da mesma peça no treino e no teste, um vazamento. Fizemos nosso próprio split estratificado 70/15/15 (910 / 195 / 195) e nossa própria augmentation, só no treino.

**Achado na exploração: um possível atalho.** As fotos foram tiradas sobre dois tipos de fundo. **79% das defeituosas estão sobre fundo cinza e 73% das OK sobre fundo branco.** Um modelo poderia acertar boa parte olhando só o fundo (*shortcut*, Aula 08). Por isso o split foi estratificado também por tipo de fundo, e o atalho foi testado explicitamente na seção 2.4.

![Brilho do fundo por classe e proporção de tipo de fundo.|80%](../figs/A1/01_fundo_por_classe.png)

## 2.2 Decisões técnicas e justificativa

**Atenção do zero, em módulos testáveis (Aula 02).** Implementamos três módulos PyTorch:

- `ScaledDotProductAttention`: `softmax(QKᵀ/√d_k)·V`, com máscara opcional.
- `AttentionHead`: uma cabeça com **projeções W_Q, W_K, W_V próprias e independentes**.
- `MultiHeadAttention`: *h* cabeças independentes, **concatenação** e projeção `W_O`.

Seis testes automatizados (`assert`) passam:

| # | Teste | O que garante |
|---|---|---|
| 1 | formas `[B,T,D]` / `[B,h,T,T]` e linhas do softmax somando 1 | contrato do módulo |
| 2 | `ScaledDotProductAttention` = `F.scaled_dot_product_attention` | a fórmula está correta |
| 3 | `MultiHeadAttention` = `nn.MultiheadAttention` (pesos copiados cabeça a cabeça) | projeções por cabeça + concatenação + W_O corretas |
| 4 | posições mascaradas recebem peso 0 | o mecanismo de máscara (o mesmo da `attention_mask` do BERT) |
| 5 | `MHA(Px) = P·MHA(x)` | **equivariância a permutação**: sem PE, a atenção ignora a ordem |
| 6 | versão fundida usada no ViT = versão de cabeças independentes | o ViT usa exatamente esse mecanismo |

**Do bloco ao ViT (Aulas 02, 04 e 05):**

- `MultiHeadSelfAttention` é a versão **fundida** das mesmas cabeças: W_Q, W_K, W_V de todas as cabeças empilhadas numa `Linear(D, 3D)`. Faz 3 multiplicações grandes em vez de 3·h pequenas, no formato dos pesos do TorchVision. O teste 6 prova a equivalência.
- `PatchEmbedding`: `Conv2d(3, 768, 16, stride=16)`, que transforma a imagem de 224 px em 196 patches.
- `TransformerEncoderBlock` Pre-LN: `z' = MSA(LN(z)) + z` e `z = MLP(LN(z')) + z'`, com *feedforward* de duas camadas (`Linear(D,4D)` → GELU → `Linear(4D,D)`), dois `LayerNorm` e duas conexões residuais.
- `ViT`: token [CLS] aprendível + *position embeddings* aprendidos (197 × 768), *N* blocos, `LayerNorm` e cabeça `Linear(768, C)`. A saída são os logits.

**Positional encoding: por que é indispensável.** Pelo teste 5, a atenção trata os tokens como um **conjunto**. O [CLS] agrega os patches com pesos que dependem só do conteúdo, então uma imagem e a sua versão "quebra-cabeça embaralhado" produziriam **exatamente a mesma** representação. O modelo perderia vizinhança (contornos que atravessam patches) e localização (defeito na borda vs no cubo). O ViT soma `E_pos ∈ ℝ^(197×768)` aos tokens. O teste no ViT-B/16 real confirma, e mostra também que os embeddings 1D aprendidos recuperam a estrutura 2D:

| Patches embaralhados antes da posição | Máx. diferença no [CLS] | Cosseno original × embaralhada |
|---|---|---|
| **sem** E_pos | 0,000003 (arredondamento) | **1,000**: o modelo não percebe que a imagem foi desmontada |
| **com** E_pos | 3,51 | **0,15**: a representação muda completamente |

![Cosseno entre o embedding de posição marcado (+) e os das 196 posições: o ViT aprende linhas e colunas sem receber geometria.|100%](../figs/A1/02b_pos_embed_similaridade.png)

**Pré-treino dentro da implementação própria.** Os pesos `ViT_B_16_Weights.IMAGENET1K_V1` do TorchVision foram carregados **na nossa implementação** por mapeamento do `state_dict` (`in_proj` → `qkv`, `out_proj` → `proj`). A diferença máxima de logits em relação ao `torchvision.models.vit_b_16` foi de **3,8 × 10⁻⁶**. A arquitetura escrita **é** o ViT-B/16, e os pesos de atenção ficam acessíveis sem *hooks*.

**Por que ViT-B/16 pré-treinado e ajustado por fine-tuning:**

- A Aula 04 mostra que o ViT tem viés indutivo quase nulo e "fome de dados". O pré-treino em escala é o que o viabiliza com 910 imagens.
- A atenção global é natural para o problema: o defeito pode estar **em qualquer ponto do anel**, e o [CLS] pondera todos os patches desde o primeiro bloco.
- A explicabilidade é **nativa**: os pesos de atenção saem do próprio forward.
- A "regra de ouro" da Aula 04 (com poucos dados, prefira CNN) foi tratada como **hipótese a testar**, por isso o experimento E1 abaixo.

**Experimentos:**

| | Experimento | O que testa |
|---|---|---|
| E1 | ViT pequeno **do zero** (dim 192, 6 blocos, 3 cabeças, 2,9 M parâmetros), 40 épocas | Fome de dados do ViT |
| E2 | ViT-B/16 **congelado** + cabeça linear (*linear probe*) | Qualidade das features do ImageNet neste domínio |
| E3 | ViT-B/16 com **fine-tuning completo** | Adaptação ao domínio industrial |
| E4 | E3 **+ CutMix** | A regularização recomendada para ViT com poucos dados serve para defeitos locais? |
| E5 | **Swin-T** pré-treinado, fine-tuning | O que um Transformer hierárquico de janelas locais muda aqui |
| E6 | **ResNet-50** pré-treinada (CNN), fine-tuning | ViT vs CNN com o mesmo pré-treino, dados e receita |

**Hiperparâmetros de fine-tuning (E3) e por quê:**

| Hiperparâmetro | Valor | Justificativa |
|---|---|---|
| LR backbone / cabeça | 3e-5 / 1e-3 (AdamW) | Faixa 2e-5–5e-5 evita *catastrophic forgetting* (Aula 03). A cabeça nova precisa de LR maior (Aula 08). |
| Warmup + cosine | 1 época | Protege o backbone dos gradientes da cabeça aleatória no início. |
| Épocas / seleção | 10 / maior balanced accuracy de validação | As classes são 60/40 e os erros têm custo assimétrico. |
| Weight decay / clipping | 0,05 / 1,0 | Regularização padrão de fine-tuning de ViT. |
| Batch efetivo | 32 | Micro-batch 16 + acumulação em GPUs com menos de 10 GB, para reproduzir a dinâmica do T4. |
| Augmentation | Grupo **D4** (rotações de 90°, flips) + brilho/contraste | A peça é circular e vista de cima, então o rótulo é invariante. As transformações são **sem perda** (sem cantos pretos de rotação arbitrária). Sem recorte, que pode remover o defeito de borda. |

## 2.3 Resultados

| Experimento | Parâm. treináveis | Tempo | Val bal. acc | Teste acc | Teste bal. acc | Recall defeito (teste) | Escapes / refugos | AUC teste |
|---|---|---|---|---|---|---|---|---|
| E1: ViT do zero | 2,86 M | 133 s | 0,833 | 78,5% | 79,9% | 72,6% | 32 / 10 | 0,857 |
| E2: linear probe | 1.538 | 9 s | 0,987 | 96,9% | 97,2% | 95,7% | 5 / 1 | 0,994 |
| **E3: fine-tuning** | 85,8 M | 182 s | **1,000** | **100%** | **100%** | **100%** | **0 / 0** | **1,000** |
| E4: fine-tuning + CutMix | 85,8 M | 185 s | 0,996 | 99,0% | 99,1% | 98,3% | 2 / 0 | 0,999 |
| E5: Swin-T, fine-tuning | 27,5 M | 130 s | 0,996 | 99,0% | 99,1% | 98,3% | 2 / 0 | 0,999 |
| E6: ResNet-50, fine-tuning | 23,5 M | 92 s | 0,996 | 99,5% | 99,6% | 99,1% | 1 / 0 | 1,000 |

A escolha do modelo foi feita **pela validação, entre os ViTs (E1–E4)**: o escolhido é o E3. O teste dos demais é reportado só para comparação.

![Curvas de loss e accuracy (treino tracejado, validação contínua) e balanced accuracy de validação por época, para os 6 experimentos.|100%](../figs/A1/03_curvas_experimentos.png)

**Teste do modelo escolhido (E3):**

<span class="kpi">Accuracy <b>100%</b></span> <span class="kpi">Balanced accuracy <b>100%</b></span> <span class="kpi">Recall defeito <b>100%</b></span> <span class="kpi">AUC <b>1,000</b></span> <span class="kpi">Escapes (FN) <b>0/117</b></span> <span class="kpi">Refugos (FP) <b>0/78</b></span>

![Matriz de confusão, curva ROC e curva precisão × recall no teste (E3).|100%](../figs/A1/04_teste_confusao_roc_pr.png)

**Leitura do resultado.** 100% em 195 imagens significa, pela regra de três, **taxa de erro abaixo de ~1,5%** (IC 95%), e não erro zero. A validação também está saturada, então não há como refinar o limiar com estes dados. Baixá-lo só acrescenta refugos:

| Limiar | Recall defeito | Escapes | Refugos |
|---|---|---|---|
| 0,5 | 100% | 0 | 0 |
| 0,2 | 100% | 0 | 1 |
| 0,05 | 100% | 0 | 2 |
| 0,01 | 100% | 0 | 6 |

Um limiar operacional, definido pelo custo de um escape contra o de um refugo, exige uma validação maior.

## 2.4 Visualização e interpretação dos pesos de atenção

Os mapas mostram `A[:, h, 0, 1:]`: quanto o [CLS] da cabeça *h* pondera cada um dos 196 patches. O mapa é reorganizado em 14×14, ampliado por interpolação bilinear e sobreposto com alfa 0,5 (Aulas 05/08).

### ViT treinado do zero (E1): regiões emergentes

![ViT do zero: atenção do [CLS] nas 3 cabeças do bloco 6, média do bloco 3 e rollout.|92%](../figs/A1/06a_atencao_vit_do_zero.png)

- **Surgem detectores locais.** Mesmo com 910 imagens, nas peças defeituosas as 3 cabeças do bloco 6 produzem pontos nítidos sobre o **anel externo**, e em várias peças (por exemplo, a do furo no topo) caem sobre o defeito. O **rollout forma um anel** em torno do cubo: o modelo descobriu sozinho a geometria circular da peça.
- **Nas peças OK a atenção é difusa e invade o fundo.** As cabeças 0 e 2 cobrem o fundo branco, e as probabilidades ficam incertas (0,20–0,29). É sinal de que o modelo usa **fundo e iluminação global** como pista, justamente o atalho correlacionado com a classe. Sem pré-treino, o ViT pequeno mistura evidência local com contexto, e por isso deixa escapar 32 defeitos e refuga 10 peças boas.

### ViT pré-treinado e ajustado (E3)

![As 12 cabeças da última camada numa peça com falha de borda (canto inferior esquerdo da peça).|100%](../figs/A1/06_atencao_12_cabecas_ultima_camada.png)

![Camadas 1, 6 e 12 (média das cabeças) e attention rollout (Aula 04).|78%](../figs/A1/08_atencao_por_camada_e_rollout.png)

![As 12 cabeças da camada 6 nas peças defeituosas.|100%](../figs/A1/10_camada6_cabecas_defeitos.png)

**O que o modelo aprende a ponderar, por profundidade:**

- **Camada 1:** atenção difusa e local, em faixas que acompanham bordas e contrastes (baixo nível).
- **Camada 6: é onde o defeito aparece.** Várias cabeças (por exemplo 2, 6 e 10) concentram a atenção em **pontos isolados do anel externo** que coincidem com furos e falhas de borda. Na peça com o furo no topo, as cabeças 2, 4, 6, 9, 10 e 11 marcam exatamente o furo. Há **divisão de trabalho entre cabeças**, como na Aula 05: a cabeça 5 se especializa no **cubo central** e a cabeça 0 contorna o **anel das pás**.
- **Camada 12: defeito + "registradores".** Na peça com falha de borda, as cabeças **0, 2, 4, 5 e 9** marcam exatamente a falha. Ao mesmo tempo, quase todas as cabeças põem parte da atenção em **3 a 5 patches de fundo nos cantos**. A cabeça 7, a mais concentrada (entropia de 3,07 nats, contra 5,28 se fosse uniforme), põe **69% da atenção nos cantos** nas peças defeituosas e 47% nas OK, contra 20% esperados pela área.

![Cabeça 7 da última camada antes (ImageNet) e depois do fine-tuning.|100%](../figs/A1/07_cabeca_escolhida_antes_depois.png)

![Perfil radial da atenção, fração da atenção nos cantos e entropia por cabeça (todo o teste).|100%](../figs/A1/09_atencao_quantitativa.png)

**A atenção nos cantos é atalho de fundo? Dois testes:**

1. **Origem do padrão.** A atenção nos cantos **já existe no ViT pré-treinado no ImageNet**, antes de ver qualquer peça. É o fenômeno de Darcet et al. (2024, *Vision Transformers Need Registers*): ViTs supervisionados reaproveitam patches de fundo, de baixa informação, como memória global.
2. **Contrafactual.** Pintamos os cantos (fora do círculo inscrito) de **branco** e, depois, de **cinza**, sem tocar na peça:

| Cenário | Accuracy | Recall defeito | Especificidade | Predições alteradas |
|---|---|---|---|---|
| original | 100% | 100% | 100% | — |
| cantos brancos | 97,4% | 95,7% | 100% | 5 |
| cantos cinza | 97,9% | 96,6% | 100% | 4 |

Branco e cinza têm efeitos na mesma direção e de tamanho parecido, então **a cor do fundo não decide a classe**. Mas apagar a região dos cantos, com qualquer cor, faz o modelo perder alguns defeitos: os "registradores" guardam informação agregada que o [CLS] usa. O E3 acerta 100% nos quatro subgrupos classe × tipo de fundo.

**Conclusão da interpretação:** o modelo **decide olhando a peça** (a camada 6 e algumas cabeças da 12 localizam o defeito), mas usa a região do fundo como memória. Uma mudança de fundo em produção (outra bancada, outro enquadramento) é um risco real que precisa ser testado antes da implantação.

## 2.5 Pré-treino do BERT × pré-treino do ViT: o que cada um maximiza

Os dois são encoders Transformer quase idênticos (Aula 03): mesmo bloco Pre-LN, 12 camadas, D = 768, 12 cabeças, [CLS] e posições aprendidas. A diferença está no **objetivo de pré-treino**:

| | **BERT** (texto) | **ViT** original (imagem) |
|---|---|---|
| Supervisão | **Auto-supervisionada**: o rótulo é o próprio texto | **Supervisionada**: rótulos humanos (ImageNet-21k / JFT-300M) |
| O que maximiza | **MLM**: `Σ log P(token mascarado | contexto bidirecional)` em 15% dos tokens, + **NSP** | `log P(classe | imagem)` sobre o [CLS] (cross-entropy) |
| O que a representação captura | Contexto e semântica **de cada token**: sintaxe, sentido, co-referência | Features **discriminativas para a taxonomia de rótulos**. O que não ajuda a separar classes é descartado. |
| Dados | Texto bruto (3,3 B palavras), sem anotação | Centenas de milhões de imagens **rotuladas** |

**Consequências observadas neste trabalho:**

- O ViT supervisionado não precisa de atenção espacialmente interpretável, só de um [CLS] separável. Por isso a última camada usa patches de fundo como memória global, o artefato de "registradores" da seção 2.4.
- A "versão BERT" da visão (**MAE/BEiT**: mascarar 75% dos patches e **reconstruí-los**) e a auto-destilação (**DINO**) produzem representações mais genéricas e mapas de atenção que segmentam o objeto (Aula 05).
- O **CLIP** (A2) maximiza o alinhamento imagem–texto (InfoNCE).
- Para defeitos, que são texturas fora da taxonomia do ImageNet, um backbone auto-supervisionado (DINOv2/MAE) tende a transferir melhor.

## 2.6 DeiT e Swin: o que resolvem que o ViT original não resolve

A Aula 05 lista três barreiras do ViT original: **fome de dados**, **escala única** (grade 14×14 em todas as camadas) e **custo O(N²)**.

- **DeiT (Touvron et al., 2021): fome de dados.** Treina o ViT **só no ImageNet-1k**, sem JFT, com augmentation pesada (RandAugment, Mixup, CutMix) e **destilação por token**: um token `[DIST]` imita uma CNN professora congelada. A destilação *hard* funciona melhor que a soft. Assim o ViT herda o viés de localidade da CNN sem tê-lo na arquitetura (top-1 de 79,9% para 85,2%). **Aqui:** seria a forma de treinar um ViT com poucas peças sem depender de pré-treino gigante. O E1 (ViT do zero) mostra o tamanho do problema que o DeiT ataca.
- **Swin (Liu et al., 2021): escala e custo.**
  - Atenção dentro de **janelas locais** M×M (M = 7), com custo **linear** no número de patches.
  - **Janelas deslocadas** a cada bloco (*cyclic shift* + máscara), para que janelas vizinhas troquem informação.
  - **Patch merging**, que gera uma pirâmide H/4 → H/32 e dá features multi-escala, como uma CNN.

  **Aqui:** permite entrada em 512 px, preservando defeitos de poucos pixels sem OOM, e features multi-escala para **localizar** o defeito. O E5 testa o Swin-T com a mesma receita.

## 2.7 Quando ViT supera CNN, e quando não: evidência neste domínio

Os três modelos pré-treinados foram ajustados com **a mesma receita** (E3, E5, E6, na tabela da seção 2.3):

- **Com pré-treino, as três arquiteturas empatam.** ViT-B/16 teve 0 erros, ResNet-50 teve 1 e Swin-T teve 2, em 195 imagens: diferença dentro do ruído amostral. Aqui, **o pré-treino importa muito mais que a arquitetura**.
- **Sem pré-treino, o ViT colapsa** (78,5%). É a "regra de ouro" da Aula 04: com poucos dados, o viés indutivo da CNN (localidade, equivariância) é uma vantagem, e o ViT só a compensa com pré-treino em escala ou destilação (DeiT).
- **CNN é preferível** para **inspeção em linha com hardware restrito** (PC industrial, *edge*): a ResNet-50 entrega o mesmo resultado com **¼ dos parâmetros** e metade do tempo de treino. Também se os dados fossem ainda mais escassos e não houvesse pré-treino adequado.
- **ViT/Swin são preferíveis:**
  - quando a **explicabilidade** importa: o ViT aponta o defeito na atenção sem custo extra;
  - quando há **mais dados**: pela curva de escala da Aula 04, com pré-treino maior o ViT passa a CNN;
  - (Swin) quando é preciso **subir a resolução** ou **localizar/segmentar** o defeito: custo linear e features multi-escala.
- **Escolha para este domínio:** o **ViT-B/16 pré-treinado**. Teve o melhor resultado, dentro do empate técnico, e é o único com mapas de atenção nativos que apontam o defeito. Para implantação embarcada, a ResNet-50 é a alternativa de menor custo com a mesma acurácia.


## 2.8 Análise crítica

- **O pré-treino é o fator dominante.**
  - Do zero, o ViT chega a 0,83 de balanced accuracy na validação e 78,5% no teste.
  - Congelado, com 1.538 parâmetros treinados, chega a 0,987 na validação.
  - Com fine-tuning, chega a 1,000 na validação e 100% no teste.
- **CutMix não ajudou, e a hipótese se sustenta.** O E4 teve 2 escapes contra 0 do E3, e accuracy de treino muito menor (0,88), por causa dos rótulos misturados inconsistentes. Em **anomalias locais**, o rótulo proporcional à área é semanticamente errado.
- **Arquitetura vs pré-treino.** ViT, Swin e ResNet-50 pré-treinados empatam (0–2 erros). A arquitetura importa menos que o pré-treino e que o protocolo de avaliação.
- **Limitações:**
  - teste de 195 imagens (erro < 1,5% com 95% de confiança, não erro zero);
  - validação saturada, sem limiar operacional;
  - correlação fundo↔classe inerente ao dataset, e o modelo é sensível a alterações na região do fundo (contrafactual).
- **O que eu mudaria:**
  1. Padronizar o fundo na captura, ou segmentar a peça e mascarar o fundo antes de classificar.
  2. Ampliar a validação para definir o limiar por custo.
  3. Usar ViT/DINOv2 **com register tokens**, para mapas de atenção limpos.
  4. Usar Swin ou resolução de 384–512 px para defeitos de poucos pixels.
  5. Em produção, enviar os casos com probabilidade intermediária para inspeção humana.
