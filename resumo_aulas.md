# Resumo das 8 aulas: Visão Computacional com CNNs e Transformers (INFNET)

Fonte: `aula_01_apresentacao.pdf` a `aula_08_apresentacao.pdf`. Os PDFs são capturas de slides interativos (sem camada de texto), lidos página a página. Como só aparece a aba ativa de cada slide, alguns conteúdos (abas ocultas, gabaritos de quiz) não estão nos PDFs.

Resumos gerados com apoio de IA (Claude, Anthropic) e revisados contra os slides; citar no relatório.

---

## Aula 01: Arquiteturas CNN clássicas e modernas, transfer learning e além da classificação (23 p.)

**Conteúdo**
- Platô das redes rasas e paradoxo da degradação (rede plain de 56 camadas pior que a de 20). Gradiente que desvanece.
- Quatro pilares: bottleneck 1×1, conexões residuais (`h(x) = f(x) + x`, gradiente com termo "+1"), conv depthwise separable (~1/9 do custo com K=3) e atenção aos canais (SENet).
- Evolução: LeNet-5, AlexNet, GoogLeNet/Inception, VGG, ResNet, Xception, DenseNet, MobileNet, EfficientNet e ConvNeXt.
- Escolha da CNN pela "Tabela 12-3" do TorchVision: EfficientNet-B0 tem 77,7% Top-1, 5,3M parâmetros e 0,4 GFLOPs.
- Memória de GPU: ResNet-50 com batch 64 usa ~8,5 GB, 84% em ativações. Contra OOM: reduzir o batch ou usar gradient checkpointing.
- Além da classificação: classificação + localização, detecção two-stage (Faster R-CNN) vs one-stage (YOLO/SSD), tracking (DeepSORT), segmentação (U-Net, FCN).

**Código e receitas**
- API Weights do TorchVision: `resnet34(weights=ResNet34_Weights.DEFAULT)`. Substitui `pretrained=True`.
- Feature extraction: congelar tudo (`requires_grad = False`), trocar `model.fc` e treinar com `torch.optim.Adam(model.fc.parameters(), lr=1e-3)`.
- Pré-processamento: sempre `weights.transforms()`, que aplica a média e o desvio do ImageNet. Pré-processamento errado gera "bugs silenciosos".

**Uso no projeto:** A3 (receita direta), A4.2 (YOLO/DeepSORT, MobileNet/EfficientNet para edge).

---

## Aula 02: Atenção, tokenização BPE e o bloco Transformer completo (26 p.)

**Conteúdo**
- BPE, embeddings densos e similaridade de cosseno. Posição somada ao embedding, senoidal ou aprendida.
- Scaled dot-product: `Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`. Sem a escala √d_k, o softmax vira degrau e mata o gradiente.
- Projeções W_Q, W_K, W_V, W_O. Multi-head com `d_k = d_model/h = 768/12 = 64`. Máscara causal (não se aplica a ViT/BERT).
- Self-attention vs cross-attention.
- Bloco Pre-LN: `x → LN → MHA → +skip → LN → MLP(d→4d→d, GELU) → +skip`.
- Custo O(T²) da matriz de atenção. Dois terços dos parâmetros ficam no MLP.

**Código**
- Classe `SelfAttention(nn.Module)` com `nn.Linear(d_model, d_k, bias=False)` para Q, K e V.
- `scores = Q @ K.transpose(-2,-1) / math.sqrt(d_k)`, depois `torch.softmax` e `A @ V`.
- Máscara: `torch.tril` + `masked_fill`.

**Uso no projeto:** A1 (implementar o módulo de atenção do zero e o heatmap T×T de pesos), A2 (embeddings e cosseno).

---

## Aula 03: BERT e modelos encoder-only (20 p.)

**Conteúdo**
- Atenção bidirecional vs causal. Embeddings token + segmento + posição.
- Pré-treino MLM (15% mascarado) + NSP. Cabeças plugáveis: `h[:,0,:]` ([CLS]) → Dropout(0.1) → `Linear(768, K)`.
- BERT-Base: 12 camadas, 768 dimensões, 12 heads, 110M parâmetros. Variantes: RoBERTa, DistilBERT, DeBERTa.
- **Fine-tuning vs feature extraction:**
  - Fine-tuning completo: AdamW, LR de 2e-5 a 5e-5, warmup, 2 a 4 épocas. LR alto causa catastrophic forgetting.
  - Feature extraction: backbone congelado, embeddings pré-calculados e guardados, classificador leve (LogReg/SVM/MLP). Indicado com poucos dados ou GPU limitada.

**Uso no projeto:** A1 (o ViT é encoder-only com a mesma mecânica de [CLS]; receita de fine-tuning com LR baixo e warmup), A3 (feature extraction com embeddings em cache).

---

## Aula 04: Vision Transformers (ViT) e classificação visual (16 p.)

**Conteúdo**
- Usar pixel como token é inviável: 224² = 50.176 tokens, o que dá OOM. Por isso a imagem vira patches 16×16: `N = HW/P² = 196`, mais o [CLS], total de 197 tokens com D = 768.
- Patch embedding via `nn.Conv2d(3, 768, kernel_size=16, stride=16)`, algebricamente igual a flatten + projeção linear.
- [CLS] aprendido + position embeddings 1D aprendidos (`E_pos ∈ ℝ^(197×768)`).
- Encoder Pre-LN:
  - `z'_ℓ = MSA(LN(z_ℓ₋₁)) + z_ℓ₋₁`
  - `z_ℓ = MLP(LN(z'_ℓ)) + z'_ℓ`
  - MLP 768 → 3072 → 768.
- Viés indutivo: CNN tem localidade e equivariância e converge com poucos dados. ViT tem viés quase nulo e "fome de dados". No ImageNet-1k o ViT perde ~4% para a CNN; com JFT-300M, vence.
- Com poucos dados: pré-treino em escala + regularização pesada com **CutMix/Mixup**. CutMix usa rótulo proporcional à área, `ŷ = λ·y_A + (1−λ)·y_B`.
- **Attention Rollout:** atenção do [CLS] sobre os patches por profundidade (blocos 1–3, 6–8, 11–12). A explicabilidade é nativa: basta ler a matriz de atenção no forward.
- Tabela: ViT-B/16 (86,6M parâmetros, 17,6 GFLOPs, 81,2% Top-1) é o "cavalo de batalha da indústria". ResNet-50: 76,1%.
- "Regra de ouro": com dataset médio e hardware restrito, CNN ou híbridos dominam.

**Uso no projeto:** A1 (aula central: arquitetura, ViT-B/16, CutMix e rollout).

---

## Aula 05: ViTs avançados (DeiT, PVT, Swin, DINO) (17 p.)

**Conteúdo**
- Revisão do pipeline do ViT-B/16 com shapes. Head: `LayerNorm(z_cls)` → `Linear(768, C)`.
- Três barreiras do ViT: fome de dados, escala rígida (não serve para FPN) e custo O(N²).
- Respostas da literatura:
  - DeiT: token de destilação; CNN professora congelada; hard distillation melhor que soft.
  - PVT: pirâmide + SRA (reduz K e V por R²).
  - Swin: janelas W-MSA/SW-MSA com M=7; custo linear.
  - DINO: auto-supervisão professor/aluno com EMA, centering + sharpening, multi-crop.
  - MAE: 75% de máscara.
  - DINOv2: linear probe sobre features congeladas.
  - ConvNeXt: ResNet-50 modernizada.
- Visualização: atenção do [CLS] na última camada, **por cabeça**. Cada cabeça foca numa parte do objeto. O ViT supervisionado mostra atenção difusa; o DINO mostra máscara nítida.
- Alerta: objeto pequeno cabe num único token 16×16 e perde contorno.

**Uso no projeto:** A1 (visualização por cabeça, argumento de pré-treino), A4.2 (objetos pequenos, backbones hierárquicos).

---

## Aula 06: CLIP, Contrastive Language-Image Pre-training (17 p.)

**Conteúdo**
- Vocabulário fechado (softmax de 1000 classes) vs aberto. Duas torres: imagem (ViT/ResNet) e texto (Transformer, 77 tokens).
- Projeção para D = 512 e **normalização L2** obrigatória (hiperesfera).
- Loss InfoNCE simétrica: `L = ½(L_img + L_txt)`. Temperatura aprendível `logit_scale = log(1/0.07)`.
- Zero-shot: embeddings de texto das classes calculados uma vez; `logits = Î·T̂ᵀ/τ`.
- **Prompts:**
  - "a photo of a {c}." no lugar da palavra solta.
  - Desambiguação, por exemplo "..., a type of bird".
  - **Ensemble de templates:** `T̄ = Normalize(Σ T_k)`. No ImageNet, Top-1 vai de 67,5% (palavra solta) a 68,8% (template) e 72,5% (ensemble).
- Robustez OOD: CLIP ViT-L/14 vs ResNet-50 no ImageNet-A, 77,1% vs 2,7%.
- Adaptação: Linear Probe (recomendado primeiro em domínio especializado), Full FT (LR 1e-5, perde robustez OOD) e CoOp.
- Anomalias zero-shot: prompts opostos ("pristine" vs "damaged"), com threshold que troca recall por falso alarme.
- Filtro LAION: descartar pares com cos < 0.28.

**Código (notebook da aula)**
- `pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git`
- `image_features /= image_features.norm(dim=-1, keepdim=True)`
- `scores, indices = torch.topk(image_features @ query_feat.T, k=5)`
- Indexação com Faiss ou Qdrant.

**Uso no projeto:** A2 (aula central: CLIP ViT-B/32, L2, templates, top-k).

---

## Aula 07: GANs, da DCGAN à CycleGAN (15 p.)

**Conteúdo**
- Jogo minimax: `min_G max_D E[log D(x)] + E[log(1−D(G(z)))]`. Equilíbrio de Nash em `D* = 1/2`.
- **Perda não-saturante obrigatória:** G minimiza `−log D(G(z))`.
- **DCGAN:**
  - z ∈ ℝ¹⁰⁰ → ConvTranspose2d (k4 s2 p1) + BN + ReLU → … → Tanh em 64×64.
  - D: Conv2d stride 2 + LeakyReLU(0.2), sem BN na entrada, Sigmoid na saída.
  - 5 regras de Radford: stride no lugar de pooling; BN exceto nas saídas; sem camadas densas; ReLU/Tanh em G; LeakyReLU em D.
- **Estabilização:**
  - TTUR (η_D = 4e-4 > η_G = 1e-4).
  - One-sided label smoothing (y_real = 0.9; nunca y_fake > 0).
  - Instance noise com annealing.
  - Spectral Normalization.
- **Mode collapse:** o recall generativo cai enquanto a precision se mantém. Sintoma: "batch clone syndrome" no grid.
- **cGAN:** G(z, y) com embedding de y [B,16,1,1]. D(x, y) com mapa espacial de y.
- **CycleGAN:** tradução sem pares, `L_cyc = ‖F(G(x))−x‖₁ + ‖G(F(y))−y‖₁`, `L_idt`. Padrão λ_cyc = 10, λ_idt = 5.
- Métricas IS e FID. A validação pelo ganho de **Recall downstream** é "mandatória". No lab, o baseline sem GAN tem 89,5% de acurácia e recall 58% (63 FN).

**Uso no projeto:** A4.1 (aula central: cGAN/DCGAN, estabilização, mode collapse, recall).

---

## Aula 08: Avaliação de modelos generativos (FID) e síntese da disciplina (23 p.)

**FID**
- `‖μ_r−μ_g‖² + Tr(Σ_r+Σ_g−2(Σ_rΣ_g)^½)` sobre features de 2048 dimensões (pool3 do Inception-v3).
- Viés amostral: `E[FID_N] = FID_∞ + C/N`. Sempre reportar N.
- Bibliotecas: `torch-fidelity` e `clean-fid`. Usar resize bicúbico consistente.

**Pontos de atenção por atividade**
- **Atenção no ViT:**
  - Hook em `model.encoder.layers[-1].self_attention` (ViT do torchvision).
  - `A[:, head, 0, 1:].view(14,14)`, upsampling bilinear para 224 e overlay com alpha 0.5.
  - Inspecionar por cabeça e por camada (1/6/12) e procurar atenção espúria (marca d'água, cantos, fundo).
  - **Requisito:** documentar o que ao menos uma cabeça revela.
- **CLIP:**
  - Cosseno não é probabilidade. Ruído entre 0,12 e 0,22; 0,20–0,24 é alinhamento fraco; ≥ 0,28 é forte.
  - **Calibrar τ pelo percentil 95 de pares negativos aleatórios.** τ = 0,30 privilegia precisão; τ = 0,22, recall.
  - Template "a clear photo of [objeto]".
  - **Requisito:** variar o threshold em ≥ 8 consultas.
- **Transfer learning:**
  - Feature extraction (congelar, `nn.Linear(2048, C)`) é ideal com dataset pequeno.
  - Fine-tuning parcial: layer4 com LR 1e-5 e head com 1e-3, AdamW.
- **Augmentation:**
  - `RandomHorizontalFlip`, `RandomRotation(15)`, `RandomResizedCrop(224, scale=(0.8,1))`, `ColorJitter(0.2,0.2)`, `RandomGrayscale(0.1)`.
  - Normalização: sempre a do pré-treino, `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`.
  - **Flip em imagem médica assimétrica corrompe o rótulo** (coração à esquerda). Jitter de cor corrompe quando a cor é discriminante.
- **Métricas:**
  - Acurácia global vs por classe e **balanced accuracy**. Exemplo: 88,2% de acurácia vs 57% de B-Acc.
  - Paradoxo da acurácia: em dados 90/10, prever sempre a maioria dá 90% de acurácia e recall 0.
  - Primazia do recall em triagem.
- **GAN para augmentation, 3 regras:** sintéticas **só no treino**, teste 100% real, reportar **ΔRecall = Recall_comGAN − Recall_semGAN** e comparar curvas base vs aumentada.
- **Divisão:** `train_test_split(..., stratify=y)`. Usar `GroupKFold` por paciente, câmera ou sessão para evitar vazamento.
- **Domain shift:** covariate shift (chuva, noite, névoa, ângulos novos). Remédio: teste de estresse OOD e split por câmera.

**Uso no projeto:** todas as atividades. É a aula que amarra os requisitos do projeto.

---

## Mapa aula → atividade

| Atividade | Aulas-base | O que usar |
|---|---|---|
| A1 ViT | 02, 03, 04, 05, 08 | Atenção do zero (02), receita de FT (03), ViT-B/16 + CutMix + rollout (04), atenção por cabeça (05), hook e overlay (08) |
| A2 CLIP | 06, 08 | ViT-B/32, L2, ensemble de prompts, top-k (06); calibração de τ por P95 e variação do threshold (08) |
| A3 CNN | 01, 03, 08 | Weights API + `weights.transforms()` + congelar + Adam 1e-3 (01); embeddings em cache (03); augmentations e seus riscos, B-Acc (08) |
| A4.1 Raio-X | 07, 08 | cGAN/DCGAN + estabilização (07); FID com N, ΔRecall, sintéticas só no treino, sem flip, split estratificado/por paciente (08) |
| A4.2 Tráfego | 01, 05, 06, 07, 08 | YOLO/DeepSORT (01); objetos pequenos (05); OOD e forgetting (06); CycleGAN para tradução de domínio (07); split por câmera e teste OOD (08) |

Lacunas (temas que as aulas não cobrem; usar só com justificativa):
- focal loss e class weights;
- Grad-CAM (citado apenas como contraste na aula 04);
- `timm` e Hugging Face para ViT (as aulas usam TorchVision e o pacote `clip` da OpenAI).
