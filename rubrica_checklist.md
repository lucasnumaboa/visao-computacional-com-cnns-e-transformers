# Rubrica × onde está a evidência

Cada item da rubrica, com o notebook (NB) e a seção do relatório (PDF) onde ele é demonstrado.

## 1. Transfer learning com CNNs pré-treinadas (A3)

| Item | Onde |
|---|---|
| CNN pré-treinada, head substituído pelo nº de classes, backbone congelado | NB A3 §4–5 (`resnet50` + `requires_grad=False` + `eval()` + `Linear(2048, 7)`; conferência de que imagem e cache dão a mesma predição) · PDF 4.2 |
| Curvas de treino + accuracy por classe e global | NB A3 §5 (curvas, tabela por classe, matriz de confusão) · PDF 4.3 |
| ≥ 3 estratégias de augmentation justificadas para o domínio | NB A3 §7–8 (teste de estresse + 4 estratégias com as classes afetadas) · PDF 4.4 |
| Feature extraction × fine-tuning conforme tamanho e domínio | NB A3 §8.3 (matriz 2×2 + evidências da A1 e da A4) · PDF 4.4 |
| Escolha do modelo considerando o T4 e o nº de classes | NB A3 §4 (tabela de 4 candidatas, VRAM medida de 0,49 GB, 7 classes) · PDF 4.2 |

## 2. Transformer: da atenção ao BERT (A1, A2)

| Item | Onde |
|---|---|
| Scaled dot-product e multi-head attention do zero, testáveis, com projeções por cabeça e concatenação | NB A1 §2.1 (`ScaledDotProductAttention`, `AttentionHead`, `MultiHeadAttention` + 6 testes `assert`) · PDF 2.2 |
| Heatmap de attention weights + interpretação | NB A1 §7 (cabeças, camadas, rollout, contrafactual) e §9.4 · PDF 2.4 |
| TransformerEncoderBlock completo (FFN de 2 camadas, LayerNorm, residual) como base do ViT | NB A1 §2.2 (`TransformerEncoderBlock`) · PDF 2.2 |
| Positional encoding e por que atenção sem PE não preserva posição | NB A1 §2.1 (teste 5) e §2.3 (patches embaralhados: cosseno 1,000 sem PE vs 0,15 com PE; mapas de similaridade das posições) · PDF 2.2 |
| Pré-treino do BERT × do ViT (o que cada um maximiza) | NB A1 §9.5 · PDF 2.5 |

## 3. Vision Transformers (A1)

| Item | Onde |
|---|---|
| Patch embedding + CLS aprendível + PE → ViT completo (imagem → logits) | NB A1 §2.2 (`PatchEmbedding`, `ViT`; validado contra o TorchVision: 3,8e-6) · PDF 2.2 |
| ViT do zero + attention maps de ao menos uma cabeça + regiões emergentes | NB A1 E1 + §7.1 + §9.4 · PDF 2.4 ("ViT treinado do zero") |
| Fine-tuning do ViT pré-treinado, head substituído, comparação com o do zero em tabela | NB A1 E3 + tabelas de validação/teste · PDF 2.3 |
| DeiT e Swin: o que resolvem | NB A1 §9.6 (+ experimento E5 com Swin-T) · PDF 2.6 |
| Quando ViT supera CNN e quando a CNN é preferível, no domínio | NB A1 §9.7 (E3 vs E5 Swin-T vs E6 ResNet-50, mesma receita) · PDF 2.7 |
| Tabela ViT do zero × pré-treinado + escolha de arquitetura justificada | NB A1 §5/§6 · PDF 2.3 e 2.7 |

## 4. CLIP (A2)

| Item | Onde |
|---|---|
| Alinhamento visual–textual e por que o pré-treino contrastivo habilita busca sem treino | NB A2 §5 (InfoNCE; busca legenda→foto com R@1 de 39,9% contra 0,08% ao acaso; modality gap) · PDF 3.5 |
| Ranking com ≥ 20 descrições + top-5 visualizados | NB A2 §3 (24 conceitos, τ calibrado, top-5 com exemplos) · PDF 3.3 |
| Busca com ≥ 8 consultas de especificidade variada, analisadas | NB A2 §4 + §6.4 (10 consultas, P@5, variação de threshold) · PDF 3.4 |
| Consulta do CLIP × tokenização do BERT (padding e attention mask) | NB A2 §6 (tokens reais do CLIP e do BERT, máscara causal, teste do padding) · PDF 3.6 |

## 5. GANs (A4.1, A4.2)

| Item | Onde |
|---|---|
| ≥ 5 problemas do raio-X com impacto clínico | NB A4 §2 (10 problemas) + baseline reproduzido · PDF 5.2 |
| GAN implementada com loop adversarial correto | NB A4 §5 (`train_cgan`: cGAN DCGAN 128×128, perda não-saturante) · PDF 5.3 |
| Instabilidade diagnosticada + mitigação com evidência | NB A4 §5 "Instabilidade de treino e mitigação" (ablação com 3 variantes: divergência na versão de livro; os estabilizadores equilibram o jogo; minibatch-std contra o mode collapse: KID −24%, FID −17%, diversidade +10%) · PDF 5.3 |
| Impacto das sintéticas no recall de COVID (com × sem) | NB A4 §6 (A/B1/B2/C × 3 sementes, teste ampliado, IC por bootstrap) · PDF 5.4 |
| ≥ 4 problemas do tráfego + riscos do transfer learning do ImageNet | PDF 6 (7 problemas + "Riscos específicos do transfer learning do ImageNet") |
| Plano integrado: modelo, métrica, aumento sintético e critério de adoção clínica | NB A4 §9 (4 fases + tabela go/no-go) · PDF 5.6 |
