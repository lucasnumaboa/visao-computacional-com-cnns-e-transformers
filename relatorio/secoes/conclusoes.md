# 7. Conclusões

| Atividade | Resultado principal | Lição técnica |
|---|---|---|
| A1: ViT, fundição | Atenção e ViT implementados do zero (6 testes passando). Fine-tuning do ViT-B/16: **100% no teste** (erro < 1,5% a 95%). ViT do zero: 78,5%. Swin-T: 99,0%. ResNet-50: 99,5%. | O pré-treino resolve a fome de dados do ViT. Com pré-treino, ViT, Swin e CNN empatam. Sem PE, a atenção é invariante à ordem (cosseno 1,000 com os patches embaralhados). CutMix é inadequado para anomalias locais. A camada 6 localiza defeitos e a 12 usa o fundo como "registrador". |
| A2: CLIP, ADS-16 | Zero-shot de **59,8%** (top-1, 20 categorias). Busca legenda→foto com **R@1 de 39,9%** (acaso: 0,08%). τ = 0,234 calibrado. Top-5: produto em fundo branco, texto/logo, pessoa, animal, roupa. | O pré-treino contrastivo alinha as modalidades por ordenação, não por sobreposição (*modality gap*), e por isso é preciso calibrar o threshold. O CLIP acerta o concreto, falha na composição e lê texto. A máscara causal + pooling no `<eot>` dispensa a `attention_mask` do BERT. |
| A3: ResNet-50 congelada | **99,62%** no teste (1 erro em 265), com 14 mil parâmetros treinados e 0,49 GB de VRAM. | Domínio próximo + poucos dados levam a feature extraction. O teste está saturado, então o valor da augmentation está na robustez: `bike` cai a 38,5% em baixa resolução, e há atalhos de aquisição por classe. |
| A4.1: COVID-19 + cGAN | Recall de COVID-19 de **41,7% para 82,6–92,6%** só corrigindo o pipeline. Sintéticas da cGAN **não** melhoraram o recall (ΔRecall de −2,8 a −8,6 p.p.). Ablação: divergência na GAN de livro, controlada pelos estabilizadores; minibatch-std reduziu o KID em 24%. | O ΔRecall com teste real e IC é o critério de adoção. Sintéticas com FID alto viram atalho. A adoção clínica exige sensibilidade ≥ 95% com IC, desempenho por subgrupo e fase silenciosa. |
| A4.2: tráfego | 7 problemas, com impacto e abordagem para cada um, mais os riscos específicos do transfer learning do ImageNet. | O problema central foi a **validação** (sem divisão por câmera nem teste por condição), não a escolha da ResNet-50. |

**Temas transversais.** Três padrões se repetiram nas quatro atividades e valem como aprendizado da disciplina:

1. **Atalhos de aquisição estão em toda parte**:
   - o fundo na A1;
   - formato e resolução por classe na A3;
   - repositório de origem por classe na A4.1;
   - texto escrito na imagem na A2.

   O dataset "fácil" esconde um modelo que pode estar aprendendo a fonte e não o conteúdo. Por isso cada notebook mede o atalho explicitamente (contrafactual, estresse, subgrupos, metadados de fonte).
2. **A métrica e o protocolo de avaliação pesam mais que a arquitetura**:
   - recall e balanced accuracy por classe;
   - teste ampliado para estimar o recall de uma classe com 18 exemplos;
   - IC por bootstrap;
   - validação saturada que impede escolher limiar;
   - divisão por câmera (A4.2).

   Na A1, três arquiteturas diferentes empataram. O que separou os resultados foi o pré-treino e o protocolo.
3. **Resultados negativos também são resultados.** A GAN da A4.1 e o CutMix da A1 não ajudaram, e a versão de livro da GAN teve melhor KID em 2.500 iterações apesar de divergir. Relatar isso com o mecanismo provável é mais útil para uma decisão de engenharia do que um ganho aparente num teste pequeno.
