# 5. Atividade 4.1: estudo de caso de COVID-19 em radiografias de tórax

## 5.1 Definição do problema

Um grupo entregou um sistema de triagem Normal / Pneumonia / COVID-19 com resultados "promissores", mas o time clínico relatou que o modelo **"ignora casos positivos"**. A configuração era:

- 1.200 radiografias: Normal 840, Pneumonia 240, COVID-19 120;
- divisão 80/20 sem estratificação;
- ResNet-18 **sem pré-treino**, SGD com LR fixo de 0,01, 15 épocas, sem augmentation;
- avaliação **só por accuracy global**;
- resultado: 93% no treino e 61% na validação.

A tarefa tem quatro partes: diagnosticar ao menos cinco problemas, implementar uma abordagem generativa para a escassez de COVID-19, medir o impacto no recall dessa classe e propor um plano integrado.

**Dados.** Usamos a *COVID-19 Radiography Database* (Kaggle) e sorteamos com semente fixa **exatamente o cenário do enunciado**: Normal 840, Pneumonia viral 240 e COVID-19 120. Também separamos um **teste ampliado** com 1.500 radiografias reais fora dessas 1.200 (600 Normal, 300 Pneumonia, 600 COVID-19). Com só 18 COVID-19 no teste do cenário, cada caso vale 5,6 p.p. de recall, e o teste ampliado dá estimativas estáveis. As imagens são convertidas para tons de cinza em 128×128, a resolução que a GAN gera, para que reais e sintéticas passem pelo mesmo pipeline.

> **Nota bibliográfica.** O enunciado atribui o artigo de referência a "Nour & Tariq (2023)". O DOI indicado (10.1038/s41598-023-37743-4) corresponde, segundo o Crossref e o Europe PMC, a *Dumakude & Ezugwu, "Automated COVID-19 detection with convolutional neural networks", Scientific Reports 13, 10607 (2023)*. Citamos o artigo pelo DOI fornecido. A afirmação de que o recall da classe COVID-19 fica abaixo de 60% em dados desbalanceados é atribuída ao enunciado.

**Um risco de *shortcut* que já está nos dados.** Os metadados mostram que **cada classe vem de repositórios diferentes**:

| Fonte | Normal | Pneumonia | COVID-19 |
|---|---|---|---|
| kaggle.com/c (RSNA) | 732 | 0 | 0 |
| kaggle.com/paultimothymooney | 108 | 240 | 0 |
| bimcv.cipf.es | 0 | 0 | 72 |
| github (armiro, ieee8023, ml-workgroup) | 0 | 0 | 32 |
| eurorad.org / sirm.org | 0 | 0 | 16 |

Um modelo pode aprender a reconhecer **a fonte** (equipamento, marcações, contraste) em vez da doença.

## 5.2 Diagnóstico: problemas técnicos e impacto clínico

| # | Problema | Evidência | Impacto clínico esperado | Correção |
|---|---|---|---|---|
| 1 | ResNet-18 **sem pré-treino** | 11,7 M parâmetros aprendidos do zero com ~960 imagens | Não aprende features gerais, decora o treino e faz diagnósticos instáveis | *Transfer learning* do ImageNet (Aula 01) |
| 2 | **Overfitting** | 93% vs 61%: gap de 32 p.p. | O "promissor" é ilusório; em produção fica perto dos 61% | Pré-treino, augmentation, regularização, *early stopping* |
| 3 | **Sem augmentation** | Cada imagem é vista sempre igual | Frágil a posicionamento, exposição e outro equipamento | Augmentation **clinicamente válida**, **sem flip horizontal** (coração à esquerda, Aula 08) |
| 4 | **Desbalanceamento não tratado** (7:1) | A *loss* é dominada por Normal | **Falsos negativos de COVID-19**: pacientes infectados liberados sem isolamento | Sintéticas, pesos por classe, *oversampling*, limiar calibrado |
| 5 | **Split sem estratificação** | A validação recebe por sorteio de ~15 a ~30 COVID-19 | A métrica de COVID-19 vira loteria | `stratify=y` (Aula 08) e várias sementes |
| 6 | **Accuracy como única métrica** | "61%" sem recall por classe | **Paradoxo da acurácia**: o "ignora positivos" fica invisível | Recall/sensibilidade por classe como métrica primária, especificidade, F1, balanced accuracy |
| 7 | **SGD com LR fixo e 15 épocas arbitrárias** | Sem *scheduler* nem critério de parada | O modelo final é o da época 15, não o melhor; não reprodutível | *Scheduler* + seleção pela validação |
| 8 | **Sem teste independente** | A validação ajusta **e** reporta | Número reportado otimista | Treino/validação/teste com o teste usado uma vez |
| 9 | **Vazamento por paciente e viés de fonte** | Classes vêm de repositórios distintos; não há ID de paciente | O modelo pode aprender o hospital, não a doença, e falhar em silêncio em outro hospital | `GroupKFold` por paciente, validação multicêntrica, auditoria com mapas de atenção |
| 10 | **Limiar não calibrado** | `argmax` padrão | Trata FN e FP como igualmente graves, o que em triagem é inadequado | Limiar escolhido para uma sensibilidade-alvo |

### Reprodução do baseline: a queixa clínica confirmada

Reproduzimos a configuração original: `resnet18(weights=None)`, SGD com 0,01, 15 épocas, sem augmentation e 80/20 sem estratificação.

<span class="kpi">Accuracy treino <b>99,8%</b></span> <span class="kpi">Accuracy validação <b>89,6%</b></span> <span class="kpi">Recall COVID-19 <b>41,7%</b></span>

A accuracy de validação "promissora" esconde que **14 de cada 24 pacientes com COVID-19 seriam liberados como "Normal"**. Normal tem recall de 99,4% e Pneumonia de 78,3%. É exatamente o comportamento descrito pelo time clínico.

![Baseline falho: a accuracy global (89,6%) esconde recall de 41,7% para COVID-19.|70%](../figs/A4/02_baseline_falho.png)

## 5.3 Abordagem generativa: GAN condicional (DCGAN), 128×128

**Escolha: cGAN, e não CycleGAN.** Há só **84 COVID-19 no treino**. Uma GAN condicional (Mirza & Osindero, 2014; Aula 07) treina com as **840** imagens das três classes: a anatomia comum é aprendida com todas, e o rótulo direciona a geração. A CycleGAN (Normal→COVID-19) seria a alternativa. Ela tem dois geradores e dois discriminadores e é mais pesada para o T4, e fica no plano de melhoria.

**Arquitetura (Aula 07):**

- **G(z, y):** z ∈ ℝ¹⁰⁰ concatenado ao *embedding* de classe (16-d), seguido de `ConvTranspose2d` + BatchNorm + ReLU de 4×4 até 128×128 e saída `Tanh` (3,75 M parâmetros).
- **D(x, y):** a imagem recebe um mapa espacial aprendido da classe como 2º canal, seguido de `Conv2d` stride 2 + LeakyReLU(0,2) com **Spectral Normalization** (2,84 M parâmetros).

**Estabilização (Aula 07):**

- perda **não-saturante**;
- **TTUR** (η_D = 4e-4, η_G = 1e-4, Adam β = (0,5; 0,999));
- *label smoothing* **unilateral** (reais = 0,9);
- ***instance noise*** com *annealing*;
- **amostragem balanceada por classe**, para que COVID-19 não apareça em só 10% dos *batches*.

O treino teve 6.000 iterações com batch 64, **só com o conjunto de treino**.

![Evolução da cGAN com o mesmo z: 8 Normal, 8 Pneumonia, 8 COVID-19 por linha.|92%](../figs/A4/04_gan_evolucao.png)

![Linha 1: COVID-19 reais. Linha 2: COVID-19 sintéticas. Linha 3: Normal sintéticas.|95%](../figs/A4/05_reais_vs_sinteticas.png)

**Qualidade das sintéticas (Aula 08):**

| Comparação | N₁ | N₂ | FID | KID ×10³ |
|---|---|---|---|---|
| sintéticas COVID-19 × reais COVID-19 (treino) | 504 | 84 | **385,8** | **463,7** |
| reais COVID-19 (teste ampliado) × reais COVID-19 (treino) | 600 | 84 | 82,9 | −1,6 (≈ 0) |
| reais Normal × reais COVID-19 (treino) | 588 | 84 | 108,1 | 26,1 |

O FID com N = 84 é inflado: o FID real×real de 82,9 vem só do N pequeno, e o KID não enviesado dá ≈ 0. Mesmo assim, **as sintéticas estão mais longe das COVID-19 reais do que as radiografias Normais estão**. A anatomia é plausível, mas há uma **textura granulada** que não existe nas reais. Outros dois indicadores:

- **Diversidade:** o MSE médio entre pares é 0,043 nas sintéticas e 0,074 nas reais. É *mode collapse* parcial: 42% menos variedade.
- **Memorização:** as sintéticas **não** copiam o treino; as vizinhas mais próximas são visivelmente outras imagens.

### Instabilidades de treino: diagnóstico e mitigação com evidência

Para testar se as receitas de estabilização fazem diferença, treinamos do zero **três variantes** com o mesmo gerador, os mesmos dados, a mesma semente e 2.500 iterações cada:

1. **DCGAN de livro:** BatchNorm no D, LR igual de 2e-4, rótulo 1,0, sem ruído.
2. **cGAN estabilizada:** SN, TTUR, *label smoothing* e *instance noise*.
3. **Estabilizada + minibatch-std** (Karras et al., 2018): o D recebe um canal com o desvio-padrão das features **ao longo do batch**, o que permite "ver" e penalizar amostras parecidas entre si.

![Perdas, saída do discriminador e diversidade/KID das três variantes.|100%](../figs/A4/06b_ablacao_estabilizacao_curvas.png)

![Amostras das três variantes na iteração 2.500 (mesmo z).|100%](../figs/A4/06c_ablacao_estabilizacao_amostras.png)

| Variante (2.500 it.) | Diversidade (reais = 0,074) | KID ×10³ ↓ | FID ↓ | D(real) / D(G(z)) finais | loss_G final |
|---|---|---|---|---|---|
| DCGAN de livro | 0,084* | 339 | 316 | 0,86 / 0,10 | 3,78 (subindo) |
| cGAN estabilizada | 0,043 | 523 | 427 | 0,62 / 0,33 | 1,83 (estável) |
| **estabilizada + minibatch-std** | **0,047** | **397** | **353** | 0,63 / 0,30 | 1,69 (estável) |

\* A diversidade em pixels da DCGAN de livro é inflada pelo **ruído granular** visível nas amostras.

**Diagnóstico 1: divergência na DCGAN de livro.** O discriminador **domina**:

- `loss_D` cai continuamente até 0,14;
- D(real) sobe a ~0,86 e D(G(z)) cai a ~0,05–0,10;
- `loss_G` sobe a ~4 e oscila.

O gerador passa a receber gradiente ruidoso e o jogo se afasta do equilíbrio (D ≈ 0,5).

**Mitigação 1 (SN + TTUR + smoothing + ruído).** O jogo fica **equilibrado e estável**: `loss_D` em ~1,0–1,1, perto do −log 4 do equilíbrio, D(real) ≈ 0,6 e D(G(z)) ≈ 0,3, sem tendência. Essa é a evidência de melhoria na dinâmica de treino. O custo é aprender mais devagar (η_G = 1e-4), com KID pior na mesma iteração. Surge também um ***mode collapse* parcial**: a diversidade cai para 58% da das reais, e as amostras ficam "médias".

**Diagnóstico 2 e mitigação 2.** Adicionar **minibatch-std** ao D estabilizado eleva a diversidade em **10%**, reduz o **KID em 24%** e o **FID em 17%**, e mantém o equilíbrio. A mitigação ataca exatamente o problema diagnosticado.

**Ressalvas.** Em 2.500 iterações, a versão de livro ainda tem o melhor KID: a divergência aparece nas curvas, mas ainda não degradou as amostras. O gerador usado no experimento de recall (6.000 iterações) não tem minibatch-std. Retreiná-lo com essa camada e medir o ΔRecall é o próximo passo.

## 5.4 Experimento: recall de COVID-19 com e sem imagens sintéticas

O split é **estratificado** 70/15/15 (COVID-19: 84 / 18 / 18). O classificador corrigido é idêntico em todas as condições:

- ResNet-18 **pré-treinada no ImageNet**;
- augmentation sem flip (rotação de ±10°, recorte leve, brilho/contraste);
- AdamW com LR diferencial e *one-cycle*, 12 épocas;
- seleção pelo macro-F1 de validação;
- **3 sementes por condição**.

**Sintéticas só no treino; validação e testes 100% reais** (Aula 08).

| Condição | Recall COVID (teste, n=18) | Recall COVID (ampliado, n=600) | ΔRecall (ampliado) | IC 95% bootstrap | Macro-F1 (ampliado) |
|---|---|---|---|---|---|
| **A**: só reais | 0,926 ± 0,032 | **0,826 ± 0,008** | — | — | 0,917 |
| **B1**: + 252 sintéticas | 0,852 ± 0,140 | 0,740 ± 0,096 | **−0,086** | [−0,110; −0,057] | 0,888 |
| **B2**: + 504 sintéticas (balanceado) | 0,963 ± 0,032 | 0,798 ± 0,063 | −0,028 | [−0,033; +0,017] | 0,903 |
| **C**: *class weights* (sem GAN) | 0,833 ± 0,056 | 0,797 ± 0,026 | −0,029 | [−0,060; −0,023] | 0,911 |

![Recall de COVID-19 por condição (barras: média; pontos: sementes).|85%](../figs/A4/07_recall_covid_condicoes.png)

![Matrizes de confusão no teste ampliado (ensemble das 3 sementes): os erros de COVID-19 vão sempre para "Normal".|100%](../figs/A4/08_confusao_condicoes.png)

![Curvas por época, base (A) vs aumentado com GAN (B2), média de 3 sementes.|95%](../figs/A4/09_curvas_A_vs_B2.png)

## 5.5 Análise crítica

1. **Corrigir o pipeline resolveu a maior parte do problema.** O recall de COVID-19 foi de **41,7%** no baseline para **92,6%** no teste do cenário e **82,6%** no teste ampliado, com precisão e especificidade de ~0,99, só corrigindo os problemas 1 a 8: pré-treino, augmentation válida, estratificação, otimização e métrica. A diferença entre os dois testes (92,6% vs 82,6%) mostra como um teste com 18 casos é otimista.

2. **As imagens da cGAN não melhoraram o recall, e na dose menor o pioraram.**
   - B2 parece ganhar 3,7 p.p. no teste pequeno, menos de uma imagem em média. No teste ampliado o efeito é −2,8 p.p., com IC contendo zero.
   - B1 piora significativamente (−8,6 p.p.).
   - As sintéticas também **aumentam muito a variância** entre sementes (±9,6 p.p. contra ±0,8 p.p.).

3. **O mecanismo é coerente com as métricas generativas.** Só a classe COVID-19 recebe sintéticas, e elas têm uma textura que nenhuma imagem real tem (FID 386). O classificador aprende **"textura de GAN ⇒ COVID-19"**, um atalho ausente no teste real. O conceito de COVID-19 do modelo se desloca para o domínio sintético: a **precisão sobe** (0,987 → 0,994) e o **recall cai**. É o alerta da Aula 07: realismo estatístico insuficiente não ajuda, e **a validação pelo recall *downstream* é obrigatória**. Sem ela, teríamos relatado o ganho aparente de B2 no teste pequeno.

4. **Os erros residuais são os perigosos.** Em todas as condições, COVID-19 errado vira **"Normal"** (16% em A), nunca Pneumonia. Esse é o caso que o limiar calibrado (problema 10) deve atacar.

5. **Limitações:**
   - viés de fonte não removível nesta base, então o desempenho em outro hospital é desconhecido;
   - sem ID de paciente;
   - uma única configuração de GAN;
   - 3 sementes.

## 5.6 Plano de melhoria integrado

| Fase | Ação | Problemas |
|---|---|---|
| **1. Dados e protocolo** | Base com ID de paciente, hospital e equipamento. COVID-19 e controles **das mesmas instituições**. RT-PCR como padrão-ouro. | 4, 9 |
| | `StratifiedGroupKFold` (classe × paciente) e **teste externo multicêntrico** usado uma vez | 5, 8, 9 |
| | **Sensibilidade de COVID-19 como métrica primária**, especificidade, F1, balanced accuracy, IC. **Limiar calibrado** para uma sensibilidade-alvo combinada com o time clínico (ex.: ≥ 95%). | 6, 10 |
| **2. Modelo** | *Transfer learning* (ImageNet ou backbones de radiografia de tórax) com LR diferencial | 1, 2 |
| | Augmentation clinicamente válida (rotação/translação/recorte leves, brilho/gama), **sem flip** | 3 |
| | AdamW + *cosine*/*one-cycle*, *early stopping* pela métrica clínica, várias sementes | 7 |
| **3. Escassez de COVID-19** | Primeiro o barato e comprovado: pré-treino + augmentation, depois pesos/*oversampling*, sempre por ΔRecall | 4 |
| | GAN **só com critério de aceite**: StyleGAN2-ADA/DiffAugment ou **CycleGAN Normal→COVID-19** (preserva a textura real; λ_cyc = 10, λ_idt = 5); resolução ≥ 256; FID/KID próximos de real×real; revisão por radiologista; filtro por confiança de um classificador treinado só com reais; usar ΔRecall com IC como critério de adoção | 4 |
| **4. Confiabilidade** | Auditoria de atalhos (Grad-CAM/atenção nos FN e FP; mascarar bordas e marcações) | 9 |
| | Implantação como **triagem assistida**: revisão humana dos negativos de baixa confiança, monitoramento de *drift* por hospital e retreino periódico | 2, 9 |

### Critério de adoção clínica (go/no-go)

O modelo só entra em uso assistido se cumprir **todos** os critérios abaixo no **teste externo multicêntrico**. Os limites são definidos com o time clínico **antes** de ver os resultados:

| Critério | Limite proposto | Por quê |
|---|---|---|
| Sensibilidade de COVID-19 no limiar operacional | ≥ 95%, com **limite inferior do IC 95% ≥ 90%** | O falso negativo libera um paciente infectado. O IC impede aprovar por sorte num teste pequeno, como o de 18 casos deste projeto. |
| Especificidade de COVID-19 | ≥ 85% | Mantém os falsos alarmes num volume que o serviço comporta. |
| Desempenho por subgrupo (hospital, equipamento, idade, sexo) | nenhum subgrupo com sensibilidade < 90% | Evita um modelo que funciona "na média" mas falha num hospital, que é o risco do viés de fonte. |
| Calibração | ECE ≤ 0,05 | A probabilidade precisa ser confiável para o limiar e para a revisão humana. |
| Não inferioridade | sensibilidade ≥ à do fluxo atual no mesmo conjunto | O sistema precisa agregar valor ao processo atual. |
| Auditoria de atalhos | atenção/Grad-CAM nos FN e FP sem foco em marcações, bordas ou texto | A decisão precisa vir do pulmão, não da fonte da imagem. |
| Dados sintéticos | só com **ΔRecall > 0 e IC excluindo zero** no teste real | Neste projeto, as sintéticas da cGAN **reprovam** nesse critério. |
| Fase silenciosa | 4–8 semanas em paralelo, sem afetar condutas | Confirma o desempenho com dados de produção. |

Se qualquer critério falhar, o projeto volta para as fases 1 e 2. Depois da adoção, os mesmos limites servem de gatilho para retreino ou desligamento.
