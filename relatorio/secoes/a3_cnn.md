# 4. Atividade 3: classificador com CNN pré-treinada (feature extraction)

## 4.1 Definição do problema

Classificar as imagens do *Images Dataset* (Kaggle, `pavansanagapati/images-dataset`) em 7 categorias: `bike`, `cars`, `cats`, `dogs`, `flowers`, `horses` e `human`. O método é *transfer learning* por **feature extraction**: o backbone pré-treinado fica congelado e só uma nova camada de classificação é treinada, em um único treino.

**Inventário e limpeza.** A inspeção dos dados revelou três problemas antes de qualquer treino:

1. O zip contém uma **cópia integral do dataset em `data/data/`**. Usá-la duplicaria as 1.803 imagens e espalharia cópias entre treino e teste. Essa pasta foi ignorada.
2. Há **39 duplicatas exatas** (MD5), 19 em `horses`, 19 em `human` e 1 em `bike`. Foram removidas, e ficaram **1.764 imagens**.
3. **Cada classe vem de uma fonte com assinatura própria**:
   - `bike` e `cars`: BMP 640×480, fotos de rua;
   - `flowers`: PNG 128×128, cores posterizadas;
   - `human`: JPG em retrato (proporção 0,73), quase sempre cavaleiros em traje de equitação;
   - `cats` e `dogs`: JPG.

   Resolução, formato e paleta se correlacionam com a classe, e isso cria **atalhos de aquisição**. O efeito aparece no teste de estresse (seção 4.4).

![Exemplos de cada classe após a limpeza.|78%](../figs/A3/02_exemplos_por_classe.png)

## 4.2 Decisões técnicas e justificativa

**Escolha do modelo pré-treinado, considerando o T4 e o número de classes:**

| Modelo | Parâmetros | GFLOPs | Top-1 ImageNet | Features | Observação |
|---|---|---|---|---|---|
| ResNet-18 | 11,7 M | 1,8 | 69,8% | 512 | features mais fracas |
| EfficientNet-B0 | 5,3 M | 0,4 | 77,7% | 1280 | a mais leve; boa para *edge* |
| **ResNet-50 (V2)** | **25,6 M** | **4,1** | **80,9%** | **2048** | **escolhida** |
| ViT-B/16 | 86,6 M | 17,6 | 81,1% | 768 | 4× mais cara, sem ganho de top-1 |

- **Capacidade do T4 (16 GB):** em feature extraction só há inferência no backbone. A VRAM medida durante a extração ficou bem abaixo de 1 GB. Mesmo um fine-tuning completo da ResNet-50 (~8,5 GB com batch 64, Aula 01) caberia no T4. Não há motivo de memória para um modelo menor.
- **Número de classes:** são só 7 classes. A cabeça `Linear(2048, 7)` tem 14.343 parâmetros para ~176 imagens de treino por classe. O gargalo de um classificador linear é a **separabilidade** das features, então vale usar o backbone com as features mais fortes dentro de um custo moderado.
- **Domínio:** as 7 categorias são objetos que o ImageNet cobre (raças de cães e gatos, *mountain bike*, *sports car*, *daisy*, *sorrel*). A ResNet-50 tem o mesmo top-1 do ViT-B/16 com ¼ do custo. A EfficientNet-B0 seria a escolha se a restrição fosse latência ou memória.

| Decisão | Escolha | Justificativa |
|---|---|---|
| Backbone | **ResNet-50**, `ResNet50_Weights.IMAGENET1K_V2` | Ver a tabela acima. É também a CNN de referência das Aulas 01/08. |
| Pré-processamento | `weights.transforms()`: resize 232, crop 224, normalização ImageNet | É a distribuição exata do pré-treino. Pré-processamento errado gera "bug silencioso" (Aula 01). |
| Congelamento | `requires_grad=False` em todo o backbone e backbone em `eval()` | Sem `eval()`, as estatísticas do BatchNorm seriam atualizadas, mesmo com os pesos congelados. |
| Nova cabeça | `nn.Linear(2048, 7)`: 14.343 parâmetros | Substitui a `fc` de 1000 classes. |
| Treino | Adam, lr = 1e-3, batch 32, 15 épocas, cross-entropy; checkpoint de menor loss de validação | Receita da Aula 01. A cabeça parte do zero e o problema é convexo nela. |
| Eficiência | Features extraídas **uma vez** e guardadas em cache (Aula 03) | Com backbone congelado e sem augmentation, a saída do backbone é determinística, e treinar sobre o cache é matematicamente idêntico. Verificado: a ResNet-50 completa, montada com a cabeça treinada, concorda em 100% com as predições feitas a partir do cache. |
| Divisão | Estratificada 70/15/15 (1.234 / 265 / 265) | Preserva a proporção das classes (`cars` tem 2,3× mais imagens que `horses`). |

## 4.3 Resultados (item 3.1)

<span class="kpi">Accuracy global (teste): <b>99,62%</b></span> <span class="kpi">Balanced accuracy: <b>99,74%</b></span> <span class="kpi">Erros no teste: <b>1 de 265</b></span>

| Classe | n teste | Acertos | Accuracy |
|---|---|---|---|
| bike | 54 | 53 | 98,15% |
| cars | 63 | 63 | 100% |
| cats | 30 | 30 | 100% |
| dogs | 31 | 31 | 100% |
| flowers | 32 | 32 | 100% |
| horses | 28 | 28 | 100% |
| human | 27 | 27 | 100% |

![Curvas de loss e accuracy por época (treino e validação). A linha tracejada marca a época escolhida.|85%](../figs/A3/03_curvas_loss_acc.png)

As curvas mostram convergência rápida, sem *overfitting*. A accuracy de validação chega a 99,2% na época 3 e se estabiliza. A loss de validação cai monotonamente até a época 15, que é a escolhida.

![Matriz de confusão e accuracy por classe no teste.|88%](../figs/A3/04_confusao_e_acc_por_classe.png)

**O único erro de teste** é uma foto de rua rotulada `bike` em que a bicicleta ocupa uma pequena área à esquerda, numa cena dominada por contêineres de lixo e asfalto. O modelo prevê `cars` com confiança de apenas 0,48. Em treino, validação e teste somados há só 3 erros: `bike→cars`, `cars→bike` e `cats→dogs`. As classes `bike` e `cars` são **fotos de cena** com contexto urbano compartilhado, enquanto as outras cinco são **centradas no objeto**. A confusão vem do contexto.

![t-SNE das features da ResNet-50 congelada: sete agrupamentos quase sem sobreposição.|55%](../figs/A3/06_tsne_features.png)

## 4.4 Análise crítica e propostas de melhoria (item 3.2)

**O teste está saturado.** Com 1 erro em 265, cada imagem vale 0,4 ponto percentual, e nenhuma melhoria de augmentation seria mensurável neste conjunto de teste. O t-SNE explica por quê: as features genéricas do ImageNet já separam as 7 classes quase perfeitamente. O valor de qualquer augmentation está na **robustez a mudança de distribuição** (outras câmeras, resoluções e enquadramentos). Para embasar a discussão com dados e não só com intuição, sem fazer um segundo treino, rodamos dois experimentos:

- um **teste de estresse** (Aula 08): o modelo treinado foi avaliado nas 530 imagens de validação e teste, transformadas;
- uma **sondagem da normalização** com classificador por centroide.

![Teste de estresse: accuracy por classe sob cada transformação (sem retreino).|85%](../figs/A3/07_teste_estresse.png)

**1. Augmentation geométrica.**
- *Flip horizontal:* **benéfico e seguro para todas as classes**, porque nenhuma tem semântica esquerda/direita. O modelo já é invariante (99,6% com flip), então o ganho é pequeno e o risco é nulo.
- *Rotação leve (±10–15°):* **benéfica**. Algumas fotos de `bike` foram tiradas com a câmera girada. A queda sob 15° é mínima (98,2% em `horses` e `human`). **Rotações grandes são irreais** para animais e pessoas, que estão sempre em pé: sob 90°, `dogs` cai para 96,7%. Não as usaria.
- *Recorte aleatório:* **benéfico com escala moderada, prejudicial se agressivo**. Um recorte de 50% no canto derruba `dogs` para 78,7%, `bike` para 89,9% e `human` para 90,9%. Um recorte central derruba `cats` para 86,9%. Em `bike` e `cars` o objeto é pequeno e fora do centro, então um recorte pode excluí-lo e gerar **ruído de rótulo**. Em `human` (retrato) o recorte corta cabeça ou pés. Em `cats`/`dogs`, perder o focinho apaga o traço que mais diferencia as duas classes. Usaria `RandomResizedCrop(scale=(0.6, 1))`, nunca o padrão (0,08).

**2. Augmentation de cor.**
- *Brilho e contraste:* **benéficos** como robustez. `bike` tem fotos noturnas, e escurecer pela metade mantém 99,8%.
- *Matiz e grayscale:* **dependem da classe**. `flowers` tem a maior saturação média (0,42, contra 0,15–0,36 nas demais), e a cor é parte legítima da classe: grayscale e matiz deslocada introduzem erros em `flowers`, além de quedas em `bike` e `dogs`. Jitter de matiz forte ou grayscale com p alto **prejudicaria `flowers`**. Por outro lado, a paleta posterizada de `flowers`, `horses` e `human` é um artefato de aquisição. Um jitter leve com grayscale a p ≈ 0,1 (como na Aula 08) reduz a dependência desse atalho sem apagar a cor. Para `cats`, `dogs` e `horses` a cor da pelagem não é discriminativa, e o jitter é seguro.

**3. Variação de escala.** **É a melhoria mais importante para este dataset.**
- Com as imagens reduzidas a 48 px e reampliadas, `bike` despenca para **38,5%**, `dogs` vai a 83,6% e `cars` a 93,7%, enquanto `flowers`, `horses` e `human` ficam em 100%.
- Bicicletas são estruturas finas que somem em baixa resolução.
- As três classes imunes são exatamente as de **origem em baixa resolução**: `flowers` (128×128), `human` (~188×257) e `horses` (~259×194).
- As bicicletas reduzidas a 48 px vão, em sua maioria, para **`human`** (27 casos), a classe de menor resolução nativa, e depois para `cats` (18) e `cars` (15).
- Para o modelo, "imagem de baixa resolução" é em parte uma pista de classe: é o atalho de aquisição descrito na seção 4.1.
- Treinar com **multi-escala** (`RandomResizedCrop` + redução e ampliação aleatórias) desacopla resolução de classe.
- Riscos: limitar a redução mínima em `bike`/`cars`, para o objeto não ficar irreconhecível, e usar *letterbox* em `human` para não cortar o corpo.

**4. Normalização ImageNet.** Já é usada no item 3.1 e é **obrigatória por construção**. Com o backbone congelado, filtros e estatísticas de BatchNorm foram calibrados para entradas com essa média e desvio, e nenhum parâmetro pode se adaptar a outra distribuição. **O experimento, porém, mostrou honestamente que neste dataset o efeito é pequeno:**
- as features com e sem normalização têm cosseno médio de 0,92;
- o classificador por centroide dá 98,9% nos dois casos;
- a cabeça treinada mantém 99,6% recebendo features sem normalização.

As classes estão tão separadas que um deslocamento de 8% no espaço de features não muda decisões. Isso **não** generaliza para classes finas ou domínios distantes do ImageNet, onde as fronteiras são apertadas. Por isso a normalização é tratada como requisito de correção, e não como hiperparâmetro.

**Feature extraction ou fine-tuning: quando usar cada um.** A decisão depende do **volume de dados rotulados** e da **distância entre o domínio e o pré-treino** (Aulas 01, 03, 06 e 08):

| | Domínio próximo do ImageNet | Domínio distante (raio-X, metal, satélite) |
|---|---|---|
| **Poucos dados** | **Feature extraction** (este caso): sem *overfitting* nem esquecimento, roda em segundos | **Fine-tuning parcial** (últimos blocos, LR 1e-5 / 1e-3) + augmentation forte, com o *linear probe* como baseline |
| **Muitos dados** | Fine-tuning das camadas finais ou completo, com ganho modesto | **Fine-tuning completo** |

Aqui são 1.234 imagens de treino de categorias que o ImageNet cobre, e o feature extraction já chega a 99,6%. Um fine-tuning atualizaria 23,5 M parâmetros com ~1.200 imagens, com custo maior, risco de sobreajuste e perda de robustez (Aula 06), e sem margem mensurável no teste. Os outros projetos deste trabalho mostram o outro lado da tabela:

- **A1 (metal fundido, domínio distante):** o *linear probe* chega a 0,987 de balanced accuracy, e o fine-tuning chega a 1,000.
- **A4.1 (raio-X):** o fine-tuning de uma ResNet-18 pré-treinada levou o recall de COVID-19 de 41,7% para 82–93%.

**Regra prática:** começar sempre por feature extraction ou *linear probe*. Partir para fine-tuning (primeiro parcial, depois completo) só se a validação mostrar que as features congeladas não bastam.

**Ordem em que eu testaria:**
1. **Criar um teste com mudança de distribuição**, porque o atual está saturado.
2. Multi-escala, voltada ao colapso de `bike` e ao atalho de resolução.
3. Flip e rotação ±10°.
4. Brilho e contraste moderados, com grayscale a p = 0,1 e sem matiz forte.

Com augmentation, o cache de features deixa de valer e o backbone congelado volta a rodar a cada época. Se a robustez ainda for insuficiente, o passo seguinte é o fine-tuning parcial da `layer4` com LR diferencial (1e-5 / 1e-3, Aula 08).
