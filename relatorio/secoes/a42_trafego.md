# 6. Atividade 4.2: transfer learning para análise de tráfego urbano

## 6.1 Definição do problema

Um grupo classificou o fluxo de tráfego de câmeras urbanas em **livre**, **moderado** ou **congestionado**. O modelo foi uma ResNet-50 pré-treinada no ImageNet, com *fine-tuning* em **800 frames rotulados de 12 câmeras**. A accuracy de validação foi de 78% e o sistema foi implantado. Em produção, surgiram **falhas sistemáticas** em três situações: chuva, períodos noturnos e câmeras com ângulos que não estavam no treino.

O raciocínio "o ImageNet reconhece objetos, veículos são objetos, logo o transfer learning resolve" esconde duas suposições falsas. A primeira é que **a distribuição de produção é a mesma do treino**. A segunda é que **a tarefa é de reconhecimento de objetos**. A referência indicada (Meegle, *Transfer Learning for Traffic Analysis*) lista como armadilhas típicas:

- o descasamento entre domínio de origem e de destino (*data mismatch*);
- o *overfitting* no fine-tuning agressivo;
- o custo computacional;
- a baixa interpretabilidade.

Como soluções, aponta:

- alinhamento de domínio;
- regularização;
- modelos híbridos (aprendizado + métodos tradicionais);
- explicabilidade.

A análise a seguir detalha esses pontos para o caso concreto.

**Riscos específicos do transfer learning do ImageNet para classificar fluxo.**

1. **Viés de objeto, não de cena.** O ImageNet treina a rede para responder "qual é o objeto dominante?" (1 rótulo por imagem, objeto centralizado). "Congestionado" é uma propriedade **coletiva e espacial**: muitos veículos pequenos, ocupação da via, filas. As features finais da ResNet-50 são invariantes justamente ao que importa aqui (contagem, densidade, posição), porque o *global average pooling* descarta a localização.
2. **Viés de distribuição visual.** O ImageNet é dominado por fotos diurnas, nítidas, de câmeras amadoras, ao nível do chão. Câmeras de tráfego são altas, com perspectiva forte, baixa resolução, compressão de vídeo, noite com faróis e lentes molhadas. A suposição de que "as features genéricas servem" falha exatamente nas condições adversas.
3. **Ausência de tempo.** O pré-treino é sobre imagens estáticas. Fluxo depende de **velocidade**, e nada no ImageNet ensina dinâmica.
4. **Falsa sensação de suficiência.** Como o *fine-tuning* converge rápido e a accuracy de validação sai alta (78%), o pré-treino mascara a falta de dados e de diversidade. O modelo aprende atalhos das 12 cenas (o poste, a placa, o recorte do céu) que o pré-treino não impede.
5. **Esquecimento catastrófico.** O *fine-tuning* completo com 800 frames pode apagar parte das representações gerais, inclusive a robustez a iluminação que o modelo tinha (Aula 06).

## 6.2 Problemas identificados, impacto operacional e abordagem

### Problema 1: *covariate shift* de condição (chuva, noite, ofuscamento)

**O que acontece.** O conjunto de 800 frames quase certamente reflete as condições mais comuns na coleta: dia e tempo seco. À noite, a cena muda por completo. Os veículos viram pares de faróis e lanternas, o contraste cai e o ruído do sensor sobe. Na chuva, aparecem reflexos no asfalto, gotas na lente, desfoque e névoa. O modelo nunca viu essas distribuições, e o ImageNet também é majoritariamente diurno. É o *covariate shift* da Aula 08: P(x) muda e P(y|x) continua a mesma.

**Impacto operacional.** As falhas se concentram **justamente** quando o tráfego é mais crítico: chuva aumenta congestionamento e acidentes, e o fim de tarde no inverno já é noturno. Um "livre" falso num corredor congestionado na chuva leva a decisões erradas de semaforização, desvio e despacho de agentes. Como o erro é sistemático e não aleatório, os operadores perdem a confiança no sistema como um todo.

**Abordagem.**
1. **Medir antes de corrigir.** Registrar a condição de cada frame (dia/noite, seco/chuva/névoa, a partir do horário e de uma estação meteorológica). Montar **conjuntos de teste por condição** e reportar as métricas separadamente: é o teste de estresse OOD da Aula 08. A accuracy agregada de 78% esconde essa estrutura.
2. **Coleta direcionada** de frames noturnos e chuvosos, até cada condição ter representação mínima no treino.
3. **Augmentation fotométrica** coerente com a física da cena: brilho, gama, contraste, ruído de sensor, desfoque de movimento, *glare* e gotas sintéticas.
4. **Tradução de domínio com CycleGAN** (Aula 07): dia→noite e seco→chuva, treinada sem pares. Cada frame diurno rotulado ganha versões noturna e chuvosa **com o mesmo rótulo**, porque a densidade de veículos não muda com a tradução. A validação desse ganho segue a regra da Aula 08: imagens sintéticas só no treino e teste 100% real, medindo o ganho por condição.
5. **Alinhamento de domínio**, como sugere a referência. Por exemplo, ajuste das estatísticas de BatchNorm por condição, ou adaptação não supervisionada com frames não rotulados de produção, que existem em abundância.

### Problema 2: vazamento no protocolo de validação; os 78% são otimistas

**O que acontece.** Com 800 frames de 12 câmeras, cada câmera contribui com ~67 frames, muitos deles **consecutivos ou quase idênticos** (mesmo fundo, mesma posição dos postes, fluxo parecido). Uma divisão aleatória põe frames quase iguais da mesma câmera no treino e na validação. Assim, a validação mede **memorização do cenário** e não generalização. Ela também não mede o que importa em produção: o desempenho numa câmera nova ou num dia novo.

**Impacto operacional.** A decisão de implantar foi tomada com base num número que não representa o uso real. O sistema entrou em produção sem ninguém saber que ele generaliza mal para outras câmeras, e o problema só apareceu em campo, com custo e desgaste institucional.

**Abordagem.**
- **Divisão por grupo:** `GroupKFold` por câmera, isto é, *leave-camera-out* (Aula 08). Assim cada câmera de validação nunca foi vista no treino.
- **Divisão temporal:** treinar com dias anteriores e testar em dias posteriores.
- Reportar a métrica **por câmera** e **por condição**, com intervalos de confiança.
- Com 12 câmeras, a variância entre as dobras será alta, e isso é informação: indica quanto o desempenho depende do cenário.

### Problema 3: generalização de ponto de vista (geometria da câmera)

**O que acontece.** A aparência de "congestionado" depende da altura, do ângulo, do zoom e da perspectiva da câmera. Uma câmera alta e de grande angular sobre uma via expressa vê dezenas de veículos pequenos. Uma câmera baixa num cruzamento vê poucos veículos grandes, com oclusão. Com só 12 pontos de vista, o modelo aprende "como é o congestionamento **nessas** 12 cenas", não o conceito. Objetos pequenos e distantes são outra dificuldade: num ViT, um veículo distante pode caber num único *patch* 16×16 e perder o contorno (Aula 05). Numa CNN, ele vira poucos pixels após o *downsampling*.

**Impacto operacional.** Cada câmera nova instalada falha até ser retreinada. A operação não escala para centenas de câmeras, e o custo de rotulagem cresce linearmente com a rede.

**Abordagem.**
- **Diversidade de pontos de vista no treino**, com coleta em mais câmeras.
- **Augmentation geométrica:** transformações de perspectiva e variação de escala. Sem flip vertical, que é irreal.
- **Máscara de região de interesse por câmera** (só as faixas de rolamento), para o modelo não usar o cenário fixo como atalho.
- **Normalização da densidade pela área de via visível**, calibrada uma vez por câmera.
- **Adaptação leve para câmeras novas:** *linear probe* ou *fine-tuning* parcial com poucas dezenas de frames rotulados (Aula 06/08), em vez de retreinar tudo.
- **Backbones hierárquicos** (Swin, PVT + FPN, Aula 05) ou resolução de entrada maior, para preservar veículos pequenos.

### Problema 4: descasamento de tarefa; fluxo é temporal e de cena, não um objeto num frame

**O que acontece.** O pré-treino no ImageNet é **centrado em objeto**: uma imagem, um objeto dominante, um rótulo. "Congestionado" é uma propriedade **da cena e do tempo**: densidade **e velocidade**. Um único frame não distingue uma fila parada no semáforo vermelho (fluxo normal, ciclo de 60 s) de um congestionamento real. Também não distingue uma via cheia andando a 60 km/h de uma via cheia parada. O modelo tenta inferir uma grandeza dinâmica a partir de uma foto estática.

**Impacto operacional.** Haverá confusão sistemática entre "moderado" e "congestionado", e alarmes falsos de congestionamento a cada ciclo semafórico. Se o sistema alimenta controle adaptativo de semáforos, isso gera oscilação de planos semafóricos.

**Abordagem.**
- **Modelo híbrido**, como recomenda a referência: **detecção + rastreamento** de veículos (YOLO + DeepSORT, Aula 01). Daí saem a contagem por faixa, a ocupação e a **velocidade média** numa janela de 30 a 60 s. O estado (livre, moderado, congestionado) é classificado a partir dessas grandezas físicas, interpretáveis e independentes do cenário, por regra ou por um classificador leve.
- Alternativa puramente aprendida: **modelos temporais** sobre clipes (fluxo óptico, CNN 3D ou transformers de vídeo), em vez de frames isolados.
- Essa decomposição também resolve parte da **interpretabilidade** apontada pela referência: o operador vê "38 veículos, 7 km/h", não só um rótulo.

### Problema 5: dataset pequeno e rótulo ordinal ambíguo

**O que acontece.** São 800 frames para 3 classes cujas fronteiras são subjetivas: onde termina "moderado" e começa "congestionado"? Sem protocolo, anotadores diferentes (ou o mesmo anotador em dias diferentes) rotulam de forma inconsistente. Provavelmente as classes também são desbalanceadas, porque o congestionamento é raro e concentrado nos horários de pico. A accuracy global trata "livre↔congestionado" como um erro tão grave quanto "moderado↔congestionado".

**Impacto operacional.** O ruído de rótulo limita o teto de desempenho. O recall de "congestionado", que é a classe que importa para a operação, pode ser baixo e ficar escondido pela accuracy (o paradoxo da Aula 08).

**Abordagem.**
- **Protocolo de anotação** com critério objetivo (veículos por faixa por 100 m, velocidade estimada).
- **Dupla anotação** com medida de concordância (κ de Cohen) e adjudicação dos casos discordantes.
- Tratar o problema como **ordinal**: regressão de densidade ou *loss* ordinal, para que errar por uma classe custe menos que errar por duas.
- **Métricas por classe** com foco no recall de "congestionado", matriz de confusão e balanced accuracy.
- **Active learning:** priorizar para rotulagem os frames de produção com baixa confiança.

### Problema 6: fine-tuning completo com pouco dado degrada a robustez

**O que acontece.** O fine-tuning de todos os 25 M parâmetros da ResNet-50 com 800 frames muito correlacionados tende a **sobreajustar** às 12 cenas (armadilha citada na referência) e a apagar parte das representações gerais do pré-treino. A Aula 06 mostra que o *full fine-tuning* ganha pouco no domínio-alvo e perde **robustez fora da distribuição**, e é exatamente essa robustez que falta à noite e na chuva.

**Impacto operacional.** O modelo fica bom nas câmeras e condições de treino e frágil em todo o resto: o padrão de falha observado em produção.

**Abordagem.**
- Comparar, com o protocolo *leave-camera-out*: *linear probe* (backbone congelado), *fine-tuning* parcial (`layer4` + cabeça) com LR diferencial (≈1e-5 no backbone, ≈1e-3 na cabeça; Aula 08) e *fine-tuning* completo.
- Regularização: *weight decay*, *early stopping* pela validação por câmera e augmentation.
- Avaliar backbones com representações mais robustas OOD, como CLIP ou DINOv2 (Aula 05/06). O CLIP permite ainda um baseline **zero-shot** ("a photo of a traffic jam at night") sem nenhum rótulo.

### Problema 7: ausência de monitoramento e de detecção de OOD em produção

**O que acontece.** As falhas "emergiram" em produção, ou seja, foram descobertas por usuários e não pelo sistema. Não havia mecanismo para detectar que a entrada estava fora da distribuição de treino nem para acompanhar o desempenho ao longo do tempo.

**Impacto operacional.** A degradação é silenciosa: decisões erradas por horas ou dias antes de alguém perceber. Também não há dados organizados para corrigir o problema.

**Abordagem.**
- Monitorar a **confiança/entropia** das predições e estatísticas da entrada (brilho médio, nitidez, embeddings).
- Ter um **detector de OOD** que marque frames atípicos e acione uma alternativa (regra baseada em detecção/contagem ou revisão humana) em vez de uma predição confiante e errada.
- Alertas de *drift*.
- Um ciclo de **retreino periódico** alimentado pelos casos difíceis coletados em produção.

## 6.3 Síntese

| # | Problema | Impacto operacional | Abordagem principal |
|---|---|---|---|
| 1 | *Covariate shift* (chuva, noite) | Falhas sistemáticas nos momentos mais críticos | Teste por condição, coleta direcionada, augmentation fotométrica, CycleGAN dia→noite/seco→chuva |
| 2 | Vazamento na validação (frames da mesma câmera) | Decisão de implantar baseada em métrica otimista | `GroupKFold` por câmera, divisão temporal, métricas por câmera |
| 3 | Pontos de vista novos | Cada câmera nova falha; não escala | Mais câmeras, augmentation de perspectiva, ROI, adaptação leve por câmera |
| 4 | Tarefa temporal e de cena tratada como objeto num frame | Confusão moderado/congestionado, alarmes a cada semáforo | Detecção + rastreamento (YOLO + DeepSORT) → contagem e velocidade; modelos temporais |
| 5 | Poucos dados, rótulo ordinal ambíguo | Teto de desempenho, recall de "congestionado" escondido | Protocolo de anotação, κ, *loss* ordinal, métricas por classe, *active learning* |
| 6 | Fine-tuning completo com pouco dado | Perda de robustez OOD | *Linear probe* / FT parcial com LR diferencial, regularização, CLIP/DINOv2 |
| 7 | Sem monitoramento nem detecção de OOD | Degradação silenciosa | Monitoramento de confiança e *drift*, detector de OOD com alternativa, retreino contínuo |

A conclusão é que o problema central não era a escolha da ResNet-50, e sim a **validação**. Com uma divisão por câmera e testes por condição, as falhas de produção teriam aparecido **antes** da implantação. A primeira ação recomendada é, portanto, refazer a avaliação com esse protocolo. Só depois vem a escolha entre as correções de dados, de modelo e de formulação da tarefa.
