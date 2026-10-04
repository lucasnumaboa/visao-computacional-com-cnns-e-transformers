# 3. Atividade 2: reconhecimento semântico em publicidade visual com CLIP (ADS-16)

## 3.1 Definição do problema

Extrair inteligência semântica do corpus de imagens do **ADS-16** (Roffo & Vinciarelli, 2016) **sem treinar nenhum modelo**: só embeddings pré-treinados do CLIP e consultas em linguagem natural. São duas tarefas:

- **2.1:** ranking dos objetos e conceitos mais presentes no corpus, com um threshold justificado.
- **2.2:** busca de imagens a partir de consultas em texto.

**O corpus real difere do enunciado, e isso foi tratado explicitamente:**

1. O ADS-16 publicado tem **300 anúncios em 20 categorias** (15 por categoria), não 16. Os nomes das categorias foram extraídos da Tabela 1 do artigo original, que acompanha o dataset.
2. Como só há 300 anúncios, nenhum subconjunto só de anúncios chega a 500 imagens. O corpus do ADS-16 também inclui **~1.200 fotos que os 120 participantes enviaram** como imagens de que gostam (positivas) ou que os repulsam (negativas). Cada foto tem uma **legenda escrita pelo próprio participante**.
3. Cada foto aparece **duas vezes** no pacote: o original e uma miniatura `_th_`, com bytes diferentes que o MD5 não detecta. As miniaturas foram descartadas.

**Corpus final: 1.501 imagens**, sendo 301 anúncios, 611 fotos positivas e 589 negativas. Todas as análises são reportadas também **por origem**.

## 3.2 Decisões técnicas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Modelo | **CLIP ViT-B/32**, pacote `clip` da OpenAI | Modelo das Aulas 06/08. Espaço conjunto de 512 dimensões, leve para o T4. |
| Embeddings | Normalizados em L2 e guardados em cache | Com a norma L2, o produto escalar é a *cosine similarity* (Aula 06). |
| Vocabulário | **24 conceitos**: pessoas, produtos, veículos, cenários, animais, dinheiro, elementos de anúncio | Cobre o que aparece em publicidade e em fotos pessoais. |
| Prompts | **Ensemble de 4 templates** por conceito ("a photo of {}", "an image showing {}", "a picture containing {}", "an advertisement featuring {}") | Na Aula 06, o ensemble supera o template único e a palavra solta. |
| Threshold τ | **Percentil 95 de pares negativos aleatórios** (Aula 08) | Detalhado abaixo. |
| Busca | `torch.topk` sobre `IMG @ query.T` | Receita do notebook da Aula 06. |

**Sanidade do espaço de embeddings.** Antes de confiar no CLIP, classificamos os 300 anúncios em 20 categorias **zero-shot**. O resultado foi **59,8% no top-1 e 76,7% no top-3**, contra 5% ao acaso. O heatmap conceito × categoria mostra associações coerentes:

- Automotive ↔ *car*
- Pet Supplies ↔ *animal*
- Jewelery ↔ *jewelry/watch*
- Health & Beauty ↔ *cosmetics*
- Grocery ↔ *food/drink/bottle*
- Baby ↔ *child*
- Social Dating ↔ *person/woman/man*

![Heatmap conceito × categoria de anúncio (z-score por conceito).|92%](../figs/A2/06_heatmap_conceito_categoria.png)

### Calibração do threshold

A *cosine similarity* do CLIP **não é probabilidade** e vive numa faixa estreita (Aula 08: ruído de 0,12 a 0,22, forte a partir de 0,28). Usamos as **legendas dos participantes**, que são ~1.200 pares imagem–texto reais escritos por humanos:

- **positivos:** foto × a própria legenda;
- **negativos aleatórios:** foto × legenda de outra foto, num total de 1,4 milhão de pares.

As legendas recebem os mesmos 4 templates dos conceitos, para que as escalas sejam comparáveis. **τ = P95 dos negativos = 0,234**: aceita 87% dos pares verdadeiros (mediana 0,281) e só 5% dos falsos (mediana dos negativos 0,191).

**Uma calibração descartada, e a lição.** A primeira tentativa comparou os anúncios com as 19 categorias erradas ("an advertisement for {categoria}"). Ela gerou τ = 0,274, com o qual praticamente nenhum conceito passava. A causa é que todo prompt "an advertisement for…" compartilha com qualquer anúncio o componente "isto é um anúncio", o que infla também os negativos (mediana de 0,238). **Calibrar com pares que não representam a tarefa produz um threshold inútil.**

![Esquerda: calibração adotada. Centro: calibração descartada. Direita: distribuição de todas as similaridades imagem × conceito.|100%](../figs/A2/03_calibracao_threshold.png)

## 3.3 Resultados do item 2.1: ranking

| # | Conceito | Frequência (cos ≥ 0,234) | n imagens | Score médio nas presentes |
|---|---|---|---|---|
| 1 | a product on a plain white background | 20,6% | 309 | 0,254 |
| 2 | text and a logo | 19,3% | 290 | 0,247 |
| 3 | a person | 14,0% | 210 | 0,244 |
| 4 | an animal | 13,6% | 204 | 0,252 |
| 5 | clothing | 10,3% | 155 | 0,245 |
| 6 | food | 10,2% | 153 | 0,252 |
| 7 | a man | 8,0% | 120 | 0,244 |
| 8 | a woman | 7,7% | 115 | 0,245 |
| 9 | outdoor nature scenery | 7,7% | 115 | 0,253 |
| 10 | a child | 7,5% | 112 | 0,251 |

O ranking é **estável em torno de τ**: Spearman ρ = 0,96 com τ = 0,22 e ρ = 0,89 com τ = 0,25. Ele desmorona com τ = 0,28 (ρ = 0,29), porque quase nada passa e a ordem vira ruído de contagem.

![Frequência de cada conceito por origem da imagem.|72%](../figs/A2/04_ranking_por_origem.png)

**A separação por origem é o que dá sentido ao ranking:**

- **anúncios:** produto em fundo branco (66%) e texto/logo (50%), depois garrafa, computador, dinheiro/cartão, carro e joias. **O anúncio mostra o produto e a marca**, e pessoas aparecem em menos de 5%.
- **fotos de que os participantes gostam:** natureza, animais de estimação, memes (texto) e praia.
- **fotos que os participantes rejeitam:** pessoas, animais (insetos, roedores), comida.

O vocabulário visual dos anúncios é quase o oposto do vocabulário que esses consumidores dizem preferir. Esse descompasso entre anúncio e preferência é justamente o que o ADS-16 foi criado para estudar.

![Os 5 conceitos mais frequentes: 4 imagens de maior score e 4 sorteadas acima de τ.|100%](../figs/A2/05_top5_conceitos_exemplos.png)

**Inspeção qualitativa do top-5:**

- **an animal:** precisão alta. Porco, guaxinim, percevejo, papagaio, gatinho e peixe estão todos corretos.
- **text and a logo:** boa (Fiat, Burger King, Movie Edit Pro). Há um erro revelador: uma **imagem toda preta** obteve 0,284.
- **clothing:** bom no topo. Entre as sorteadas há casos-limite.
- **product on plain white background:** mede o **layout**, não o produto. Anúncios **só de texto** (texto azul em fundo branco) também passam.
- **a person:** o topo é correto, mas as sorteadas acima de τ incluem pug, besouro, feijão e chuva. Conceitos **genéricos** têm similaridade-base maior com qualquer imagem, então um τ único favorece esses conceitos.

**Melhoria proposta:** usar um τ **por conceito**.

## 3.4 Resultados do item 2.2: busca semântica

![Top-5 das 10 consultas (score · origem · categoria).|74%](../figs/A2/07_busca_top5.png)

| # | Consulta | Tipo | P@5 | Interpretação |
|---|---|---|---|---|
| 1 | a car | concreto, genérico | 5/5 | Correto. Inclui um **meme** recuperado pela **legenda escrita** "...A DOG DRIVING A CAR" (*typographic bias*). |
| 2 | a red sports car on the road | concreto, específico | 1/5 | **Falha de composição.** Esportivo **cinza**, Mini **vermelho** em estúdio, carrinho **vermelho** de criança, estrada ao pôr do sol **avermelhado**. Cada atributo é satisfeito por uma imagem diferente (*attribute binding*). |
| 3 | food | concreto, genérico | 5/5 | Correto. 4/5 vêm das fotos **rejeitadas**: o CLIP recupera o conteúdo, não o sentimento. |
| 4 | a cup of coffee | concreto, específico | 4/5 | *Latte*, xícara, caneca com lareira, pote de Nescafé. Erro: copo de suco. |
| 5 | a smiling woman | atributo | 2/5 | Todas são mulheres, mas o **sorriso** é ignorado (niqab, retratos sérios). |
| 6 | a woman applying makeup | ação | 1/5 | A **ação** se perde e sobra "close de rosto feminino". |
| 7 | a mobile phone advertisement | gênero + produto | 2/5 | A consulta é decomposta em "anúncio" **ou** "celular": anúncios de Fiat e Burger King, e uma foto de celular que não é anúncio. |
| 8 | luxury | abstrato | 1/5 | Scores baixos. O único acerto é **lexical** (anúncio de texto "Luxury Pens"). |
| 9 | freedom and adventure | abstrato | 4/5 | Bons **proxies visuais**: cavalos galopando, montanhas, wakeboard. |
| 10 | family happiness | abstrato, social | 4/5 | Família à mesa, silhueta ao pôr do sol, família numerosa. |

**Padrões:**

1. **Genérico e concreto funciona** (5/5), mas **cada atributo composicional** (cor, expressão, ação) é tratado separadamente do objeto. É uma limitação de modelos contrastivos com embedding global.
2. O **abstrato funciona quando há clichê visual** ("liberdade" vira cavalo e montanha) e **falha quando não há** ("luxo"). Nesse caso, o modelo se apoia em **texto escrito na imagem**.
3. **Typographic bias** aparece em 3 das 10 consultas. Num corpus de publicidade, isso é útil e perigoso ao mesmo tempo.
4. **Viés de template:** com "a photo of…", as fotos naturais vencem os anúncios diagramados em 9 de 10 consultas. Para buscar anúncios, o prompt teria que mudar (o prompt "faz parte do modelo", Aula 06).
5. **Variação do threshold:**
   - Com τ = 0,22, "a mobile phone advertisement" aceita 292 imagens e "luxury" aceita 144, por associação fraca: pet na cama, logo da Chipotle.
   - Com τ = 0,28, só 6 imagens passam somando as 10 consultas.
   - **Para busca, o top-k ordenado é mais útil que um corte absoluto.** O τ serve de filtro mínimo de confiança.

![Zona cinzenta: imagens aceitas só quando τ cai de 0,28 para 0,22.|85%](../figs/A2/09_zona_cinzenta_threshold.png)

## 3.5 Por que o CLIP recupera imagens por texto sem treino supervisionado

**O mecanismo.** O CLIP tem duas torres, um encoder de imagem (ViT-B/32) e um encoder de texto (Transformer). Ambas projetam a entrada para o **mesmo espaço de 512 dimensões**, normalizado em L2. O pré-treino em 400 M pares imagem–legenda da web minimiza a **InfoNCE simétrica**:

`L = ½ [CE(imagem → texto) + CE(texto → imagem)]`, com `S_ij = cos(I_i, T_j)/τ`.

Isso **maximiza a similaridade dos pares corretos** (a diagonal da matriz N×N do *batch*) e **minimiza a dos N²−N pares errados**. Duas consequências tornam a busca possível sem treino:

1. **O texto vira o classificador.** Qualquer frase gera um vetor de consulta, e o vocabulário é aberto. Não há a camada `Linear(D, 1000)` de um modelo supervisionado, presa às classes do treino.
2. **A supervisão vem da linguagem natural**, muito mais rica que um rótulo único: objetos, cenas, ações, estilos e até texto escrito na imagem.

**Evidência medida no ADS-16.** Usamos os ~1.200 pares foto × legenda do participante (nada é ajustado neles):

| Busca legenda → foto (1.183 fotos) | R@1 | R@5 | R@10 | R@50 | Posição mediana da foto certa |
|---|---|---|---|---|---|
| **CLIP zero-shot** | **39,9%** | **64,3%** | **72,8%** | **86,1%** | **2ª** |
| Acaso | 0,08% | 0,42% | 0,85% | 4,23% | 592ª |

![Esquerda: cossenos dentro e entre modalidades. Direita: PCA 2D mostrando o *modality gap* (linhas ligam pares foto–legenda).|100%](../figs/A2/10_alinhamento_modalidades.png)

**Modality gap.** As medidas mostram que as duas modalidades ocupam regiões separadas da hiperesfera:

- cosseno médio imagem×imagem de 0,46 e texto×texto de 0,75;
- imagem×texto de **0,28 no par certo** e **0,19 no par errado**;
- centroides das modalidades a 0,91 de distância.

O objetivo contrastivo só exige a **ordenação** (o par certo mais perto que os errados), não a sobreposição. O sinal útil está na diferença 0,28 vs 0,19, e por isso o threshold teve de ser calibrado.

## 3.6 Consulta textual: CLIP × BERT (tokenização, padding e attention mask)

| | CLIP (encoder de texto) | BERT |
|---|---|---|
| Tokenização | BPE em bytes, 49.408 tokens | WordPiece, 30.522 tokens |
| "a car" | `[49406, 320, 1615, 49407, 0, 0, …]` (77 posições) | `[101, 1037, 2482, 102, 0, …]` |
| Tokens especiais | `<start_of_text>`, `<end_of_text>` | `[CLS]`, `[SEP]`, `[PAD]` |
| Comprimento | **fixo em 77**, com zeros de padding | variável, padding até o maior do *batch* ou `max_length` |
| Máscara | **causal** fixa (−∞ acima da diagonal) | **bidirecional** + `attention_mask` (1 = real, 0 = `[PAD]`) |
| Vetor da frase | estado do `<end_of_text>` → 512-d | estado do `[CLS]` (ou *pooling*) |

**Papel do padding e da attention mask.**

- **No BERT**, a atenção é bidirecional. Sem máscara, todo token (inclusive o `[CLS]`) atenderia aos `[PAD]`, e a representação mudaria conforme o tamanho do padding. A `attention_mask` soma −∞ aos scores das colunas de padding e zera o peso delas no softmax. É o mesmo mecanismo testado no módulo de atenção da A1 (teste 4).
- **No CLIP**, a máscara causal já impede que o `<end_of_text>` veja o que vem depois dele, e o padding vem justamente depois. **Teste:** trocando todo o conteúdo do padding, o embedding da consulta muda **0,00**.

**Consequências para a busca:**

- A consulta vira um vetor único de tamanho fixo, independente do *batch*.
- Consultas com mais de 77 tokens são truncadas.
- A fraca composição de atributos observada (por exemplo, "red sports car") vem principalmente do **objetivo contrastivo**, que raramente exige ligar o atributo ao objeto certo para separar a legenda correta das erradas. Esses modelos se comportam como "saco de palavras" (Yuksekgonul et al., 2023). A leitura causal resumida num único token não compensa isso como a atenção bidirecional do BERT.

## 3.7 Análise crítica

- **O que funcionou:**
  - zero-shot de 60% top-1 em 20 categorias, sem treino;
  - ranking interpretável e estável em torno do τ calibrado;
  - busca excelente para consultas concretas e genéricas.
- **Limitações:**
  - patches de 32 px perdem detalhes pequenos (logos, joias);
  - o embedding global não compõe atributos;
  - um τ único favorece conceitos genéricos;
  - a P@5 foi anotada por um único avaliador.
- **O que eu mudaria:**
  - testar o CLIP ViT-L/14;
  - calibrar τ por conceito;
  - usar prompts específicos por origem ("an advertisement for…" para anúncios);
  - usar o CLIP só como gerador de candidatos, com verificação por um detector de objetos quando a "presença" precisar ser auditável.
