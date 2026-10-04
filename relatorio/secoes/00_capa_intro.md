<div class="cover" markdown="1">

<div class="repo">
<b>Código, notebooks e instruções para rodar:</b><br>
<a href="https://github.com/lucasnumaboa/visao-computacional-com-cnns-e-transformers">https://github.com/lucasnumaboa/visao-computacional-com-cnns-e-transformers</a><br>
<code>git clone https://github.com/lucasnumaboa/visao-computacional-com-cnns-e-transformers.git</code><br>
<span class="small">Ou abra cada notebook direto no Google Colab pelos botões "Abrir no Colab" do README (GPU T4 → Executar tudo).</span>
</div>

<h1>Visão Computacional com CNNs e Transformers</h1>

<div class="sub">Projeto de Disciplina: relatório técnico</div>

<div class="meta">
<b>Aluno:</b> Lucas de Oliveira Ferreira<br>
<b>Instituição:</b> Instituto INFNET<br>
<b>Disciplina:</b> Visão Computacional com CNNs e Transformers [26E3_3]<br>
<b>Data:</b> outubro de 2026
</div>

</div>

# Sumário

<div class="toc" markdown="1">

1. Introdução e visão geral
2. Atividade 1: Vision Transformers em inspeção visual de qualidade (fundição)
3. Atividade 2: reconhecimento semântico em publicidade visual com CLIP (ADS-16)
4. Atividade 3: classificador com CNN pré-treinada (feature extraction)
5. Atividade 4.1: estudo de caso de COVID-19 em radiografias de tórax, com GAN condicional
6. Atividade 4.2: transfer learning para análise de tráfego urbano
7. Conclusões
8. Declaração de uso de Inteligência Artificial
9. Referências

</div>

# 1. Introdução e visão geral

Este relatório documenta as quatro atividades do Projeto de Disciplina. Em cada uma, o foco não é só fazer o código rodar, mas **decidir qual arquitetura usar, por quê, e como adaptá-la ao problema**, e depois verificar com dados se a decisão se sustenta. Cada atividade segue a mesma estrutura: definição do problema, decisões técnicas com justificativa, resultados com métricas e gráficos, e análise crítica.

| Atividade | Problema | Abordagem | Notebook |
|---|---|---|---|
| A1 | Detectar defeitos em peças fundidas (rotores de bomba) | Atenção e ViT **implementados módulo a módulo** (com testes), pesos ImageNet, fine-tuning, comparação com ViT do zero, Swin-T e ResNet-50, análise de atenção | `A1_vision_transformers.ipynb` |
| A2 | Extrair semântica de um corpus de anúncios sem treinar | CLIP ViT-B/32: ranking de conceitos com threshold calibrado, busca texto→imagem, alinhamento das modalidades e tokenização CLIP × BERT | `A2_clip_ads16.ipynb` |
| A3 | Classificar 7 categorias de objetos | ResNet-50 congelada + nova camada linear; teste de estresse para embasar as melhorias | `A3_cnn_kaggle.ipynb` |
| A4.1 | Diagnosticar e corrigir uma triagem de COVID-19 que "ignora positivos" | Diagnóstico, cGAN (DCGAN condicional) com ablação de estabilidade, experimento controlado de recall, plano e critério de adoção clínica | `A4_estudo_caso_raio_x.ipynb` |
| A4.2 | Diagnosticar um sistema de tráfego que falha em produção | Análise escrita (sem implementação) | — |

**Reprodutibilidade.** O código, os notebooks e as instruções estão em https://github.com/lucasnumaboa/visao-computacional-com-cnns-e-transformers (botão "Abrir no Colab" para cada notebook). Os quatro notebooks rodam do início ao fim no Google Colab com GPU T4, sem montar o Drive e sem credenciais. Os datasets são baixados do endpoint público do Kaggle (`/api/v1/datasets/download/<owner>/<dataset>`), com `kagglehub` como alternativa. Cada notebook informa no topo a memória e o tempo estimados, e fixa sementes aleatórias. Os resultados deste relatório vêm da execução completa dos notebooks numa GPU NVIDIA RTX 3050 (6 GB). O código detecta a VRAM disponível e mantém o mesmo batch efetivo por acumulação de gradiente, de modo que os números se reproduzem no T4.

**Alinhamento com a disciplina.** As técnicas usadas vêm das aulas: TorchVision Weights API e feature extraction (Aula 01), self-attention e bloco Transformer (Aula 02), receita de fine-tuning de encoders (Aula 03), ViT, CutMix e attention rollout (Aula 04), atenção por cabeça (Aula 05), CLIP e prompt ensembling (Aula 06), DCGAN/cGAN e estabilização (Aula 07), FID, calibração de threshold e protocolo de avaliação com dados sintéticos (Aula 08). Quando uma escolha foge do material das aulas, ela é justificada no texto.
