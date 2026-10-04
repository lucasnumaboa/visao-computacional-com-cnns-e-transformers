# 8. Declaração de uso de Inteligência Artificial

Conforme a política da disciplina ("Sinal Verde"), declaro o uso de ferramentas de IA neste trabalho:

- **Ferramenta:** Claude (Anthropic), via Claude Code.
- **Finalidade:**
  1. Resumir os slides das 8 aulas, para alinhar o projeto ao conteúdo ensinado.
  2. Estruturar o plano de trabalho.
  3. Escrever e depurar o código dos notebooks.
  4. Redigir as análises e este relatório a partir dos resultados obtidos.
- **Verificação:**
  - Todos os notebooks foram executados de ponta a ponta. Os números e as figuras do relatório são os outputs reais dessas execuções e conferem com os arquivos `results.json` de cada notebook.
  - As afirmações sobre os dados foram checadas diretamente nos datasets, por exemplo: a estrutura e as 20 categorias do ADS-16 conferidas no artigo original; a pasta duplicada `data/data` e as 39 duplicatas do dataset da A3; os tipos de fundo do dataset de fundição.
  - A implementação do ViT foi validada numericamente contra o `torchvision.models.vit_b_16`, com diferença máxima de 3,8 × 10⁻⁶ nos logits.
  - Onde um resultado contrariou a expectativa, isso foi reportado como observado e não ajustado.
- **Limites:** a IA pode produzir afirmações imprecisas. Por isso, as afirmações quantitativas do texto se apoiam em outputs dos notebooks ou nas referências abaixo.

# 9. Referências

**Artigos e livros**

1. Vaswani, A. et al. *Attention Is All You Need*. NeurIPS, 2017.
2. Dosovitskiy, A. et al. *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR, 2021.
3. Devlin, J. et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL, 2019.
4. Touvron, H. et al. *Training data-efficient image transformers & distillation through attention* (DeiT). ICML, 2021.
5. Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. ICCV, 2021.
6. He, K. et al. *Deep Residual Learning for Image Recognition*. CVPR, 2016.
7. Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML, 2021.
8. Yuksekgonul, M. et al. *When and Why Vision-Language Models Behave like Bags-of-Words, and What to Do About It?* ICLR, 2023.
9. Yun, S. et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*. ICCV, 2019.
10. Abnar, S.; Zuidema, W. *Quantifying Attention Flow in Transformers* (attention rollout). ACL, 2020.
11. Darcet, T.; Oquab, M.; Mairal, J.; Bojanowski, P. *Vision Transformers Need Registers*. ICLR, 2024.
12. Goodfellow, I. et al. *Generative Adversarial Nets*. NeurIPS, 2014.
13. Mirza, M.; Osindero, S. *Conditional Generative Adversarial Nets*. arXiv:1411.1784, 2014.
14. Radford, A.; Metz, L.; Chintala, S. *Unsupervised Representation Learning with Deep Convolutional GANs* (DCGAN). ICLR, 2016.
15. Zhu, J.-Y. et al. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (CycleGAN). ICCV, 2017.
16. Karras, T. et al. *Progressive Growing of GANs for Improved Quality, Stability, and Variation* (minibatch-std). ICLR, 2018.
17. Miyato, T. et al. *Spectral Normalization for Generative Adversarial Networks*. ICLR, 2018.
18. Heusel, M. et al. *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (TTUR, FID). NeurIPS, 2017.
19. Salimans, T. et al. *Improved Techniques for Training GANs*. NeurIPS, 2016.
20. Bińkowski, M. et al. *Demystifying MMD GANs* (KID). ICLR, 2018.
21. Artigo indicado no enunciado como "Nour & Tariq (2023)", https://www.nature.com/articles/s41598-023-37743-4. Segundo o Crossref e o Europe PMC (PMC10313722), esse DOI corresponde a: Dumakude, A.; Ezugwu, A. E. *Automated COVID-19 detection with convolutional neural networks*. Scientific Reports, 13, 10607, 2023.
22. Roffo, G.; Vinciarelli, A. *Personality in Computational Advertising: A Benchmark*. EMPIRE Workshop, 2016.
23. Chowdhury, M. E. H. et al. *Can AI Help in Screening Viral and COVID-19 Pneumonia?* IEEE Access, 8, 2020.
24. Rahman, T. et al. *Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection Using Chest X-ray Images*. Computers in Biology and Medicine, 2021.
25. Meegle. *Transfer Learning for Traffic Analysis*. Disponível em: https://www.meegle.com/en_us/topics/transfer-learning/transfer-learning-for-traffic-analysis

**Datasets** (Kaggle)

- *Casting Product Image Data for Quality Inspection*, Ravirajsinh Dabhi. CC BY-NC-ND 4.0. `ravirajsinh45/real-life-industrial-dataset-of-casting-product`
- *ADS-16 Computational Advertising Dataset*, Giorgio Roffo. `groffo/ads16-dataset`
- *Images Dataset*, Pavan Sanagapati. CC0. `pavansanagapati/images-dataset`
- *COVID-19 Radiography Database*, Tawsifur Rahman et al. `tawsifurrahman/covid19-radiography-database`

**Software:** PyTorch e TorchVision; OpenAI CLIP (`github.com/openai/CLIP`); torch-fidelity; scikit-learn; Matplotlib; pandas.

**Material da disciplina:** slides das Aulas 01 a 08 de *Visão Computacional com CNNs e Transformers*, INFNET.
