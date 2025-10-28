# Awesome Prior-Data Fitted Networks

A curated collection of papers, repositories, and resources on **Prior-data Fitted Networks (PFNs)**.



## What are PFNs?

As the name says, [**Prior-data Fitted Networks (PFNs)**](https://www.automl.org/pfns) are a class of neural networks trained on **synthetic datasets sampled from a prior distribution** to directly approximate the **posterior predictive distribution (PPD)**. They enable **Bayesian prediction** through in-context learning and have been applied across tabular data, time series, Bayesian optimization, symbolic regression, and beyond.

### Example — TabPFN

A landmark success of the PFN framework is the **Tabular Foundation Model (TabPFN)** published in *Nature*: [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6). It showed that a TabPFN trained on over 100 million synthetic tabular tasks can outperform traditional ML models (like XGBoost, CatBoost, LightGBM) on a wide range of small real datasets.  

- Key idea: Train a transformer on 100 million synthetic tabular tasks, enabling zero-training predictions on new datasets.  
- Architecture: An adapted transformer encoder designed for two-dimensional tabular data, supporting categorical + numeric features, missing values, and heterogeneous distributions.  
- Impact: TabPFN demonstrates that PFNs can act as *foundation models* for tabular data, achieving state-of-the-art accuracy on small-data benchmarks in seconds, outperforming conventional AutoML systems.

This work established PFNs as a new family of **foundation models for structured data**, analogous to LLMs for text.

<p align="center">
  <img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-024-08328-6/MediaObjects/41586_2024_8328_Fig1_HTML.png" alt="Overview of TabPFN" width="650">
  <br>
  <em>Figure: Overview of TabPFN.</em>
</p>



## Highlights

| Venue     | Title                                                        | Code                                        |
| :-------- | :----------------------------------------------------------- | :------------------------------------------ |
| ICLR 2023 | [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848) | [Code](https://github.com/PriorLabs/TabPFN) |
| Nature    | [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6) | [Code](https://github.com/PriorLabs/TabPFN) |



## Foundations

*Foundations and theoretical insights into PFNs, amortized inference, and Bayesian learning.*

| Venue        | Title                                                        | Code                                                         |
| :----------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| ICLR 2022    | [Transformers Can Do Bayesian Inference](https://arxiv.org/abs/2112.10510) | [Code](https://github.com/automl/TransformersCanDoBayesianInference) |
| ICML 2023    | [Statistical Foundations of Prior-Data Fitted Networks](https://arxiv.org/abs/2305.11097) | [Code](https://gist.github.com/tnagler/62f6ce1f996333c799c81f1aef147e72) |
| NeurIPS 2023 | [Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection](https://arxiv.org/abs/2306.04637) | [Code](https://github.com/allenbai01/transformers-as-statisticians) |
| ICML 2024    | [Is In-Context Learning in Large Language Models Bayesian? A Martingale Perspective](https://arxiv.org/abs/2406.00793) | [Code](https://github.com/meta-inf/bayes_icl)                |
| ICML 2025    | [Can Transformers Learn Full Bayesian Inference in Context?](https://arxiv.org/abs/2501.16825) | [Code]([DongWooLee-Eli/nslpfn](https://github.com/DongWooLee-Eli/nslpfn)) |



## Papers

| Venue                   | Title                                                        | Code                                                         |
| :---------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| UnderReview @ ICLR 2026 | [PDE-PFN: Prior-Data Fitted Neural PDE Solver](https://openreview.net/forum?id=z7ilspv4uH) | —                                                            |
| UnderReview @ ICLR 2026 | [SR-PFN: Yet Another Sequential Recommendation Paradigm](https://openreview.net/forum?id=xffb9X08Fv) | —                                                            |
| UnderReview @ ICLR 2026 | [Time-Aware Prior Fitted Networks for Zero-Shot Forecasting with Exogenous Variables](https://openreview.net/forum?id=90HpWIBBwE) | —                                                            |
| UnderReview @ ICLR 2026 | [Transformers Can Do Bayesian Clustering](https://openreview.net/forum?id=MCya4TeDW6) | —                                                            |
| UnderReview @ ICLR 2026 | [DistPFN: Test-Time Posterior Adjustment for Tabular Foundation Models under Label Shift](https://openreview.net/forum?id=vlpAgjkw39) | —                                                            |
| UnderReview @ ICLR 2026 | [Task-Aligned Attention Retrieval for Scaling Tabular Foundation Models](https://openreview.net/forum?id=qBMNsGiE3u) | —                                                            |
| UnderReview @ ICLR 2026 | [Large-Scale Pretraining Offers Modest Benefits for Tabular Transfer Learning](https://openreview.net/forum?id=G5zJaSxMGN) | —                                                            |
| UnderReview @ ICLR 2026 | [MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning](https://openreview.net/forum?id=pSyuFl8mau) | —                                                            |
| UnderReview @ ICLR 2026 | [RaBEL: Scale-Aware Radial-Basis Embeddings for Tabular Foundation Models](https://openreview.net/forum?id=odoTDh3QUk) | —                                                            |
| UnderReview @ ICLR 2026 | [Using maximal information auxiliary variables to improve synthetic data generation based on TabPFN foundation models](https://openreview.net/forum?id=6PkiUAcTWF) | —                                                            |
| arXiv 2025              | [GIT-BO: High-Dimensional Bayesian Optimization with Tabular Foundation Models](https://openreview.net/forum?id=9iTdKS4SRQ) | [Code](https://github.com/deepbiolab/gitbo)                  |
| arXiv 2025              | [TabPFN: One Model to Rule Them All?](https://arxiv.org/abs/2505.20003) | [Code](https://github.com/qinglong-tian/tabpfn_study)        |
| arXiv 2025              | [TabImpute: Accurate and Fast Zero-Shot Missing-Data Imputation with a Pre-Trained Transformer](https://arxiv.org/abs/2510.02625) | [Code](https://github.com/jacobf18/tabular)                  |
| arXiv 2025              | [Decoupled-Value Attention for Prior-Data Fitted Networks: GP Inference for Physical Equations](https://arxiv.org/abs/2509.20950) | [Code](https://github.com/PSquare-Lab/DVA-PFN)               |
| arXiv 2025              | [Efficient Autoregressive Inference for Transformer Probabilistic Models](https://arxiv.org/abs/2510.09477) | [Code](https://github.com/acerbilab/transformer-ar-buffer)   |
| arXiv 2025              | [GraphPFN: A Prior-Data Fitted Graph Foundation Model](https://arxiv.org/abs/2509.21489) | [Code](https://github.com/yandex-research/graphpfn)          |
| arXiv 2025              | [Turning Tabular Foundation Models into Graph Foundation Models](https://arxiv.org/abs/2508.20906) | [Code](https://github.com/yandex-research/G2T-FM)            |
| arXiv 2025              | [Bringing Graphs to the Table: Zero-shot Node Classification via Tabular Foundation Models](https://arxiv.org/abs/2509.07143) | [Code](https://github.com/ahayler/tag)                       |
| arXiv 2025              | [LimiX: Unleashing Structured-Data Modeling Capability for Generalist Intelligence](https://arxiv.org/abs/2509.03505) | [Code](https://github.com/limix-ldm/LimiX)                   |
| arXiv 2025              | [Gradient Free Deep Reinforcement Learning With TabPFN](https://arxiv.org/abs/2509.11259) | —                                                            |
| arXiv 2025              | [Clustering by Attention: Leveraging Prior Fitted Transformers for Data Partitioning](https://arxiv.org/abs/2507.20369) | —                                                            |
| arXiv 2025              | [On Finetuning Tabular Foundation Models](https://arxiv.org/abs/2506.08982) | [Code](https://github.com/yandex-research/tabpfn-finetuning) |
| arXiv 2025              | [TabPFN-Wide: Continued Pre-Training for Extreme Feature Counts](https://arxiv.org/abs/2510.06162) | [Code](https://github.com/pfeiferAI/TabPFN-Wide)             |
| arXiv 2025              | [Foundation Models for Causal Inference via Prior-Data Fitted Networks](https://arxiv.org/abs/2506.10914) | [Code](https://github.com/yccm/CausalFM)                     |
| arXiv 2025              | [Chunked TabPFN: Exact Training-Free In-Context Learning for Long-Context Tabular Data](https://arxiv.org/abs/2509.00326) | [Code](https://github.com/mrsergazinov/chunk_tabpfn)         |
| arXiv 2025              | [From Tables to Time: How TabPFN-v2 Outperforms Specialized Time Series Forecasting Models](https://arxiv.org/abs/2501.02945) | [Code](https://github.com/PriorLabs/tabpfn-time-series)      |
| arXiv 2025              | [Realistic Evaluation of TabPFN v2 in Open Environments](https://arxiv.org/abs/2505.16226) | [Code](https://anonymous.4open.science/r/tabpfn-ood-4E65)    |
| SSRN 2025               | [MultiTabPFN: Codebook-based Extensions of TabPFN for High-Class-Count Tabular Classification](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5545797) | —                                                            |
| TCBBIO 2025             | [GPFN: Prior-Data Fitted Networks for Genomic Prediction](https://www.biorxiv.org/content/10.1101/2023.09.20.558648) | [Code](https://github.com/jubbens/gpfn)                      |
| RML @ NeurIPS 2025      | [Robust Multi-task Modeling for Bayesian Optimization via In-Context Learning](https://openreview.net/forum?id=iwqJLEPgvF) | —                                                            |
| NeurIPS 2025            | [Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models) | [Code](https://huggingface.co/autogluon/mitra-classifier)    |
| NeurIPS 2025            | [Effortless, Simulation-Efficient Bayesian Inference using Tabular Foundation Models](https://arxiv.org/abs/2504.17660) | [Code](https://github.com/mackelab/npe-pfn)                  |
| NeurIPS 2025            | [Do-PFN: In-context Learning for Causal Effect Estimation](https://arxiv.org/abs/2506.06039) | [Code](https://github.com/jr2021/Do-PFN)                     |
| NeurIPS 2025            | [ConTextTab: A Semantics-Aware Tabular In-Context Learner](https://arxiv.org/abs/2506.10707) | [Code](https://github.com/SAP-samples/contexttab)            |
| NeurIPS 2025            | [ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data](https://arxiv.org/abs/2505.10704) | [Code](https://github.com/gmum/zeus)                         |
| NeurIPS 2025            | [TabDPT: An Open Tabular Foundation Model](https://arxiv.org/abs/2410.18164) | [Code](https://github.com/layer6ai-labs/TabDPT-inference)    |
| NeurIPS 2025            | [CausalPFN: Amortized Causal Effect Estimation via In-Context Learning](https://arxiv.org/abs/2506.07918) | [Code](https://github.com/vdblm/CausalPFN)                   |
| NeurIPS 2025            | [A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities](https://arxiv.org/abs/2502.17361) | —                                                            |
| NeurIPS 2025            | [EquiTabPFN: A Target-Permutation Equivariant Prior Fitted Networks](https://arxiv.org/abs/2502.06684) | [Code](https://github.com/geoalgo/equitabpfn)                |
| ECAI 2025               | [In-Context Decision Making for Optimizing Complex AutoML Pipelines](https://arxiv.org/abs/2508.13657) | [Code](https://github.com/amirbalef/CASHPlus)                |
| FMSD @ ICML 2025        | [State-Space Models for Tabular Prior-Data Fitted Networks](https://arxiv.org/abs/2510.14573) | [Code](https://github.com/felixmkoch/Structured-State-Space-Models-for-PFNs) |
| FMSD @ ICML 2025        | [Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data](https://arxiv.org/abs/2507.03971) | —                                                            |
| FMSD @ ICML 2025        | [From Tabular to Time Series: Can TabPFN Handle Mixed Data? A Study on PhysioNet](https://openreview.net/forum?id=HVugmZyXbd) | —                                                            |
| FMSD @ ICML 2025        | [Early Stopping Tabular In-Context Learning](https://arxiv.org/abs/2506.21387) | —                                                            |
| FMSD @ ICML 2025        | [Explore the Time Series Forecasting Potential of TabPFN Leveraging the Intrinsic Periodicity of Data](https://openreview.net/forum?id=7JGD1kNlzU) | [Code](https://github.com/sibo-cai/TabPFN-TSP)               |
| ICML 2025               | [Position: The Future of Bayesian Prediction Is Prior-Fitted](https://arxiv.org/abs/2505.23947) | —                                                            |
| ICML 2025               | [Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks](https://arxiv.org/abs/2505.23032) | [Code]([DongWooLee-Eli/nslpfn](https://github.com/DongWooLee-Eli/nslpfn)) |
| ICML 2025               | [Zero-shot Meta-learning for Tabular Prediction Tasks with Adversarially Pre-trained Transformer](https://arxiv.org/abs/2502.04573) | [Code](https://github.com/yulun-rayn/APT)                    |
| ICML 2025               | [TabPFN-Unleashed: A Scalable and Effective Solution to Tabular Prediction](https://proceedings.mlr.press/v267/liu25cn.html) | [Code](https://github.com/LAMDA-Tabular/BETA)                |
| ICML 2025               | [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](https://arxiv.org/abs/2502.05564) | [Code](https://github.com/soda-inria/tabicl)                 |
| ICML 2025               | [TabFlex: Scaling Tabular Learning to Millions with Linear Attention](https://arxiv.org/abs/2506.05584v1) | [Code](https://github.com/tuanqdinh/ICML25_TabFlex)          |
| ICML 2025               | [FairPFN: A Tabular Foundation Model for Causal Fairness](https://arxiv.org/abs/2506.07049) | [Code](https://github.com/jr2021/FairPFN)                    |
| ICML 2025               | [Can Transformers Learn Full Bayesian Inference in Context?](https://arxiv.org/abs/2501.16825) | [Code](https://github.com/ArikReuter/ICL_for_Full_Bayesian_Inference) |
| AISTATS 2025            | [Prior-Fitted Networks Scale to Larger Datasets When Treated as Weak Learners](https://arxiv.org/abs/2503.01256) | [Code](https://github.com/yxzwang/BoostPFN)                  |
| TMLR 2025               | [FoMo-0D: A Foundation Model for Zero-shot Tabular Outlier Detection](https://arxiv.org/abs/2409.05672) | [Code](https://github.com/A-Chicharito-S/FoMo-0D)            |
| AABI @ ICLR 2025        | [Uncertainty Quantification for Prior-Data Fitted Networks using Martingale Posteriors](https://arxiv.org/abs/2505.11325) | —                                                            |
| FPI @ ICLR 2025         | [α-PFN: In-Context Learning Entropy Search](https://openreview.net/forum?id=IMVqPGYxyD) | —                                                            |
| ICLR 2025               | [Mixture of In-Context Prompters for Tabular PFNs](https://arxiv.org/abs/2405.16156) | —                                                            |
| ICLR 2025               | [KinPFN: Bayesian Approximation of RNA Folding Kinetics using Prior-Data Fitted Networks](https://openreview.net/forum?id=E1m5yGMOiV) | [Code](https://github.com/automl/KinPFN)                     |
| AAAI 2025               | [TimePFN: Effective Multivariate Time Series Forecasting with Synthetic Data](https://arxiv.org/abs/2502.16294) | [Code](https://github.com/egetaga/TimePFN)                   |
| ICLR 2025               | [MotherNet: Fast Training and Inference via Hyper-Network Transformers](https://arxiv.org/abs/2312.08598) | [Code](https://github.com/microsoft/ticl)                    |
| Nature 2025             | [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6) | [Code](https://github.com/PriorLabs/TabPFN)                  |
| OpenReview              | [Attic: A New Architecture for Tabular In-Context Learning Transformers](https://openreview.net/forum?id=DSl9sSuUhp) | [Code](https://openreview.net/attachment?id=DSl9sSuUhp&name=supplementary_material) |
| arXiv 2024              | [LaT-PFN: A Joint Embedding Predictive Architecture for In-context Time-series Forecasting](https://arxiv.org/abs/2405.10093) | [Code](https://github.com/StijnVerdenius/Lat-PFN)            |
| arXiv 2024              | [Fine-tuned In-Context Learning Transformers are Excellent Tabular Data Classifiers](https://arxiv.org/abs/2405.13396) | [Code](https://openreview.net/attachment?id=pE0UM18TQh&name=supplementary_material) |
| arXiv 2024              | [Tokenize Features, Enhancing Tables: The FT-TabPFN Model for Tabular Classification](https://arxiv.org/abs/2406.06891) | [Code](https://github.com/ds-brx/seminar-LLMTab-FTtabpfn)    |
| xAI 2024                | [Interpretable Machine Learning for TabPFN](https://arxiv.org/abs/2403.10923) | [Code](https://github.com/david-rundel/tabpfn_iml)           |
| TSALM @ NeurIPS 2024    | [Mamba4Cast: Efficient Zero-Shot Time Series Forecasting with State Space Models](https://openreview.net/forum?id=YBOQ5HnzI6) | [Code](https://github.com/automl/Mamba4Cast)                 |
| TRL @ NeurIPS 2024      | [GAMformer: Bridging Tabular Foundation Models and Interpretable Machine Learning](https://openreview.net/forum?id=5Taa8ZaZ5o) | —                                                            |
| TRL @ NeurIPS 2024      | [Adapting TabPFN for Zero-Inflated Metagenomic Data](https://openreview.net/forum?id=3I0bVvUj25#discussion) | —                                                            |
| TRL @ NeurIPS 2024      | [Towards Localization via Data Embedding for TabPFN](https://openreview.net/forum?id=LFyQyV5HxQ) | —                                                            |
| TRL @ NeurIPS 2024      | [Exploration of autoregressive models for in-context learning on tabular data](https://openreview.net/forum?id=4dOJ0PRY7R) | —                                                            |
| NeurIPS 2024            | [TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks](https://arxiv.org/abs/2402.11137) | [Code](https://github.com/penfever/TuneTables)               |
| NeurIPS 2024            | [Retrieval & Fine-Tuning for In-Context Tabular Models](https://proceedings.nips.cc/paper_files/paper/2024/hash/c40daf14d7a6469e65116507c21faeb7-Abstract-Conference.html) | [Code](https://github.com/layer6ai-labs/LoCalPFN)            |
| NeurIPS 2024            | [Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data](https://arxiv.org/abs/2411.10634) | [Code](https://github.com/automl/Drift-Resilient_TabPFN)     |
| NeurIPS 2024            | [TabEBM: A Tabular Data Augmentation Method with Distinct Class-Specific Energy-Based Models](https://arxiv.org/abs/2409.16118) | [Code](https://github.com/andreimargeloiu/TabEBM)            |
| ICL @ ICML 2024         | [TabMDA: Tabular Manifold Data Augmentation for Any Classifier using Transformers with In-context Subsetting](https://arxiv.org/abs/2406.01805) | [Code](https://github.com/AdrianBZG/TabMDA)                  |
| ICML 2024               | [In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization](https://arxiv.org/abs/2404.16795) | [Code](https://github.com/automl/ifBO)                       |
| ME-FoMo @ ICLR 2024     | [In-Context Data Distillation with TabPFN](https://arxiv.org/abs/2402.06971) | —                                                            |
| Blogposts  @ ICLR 2024  | [What exactly has TabPFN learned to do?](https://arxiv.org/abs/2502.08978) | [Code](https://github.com/calvinmccarter/tabpfn-eval)        |
| TRL @ NeurIPS 2023      | [Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks](https://arxiv.org/abs/2311.10609) | [Code](https://github.com/penfever/TuneTables)               |
| TRL @ NeurIPS 2023      | [Fine-Tuning the Retrieval Mechanism for Tabular Deep Learning](https://arxiv.org/abs/2311.07343) | —                                                            |
| TRL @ NeurIPS 2023      | [TabPFGen -- Tabular Data Generation with TabPFN](https://arxiv.org/abs/2406.05216) | [Code](https://github.com/sebhaan/TabPFGen)                  |
| NeurIPS 2023            | [Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks](https://arxiv.org/abs/2310.20447) | [Code](https://github.com/automl/lcpfn)                      |
| NeurIPS 2023            | [ForecastPFN: Synthetically-Trained Zero-Shot Forecasting](https://arxiv.org/abs/2311.01933) | [Code](https://github.com/abacusai/ForecastPFN)f             |
| ICML 2023               | [PFNs4BO: In-Context Learning for Bayesian Optimization](https://arxiv.org/abs/2305.17535) | [Code](https://github.com/automl/PFNs4BO)                    |
| ICML 2023               | [Statistical Foundations of Prior-Data Fitted Networks](https://arxiv.org/abs/2305.11097) | [Code](https://gist.github.com/tnagler/62f6ce1f996333c799c81f1aef147e72) |
| ICLR 2023               | [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848) | [Code](https://github.com/PriorLabs/TabPFN)                  |
| ICLR 2022               | [Transformers Can Do Bayesian Inference](https://arxiv.org/abs/2112.10510) | [Code](https://github.com/automl/TransformersCanDoBayesianInference) |



## GitHub Repositories

| Repository                                                   | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **[`automl/PFNs`](https://github.com/automl/PFNs)**          | Canonical PFN implementation; synthetic task generation, Bayesian inference via transformers. |
| **[`PriorLabs/TabPFN`](https://github.com/PriorLabs/TabPFN)** | Official Tabular PFN implementation (classification + regression) |
| **[`PriorLabs/tabpfn-extensions`](https://github.com/PriorLabs/tabpfn-extensions)** | Extensions: interpretability, more classes, imputation, and analysis tools |
| **[`PriorLabs/awesome-tabpfn`](https://github.com/PriorLabs/awesome-tabpfn)** | Community-curated list of TabPFN applications and papers     |
| **[`PriorLabs/tabpfn-time-series`](https://github.com/PriorLabs/tabpfn-time-series)** | Time-series adaptation of TabPFN                             |
| **[`david-rundel/tabpfn_iml`](https://github.com/david-rundel/tabpfn_iml)** | Interpretability module for TabPFN (SHAP, feature attribution) |
| **[`yandex-research/G2T-FM`](https://github.com/yandex-research/G2T-FM)** | Graph-to-Table Foundation Model: extend TabPFN to graph data |
| **[`yandex-research/graphpfn`](https://github.com/yandex-research/graphpfn)** | GraphPFN: PFN with graph priors and message-passing transformer |
| **[`abacusai/ForecastPFN`](https://github.com/abacusai/ForecastPFN)** | PFN for zero-shot time-series forecasting                    |
| **[`soda-inria/tabicl`](https://github.com/soda-inria/tabicl)** | a more scalable tabular foundation model                     |
| **[`limix-ldm/LimiX`](https://github.com/limix-ldm/LimiX)**  | LimiX: a tabular foundation model generalizing TabPFN        |
| **[`autogluon/tabrepo`](https://github.com/autogluon/tabrepo)** | Living benchmark for tabular ML; includes foundation-model baselines |
| **[`autogluon/autogluon`](https://github.com/autogluon/autogluon)** | End-to-end AutoML framework supporting tabular, time-series, and multimodal data; often used as benchmark in PFN/TabPFN work |



## Other Applications

1. [Predicting Early Outcomes of Prostatic Artery Embolization Using n-Butyl Cyanoacrylate Liquid Embolic Agent: A Machine Learning Study](https://www.mdpi.com/2075-4418/15/11/1351)
2. [Virtual Screening of Natural Anti-Senescent Compounds Based on Sq-TabPFN](https://ieeexplore.ieee.org/document/10864333)
3. [From Rows to Yields: How Foundation Models for Tabular Data Simplify Crop Yield Prediction](https://arxiv.org/abs/2506.19046)
4. [Uncertainty-Aware Tabular Prediction: Evaluating VBLL-Enhanced TabPFN in Safety-Critical Medical Data](https://arxiv.org/abs/2509.10048)
5. [Early Fault Classification in Rotating Machinery With Limited Data Using TabPFN](https://ieeexplore.ieee.org/document/10318062)
6. [Improved Ethereum Fraud Detection Mechanism with Explainable Tabular Transformer Model](https://ieeexplore.ieee.org/document/10835625)
7. [Fast and Accurate Zero-Training Classification for Tabular Engineering Data](https://arxiv.org/abs/2401.06948)
8. [A Fast and Reliable Transformer-Based TabPFN Model for Liver Disease Diagnosis](https://www.cureusjournals.com/articles/4072-a-fast-and-reliable-transformer-based-tabpfn-model-for-liver-disease-diagnosis#!/)
9. [Machine learning and radiomics for ventricular tachyarrhythmia prediction in hypertrophic cardiomyopathy: insights from an MRI-based analysis](https://pubmed.ncbi.nlm.nih.gov/39350610/)
10. [Explainable Classification for Non-Small Cell Lung Cancer Based on Positron Emission Tomography Features and Clinical Data](https://ieeexplore.ieee.org/abstract/document/10345893)
11. [Class-Imbalanced-Aware Adaptive Dataset Distillation for Scalable Pretrained Model on Credit Scoring](https://arxiv.org/abs/2501.10677)
12. [Fault Diagnosis of Slewing Bearing Using Audible Sound Signal Based on Time Generative Adversarial Network–TabPFN Method](https://doi.org/10.1115/1.4068223)
13. [A machine learning-based approach for individualized prediction of short-term outcomes after anterior cervical corpectomy](https://pubmed.ncbi.nlm.nih.gov/39113482/)
14. [Foundation Models for Cybersecurity: A Comprehensive Multi-Modal Evaluation of TabPFN and TabICL for Tabular Intrusion Detection](https://www.mdpi.com/2079-9292/14/19/3792)
15. [TACO: TabPFN Augmented Causal Outcomes for Early Detection of Long COVID](https://www.medrxiv.org/content/10.1101/2025.10.02.25337138)
16. [Tabular Data with Class Imbalance: Predicting Electric Vehicle Crash Severity with Pretrained Transformers (TabPFN) and Mamba-Based Models](https://arxiv.org/abs/2509.11449)
17. [Tabular prior-data fitted network for urban air temperature inference and high temperature risk assessment](https://www.sciencedirect.com/science/article/abs/pii/S2210670725003609)
18. [Kriging prior Regression: A Case for Kriging-Based Spatial Features with TabPFN in Soil Mapping](https://arxiv.org/abs/2509.09408)
19. [Uncertainty-Aware Tabular Prediction: Evaluating VBLL-Enhanced TabPFN in Safety-Critical Medical Data](https://arxiv.org/abs/2509.10048)
20. [Tabular foundation model for GEOAI benchmark problems BM/AirportSoilProperties/2/2025](https://arxiv.org/abs/2509.03191)



## Contributing

Contributions are welcome!

- Add papers, repos, or tutorials.
- Fix broken links.  
- Suggest new categories.  



## License & Credits

This list aggregates publicly available PFN-related resources. Each work retains its own license and citation requirements. If you use these resources, please cite the corresponding paper and repository.

