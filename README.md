# EEG Based Brain Computer Interface System
![](https://pub.mdpi-res.com/applsci/applsci-10-05662/article_deploy/html/images/applsci-10-05662-ag.png?1597564574)
This repository presents a complete pipeline for EEG-based brain-computer interface (BCI) tasks, developed as part of the MTC-AIC3 Competition, where our team ranked among the top 16 out of 215+ teams in the qualification round.

The system addresses both Motor Imagery (MI) and SSVEP paradigms, focusing on cross-subject generalization through deep learning and statistical feature analysis.

The Motor Imagery part consists of a system built around a custom Universal Feature Extractor (UFE), followed by per-subject lightweight networks (DDFilter) to analyze and select robust latent features.

The SSVEP part is composed of feature extraction using FBCCA and a classifier.

System descriptions and methodology explanations are included for both pipelines. Pretrained models, checkpoints, and reproducibility assets (e.g., requirements.txt, dependencies/) are also provided.

