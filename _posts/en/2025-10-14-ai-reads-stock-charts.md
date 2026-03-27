---
layout: post
title: "Can AI Read Stock Charts? An Experiment with DistilBERT"
description: "A developer taught a language model to predict price movements through textual descriptions of charts. Testing on 200+ Moscow Exchange stocks showed an AUC of 0.53."
date: 2025-10-14
image: /assets/images/blog/llm_stock_charts.png
tags: [machine learning, Moscow Exchange, experiment]
lang: en
---

Mikhail Shardin conducted an experiment: can a language model predict prices if charts are described in text?

## The Idea

Instead of raw quotes, the model received natural language descriptions: price rising strongly, volume increasing, near resistance.

The DistilBERT model was trained to predict next-day price increases.

## Results

Tested on 200+ Moscow Exchange stocks:

- Average AUC: 0.53 (slightly better than random)
- Best performers: AFLT (0.72), RTSB (0.70), PIKK (0.70)
- Worst performers: PLZL (0.33), VJGZP (0.33)

For trading purposes the result is weak, but the model picked up patterns without direct access to numbers -- that alone is interesting.

## Technology

Python + PyTorch + Hugging Face + Docker. Walk-forward validation, vectorized processing via pandas. The entire process is reproducible.

**Code on GitHub:** [github.com/empenoso/llm-stock-market-predictor](https://github.com/empenoso/llm-stock-market-predictor)

---

**Source:** [Habr](https://habr.com/ru/articles/955612/) | **Author:** Mikhail Shardin
