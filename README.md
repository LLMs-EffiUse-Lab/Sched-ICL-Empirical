# Study on Large Language Model Performance

This paper presents an exploratory study that investigates the possibility of enhancing LLM performance by combining LLM techniques and schedule optimization. We focus on prompt design as an LLM optimization technique and formulate the problem as a prompt template assignment task for each query, aiming to maximize performance while minimizing costs (referred to as prompt template allocation problem).

## Study Design
### Research Questions
RQ1: How to build the fitness function that can effectively evaluate candidate solutions without actually submitting them to the LLM?

To address this challenge, we explore the following three methods: 1) similarity-based confidence estimation (SCE), which leverages historical data and similarity measures to estimate the probability of success for prompt templates; 2) machine learning-based prediction, which utilizes historical data to predict the performance of prompt templates; and 3) random selection, which serves as a baseline for comparison.

RQ2: How do different search-based techniques perform in finding optimal solutions for the prompt template allocation problem?

we select classic algorithms from various categories of search-based techniques including evolutionary algorithms, swarm intelligence, and local search methods.

