# Study on Large Language Model Performance

This paper presents an exploratory study that investigates the possibility of enhancing LLM performance by combining LLM techniques and schedule optimization. We focus on prompt design as an LLM optimization technique and formulate the problem as a prompt template assignment task for each query, aiming to maximize performance while minimizing costs (referred to as prompt template allocation problem).

### 1. Study Design

<p align="center"><img src="figs/framework.png" width="800"><br></p>


### 2. Research Questions
 - RQ1: What kind of prediction techniques are helpful to evaluate the accuracy objective in the fitness function without actually submitting the queries to the LLM?

 - RQ2: How do various search-based techniques perform in identifying optimal solutions for the prompt template allocation problem?

 - RQ3: How generalizable are our findings across different LLMs and datasets?

### 3. Additional Results

<p align="center"><img src="figs/training_size.png" width="800"><br>Impact of Training Data Size on Prediction Accuracy and Label Collection Cost</p>

