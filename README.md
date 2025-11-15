
# Probabilistic Knowledge Graph Reasoning with DistMult, Pyro, and Neo4j

This repository implements a full pipeline for building, training, and reasoning over a probabilistic biomedical knowledge graph using the PrimeKG dataset. The aim is to combine neural link prediction via DistMult with Bayesian inference in order to assign confidence scores to knowledge‑graph triples and support multi‑hop probabilistic querying. The project optionally integrates with a Neo4j graph database for storage and interactive exploration.

## Project Overview

The core features of the pipeline are:

- **Knowledge graph embedding** using the DistMult model
- **Triple‑level confidence estimation** by applying a sigmoid function to the learned DistMult scores
- **Bayesian modeling** using a Beta–Bernoulli hierarchical prior via the Pyro probabilistic programming library
- **Posterior inference** using MCMC sampling with the No‑U‑Turn Sampler (NUTS)
- **Optional Neo4j integration** to persist both prior and posterior confidence scores as relationship attributes
- **Multi‑hop probabilistic reasoning**, enabling interpretable queries over chains of relations

## Dependencies

The notebook relies on the following Python libraries:

- `pykeen` for loading PrimeKG and training the DistMult model
- `pyro-ppl` for probabilistic modeling and inference
- `torch-geometric` as the backend for the DistMult implementation
- `neo4j` for Cypher graph queries and data upload
- `pandas`, `numpy` and `matplotlib` for data manipulation and visualisation

To install all the required packages, run:

```bash
pip install --upgrade pip setuptools wheel torch-geometric pykeen neo4j pyro-ppl

## Workflow Breakdown

### 1. Load the PrimeKG Dataset
Using PyKEEN, the biomedical knowledge graph is loaded and partitioned into training, validation and test sets. A few sample triples are printed to show the structure of the data.

### 2. Train the DistMult Model
A DistMult embedding model is trained on a subset of the training triples using margin‑based negative sampling. The model parameters are updated with the Adam optimizer, and the training loss is printed periodically to monitor convergence.

### 3. Compute Prior Probabilities
The raw DistMult scores for sampled triples are passed through the sigmoid function to produce probabilities in the range [0, 1]. These values serve as the model's initial confidence (priors) in the validity of each triple.

### 4. Perform Bayesian Inference
Each prior probability is treated as the mean of a Beta distribution. We construct a Beta–Bernoulli model where each triple has a latent probability sampled from a Beta prior, and the observed outcome is assumed to be true. MCMC sampling with NUTS is run to obtain posterior samples for each triple's probability. The means of these samples form the posterior confidence scores.

### 5. Store and Visualise Posterior Estimates
The DataFrame of triples is extended with both prior and posterior confidence columns. These can be uploaded to Neo4j as relationship properties to visualise and query within a graph database.

### 6. Build an In‑Memory Graph and Reason Over Paths
Using the posterior scores as edge weights, we build an undirected graph and define helper functions:

- `find_paths` discovers all possible paths between two nodes up to a given hop limit
- `aggregate_path_probs` combines probabilities of multiple paths using the noisy‑or model
- `query_multihop` outputs the aggregated probability and details of individual paths for a given start and end node

This setup supports multihop biological inference such as determining how strongly two metabolic processes are connected via intermediate steps.

## Example Output
