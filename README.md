
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
