# Sentimnet polarity classifcication 
This repository contains the implementation and experiments for applying the Graph Tsetlin Machines for sentiment analysis on text datasets. 
The GraphTM approach represents text as graph structures where words are nodes and relationships between words are edges. 
This allows the Tsetlin Machine to learn patterns in the graph structures for effective sentiment classification.

The experiment included running the GraphTM on 3 datasets

IMDB movie review dataset 
Yelp business review dataset
MPQA Fine-grained opinion 

**IMDB Dataset**

positive/negative sentiments
50,000 movie reviews 
25,000 training, 25,000 testing

Yelp Dataset

Binary sentiment classification (1-2 stars = negative, 4-5 stars = positive, 3 stars= ignored for considering them neutral)
Dataset sample size 50,000 training, 20,000 testing
Balanced class distribution

MPQA Opinion Corpus
positive/negative sentiments
7422 Training, Test 3181 

## Experiment Goals
Experiment the graph Tsetlin Machine for sentiment polarity classifciation
Compare the performance of the Graph TM with the Standard TM and GNN
Evaluate the experminetal results on 3 datasets


## Evaluation Metrics

Classification accuracy 
Per-class accuracy (positive and negative)
Precision, recall, and F1 score
Within run and between run standard deviations
Confusion matrix

## Graph structure:

Each word is a node in the graph
Edges connect words within a window of ±4 positions
Edge labels indicate relative position 
Word properties are attached to nodes

## Experiments and comparsion:

Each of the three datasets are run on a comparsion between the GNN, and Standard TM. 
Each file is seperated by the dataname for the Graph TM experiments.
While for the Standard TM and GNN there is a file sperate for IMDB and a combined for YELP & MPQA

## Installation
1: Using Docker (Recommended)
Docker provides an isolated environment with all dependencies pre-configured, ensuring consistent behavior across different systems.

**Install Docker**
Make sure you have Docker installed on your system:

Docker Desktop for Windows/Mac
For Linux: sudo apt-get install docker.io docker-compose

**Clone the Repository**
bashgit clone https://github.com/cair/GraphTsetlinMachine


**Build and Run the Docker Container**

Using the provided docker-compose.yml file:
Build the Docker image with all required dependencies
Start a container with the application ready to use
Mount your local directory to allow file persistence

**Running Experiments**
You can run experiments inside the Docker container:
bashdocker exec it container name python experiments/run_experiment.py



## Refrences
Graph Tsetlin Machine[Granmo et al., 2025] https://github.com/cair/GraphTsetlinMachine
Vanilla Tsetlin Machine[Granmo et al., 2018] https://github.com/cair/tmu
IMDB Dataset [chollet, 2015] https://github.com/fchollet/keras
YELP Dataset [zhang et al., 2015], https://proceedings.neurips.cc/paper/2015
MPQA Dataset [Wiebe et al., 2005]. https://doi.org/10.1007/s10579-005-7880-9

