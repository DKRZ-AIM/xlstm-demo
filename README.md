# Overview

This repository is a merge of the first "Understanding Transformers Workshop" repo by Danu Caus at https://github.com/ducspe/understanding_transformers_workshop with the original xLSTM code by NX-AI published at https://github.com/NX-AI/xlstm.

The intention was to use a simple vanilla implementation of transformers as done by Danu Caus and replace the transformer architecture with xLSTM as a proof of concept. The transformer demo used uses a Mini Shakespeare dataset to train a model that generates language. Note that the model uses characters as tokens, not words, so it can be considered a success if the trained model generates meaningful words despite generating meaningless sentences.

The intention was to change nothing aside the actual NN architecture and layers used. In the end, this works out, with some additional changes made to include yaml-based xLSTM config.

## From transformers to xlstm

To understand the history of this repository, start by looking into the original Transformers repo by Danu Caus.

The demo notebook "mini_gpt.ipynb" from that repo was migrated into a Python module "transformer_shakespeare.py". No changes were made other than transferring the Python code.

The core implementation of xLSTM was copied into the repo; it can be found in the xlstm subfolder.

The new module "xlstm_shakespeare.py" is the merger of the xLSTM architecture into the original transformer demo. Run this to generate shakespeare language samples based on the xLSTM architecture.

## Acknowledgments

Many thanks to the following institutions that made this work possible:

- German Climate Computing Center (DKRZ), Hamburg, Germany
- Helmholtz Center Hereon, Geesthacht, Germany
- Helmholtz AI

This work was supported by Helmholtz Association's Initiative and Networking Fund through Helmholtz AI [grant number: ZT-I-PF-5-01]. 
I also used resources of the Deutsches Klimarechenzentrum (DKRZ) granted by its Scientific Steering Committee (WLA) under project ID AIM.

This work is also using the xLSTM code from NX-AI published at https://github.com/NX-AI/xlstm.