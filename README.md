# Unifying Vision, Text, and Layout for Universal Document Processing
## Finetuning UDOP model for recommendation and modification of PowerPoint template

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)


## Project Overview![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)
---

This repository contains the source code for 2023 MIRIDIH Corporate Collaboration Project. This project utilises UDOP as baseline model to recommend and further modify a PPT template based on user's query.

This model supports korean query by using ke-t5 tokenizer.
ke-t5 Tokenizer Source : https://github.com/AIRC-KETI/ke-t5

## Install
---
### Setup `python` environment
```
conda create -n UDOP python=3.8   # You can also use other environment.
```
### Install other dependencies
```
pip install -r requirements.txt
```

## Repository Structure
---
``` bash

├── LICENSE
├── README.md
├── config/                         # Train/Inference configuration files
│   ├── inference.yaml
│   └── config.yaml
├── core/                           # Main source code
│   ├── common/
│   ├── datasets/
│   ├── models/
│   └── trainers/
├── main.py                         # Source code for inference 
├── requirements.txt
├── run.py
└── savevector.py                   # vector embedding source code for recommendataion system
```

## Scripts
Setup folder structures as above and modify config/ yaml files for customization

### Finetune UDOP model
```
python main.py config/config.yaml
```

### Inference UDOP model
```
python main.py config/inference.yaml
```