# RUN THE SYSTEM 

## Data preparation.   
- Download the database from Mocheg: https://github.com/PLUM-Lab/Mocheg. Two requires compnents are: *Corpus3.csv* -- containning the text evidence, and *images/* directory containing image evidence.  
- Config the path to the evidence in the end2end/config.py file, params EVIDENCE_DB and IMAGE_DB. 
- Put the HuggingFace access token to HF_TOKEN in end2end/config.py file.  

## Enviroments:
- Transformers (newest version is better)    
- Flask.  
- Pytorch  
- BitAndBytes (for quantization).  

## Run the system   
Run the python commandlines: 
```
CUDA_VISIBLE_DEVICES=0 flask run --debug.    
```

# FINE-TUNE THE LLMS
TBA 