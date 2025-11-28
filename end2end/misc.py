import torch
import numpy as np
import heapq
import clip
import pandas as pd

from FlagEmbedding import BGEM3FlagModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BartForConditionalGeneration, BartTokenizer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from vllm import LLM, SamplingParams
from .config import *

def load_peft_model_vision(peft_model_name, device="auto", flash_attention=True, token="", quantize=False):
    processor = AutoProcessor.from_pretrained(
        peft_model_name,
        model_max_length=8000,
        padding_side="left",
        truncation_side="left",
        trust_remote_code=True,
        token=token
    )
    
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    atten_type = "flash_attention_2" if flash_attention else "eager" 
    if quantize:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            peft_model_name,
            device_map=device,
            attn_implementation=atten_type,
            quantization_config=quantization_config, 
            torch_dtype=torch.bfloat16,
            token=token
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        peft_model_name,
        device_map=device,
        attn_implementation=atten_type,
        torch_dtype=torch.bfloat16,
        token=token
    )

    return processor, model


# def load_peft_model_vision_with_vllm(peft_model_name, token="", quantize=False):
#     processor = AutoProcessor.from_pretrained(
#         peft_model_name,
#         model_max_length=8192,
#         padding_side="left",
#         truncation_side="left",
#         trust_remote_code=True
#     )
    
#     if quantize:
#         model = LLM(model=peft_model_name, dtype=torch.float16, quantization="bitsandbytes", load_format="bitsandbytes")
#     else:
#         model = LLM(model=peft_model_name)
#     return processor, model


# def load_peft_model_with_vllm(peft_model_name, token="", quantize=False):
#     processor = AutoTokenizer.from_pretrained(peft_model_name)

#     if quantize:
#         model = LLM(model=peft_model_name, dtype=torch.float16, quantization="bitsandbytes", load_format="bitsandbytes")
#     else:
#         model = LLM(model=peft_model_name)
#     return processor, model


def load_peft_model(peft_model_name, flash_attention=True, token="", quantize=False):
    processor = AutoTokenizer.from_pretrained(peft_model_name, token=token)

    atten_type = "flash_attention_2" if flash_attention else "eager" 
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    
    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
            peft_model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=atten_type,
            quantization_config=quantization_config, 
            token=token
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            peft_model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=atten_type,
            token=token
        )

    return processor, model


def get_top_k(predict_output, top_k):
    lst_out_one_hot = np.zeros(len(predict_output))
    idx_top_k = heapq.nlargest(top_k, range(len(predict_output)), predict_output.take)

    i = 1
    for idx in idx_top_k:
        lst_out_one_hot[idx] = i
        i = i + 1

    return lst_out_one_hot


def load_summary_model():
    model = BartForConditionalGeneration.from_pretrained(SUMMARY_MODEL, use_auth_token=HF_TOKEN)
    tokenizer = BartTokenizer.from_pretrained(SUMMARY_MODEL, use_auth_token=HF_TOKEN)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return tokenizer, model.to(device)

def load_bge_model():
    model = BGEM3FlagModel(BGE_MODEL, use_fp16=False, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model

def load_clip_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_clip_model, _ = clip.load(CLIP_MODEL, device=device)
    return visualize_clip_model, device


def get_full_doc(lst_doc, text_evidence_db):
    doc_list = []
    for d_id in lst_doc:
        document = text_evidence_db.loc[text_evidence_db.relevant_document_id == d_id]['Origin Document'].values.tolist()[0]
        doc_list.append(document)
        
    return doc_list


def get_full_image(lst_image, image_db_path, return_image=True):
    image_list = []
    for i_id in lst_image:
        if return_image:
            image = Image.open(image_db_path + "/" + i_id)
            image_list.append(image)
        else:
            image_list.append(image_db_path + "/" + i_id)
    
    return image_list

def get_text_evidences_db(path):
    text_corpus_path = path + '/Corpus3.csv'
    text_evidence = pd.read_csv(text_corpus_path, low_memory=False)
    return text_evidence

