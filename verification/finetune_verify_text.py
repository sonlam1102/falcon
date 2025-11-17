from tqdm import tqdm
import json

# from unsloth import FastLanguageModel, FastModel
import torch
# from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig, Mxfp4Config
from qwen_vl_utils import process_vision_info

# from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from PIL import Image

def collate_fn(examples):
    texts = [
        processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True, enable_thinking=False) for example in examples
    ]  
    batch = processor(
        text=texts, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.pad_token_id] = -100  # Mask padding tokens in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch


def load_peft_model(peft_model_name, flash_attention=True, quantize=False):
    processor = AutoTokenizer.from_pretrained(peft_model_name)
    
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True, 
    #     bnb_4bit_use_double_quant=True, 
    #     bnb_4bit_quant_type="nf4", 
    #     bnb_4bit_compute_dtype=torch.float16
    # )
    quantization_config = Mxfp4Config()

    atten_type = "flash_attention_2" if flash_attention else "eager" 
    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
            peft_model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=atten_type,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            peft_model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=atten_type,
        )

    return processor, model


def convert_to_conversation(sample):
    label_map = {
        'NEI': "Not enough information",
        'supported': "True",
        'refuted': "False"
    }
    
    system_message = f"""
       You are an assistant that help verifying the veracity of a claim based on the multimodal evidence including text and image.
       The claim is: {sample['claim']}
       Belows are list of evidence containing the text evidence and a sentence describe the consistency between text and image:
    """
    system_message2 = f"""
        Base on given evindece belows, let's determine the truthfulness of the claim as True, False or Not enough information. Response only one of three values: True, False, or Not enough information.
    """
    
    # multimodal_content = []
    multimodal_content = ""
    for s in sample['aligment']:
        multimodal_content += "TEXT EVIDENCE: " + s['text'] + "\n" + "CONSISTENCY OF TEXT and IMAGE: " + s['alignment'][0] + s['alignment'][1].replace("assistant ", " In conclusion, ") + "\n"
    
    conversation = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": multimodal_content,
        },
        {
            "role": "system",
            "content": system_message2,
        },
        {
            "role": "assistant",
            "content": label_map[sample['label']],
        },
    ]
    return conversation


def convert_to_conversation_infer(sample): 
    system_message = f"""
       You are an assistant that help verifying the veracity of a claim based on the multimodal evidence including text and image.
       The claim is: {sample['claim']}
       Belows are list of evidence containing the text evidence and a sentence describe the consistency between text and image:
    """
    system_message2 = f"""
        Base on given evindece belows, let's determine the truthfulness of the claim as True, False or Not enough information. Response only one of three values: True, False, or Not enough information.
    """
    
    # multimodal_content = []
    multimodal_content = ""
    for s in sample['aligment']:
        multimodal_content += "TEXT EVIDENCE: " + s['text'] + "\n" + "CONSISTENCY OF TEXT and IMAGE: " + s['alignment'][0] + s['alignment'][1].replace("assistant ", " In conclusion, ") + "\n"
    
    conversation = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": multimodal_content,
        },
        {
            "role": "system",
            "content": system_message2,
        },
    ]
    return conversation


def fine_tune_model():
    global processor, model
    processor, model = load_peft_model("/SSD_data1/huggingface_models/Qwen3-4B-Instruct-2507", flash_attention=True)

    with open("./data/train_aug_full_expl_2.json", "r") as f:
        dataset_train = json.load(f)
    f.close()
    
    with open("./data/dev_aug_full_expl_2.json", "r") as f:
        dataset_dev = json.load(f)
    f.close()

    dataset_train_new = [convert_to_conversation(d) for d in tqdm(dataset_train) if len(d['aligment']) > 0 and len(d['aligment']) < 26]
    dataset_dev_new = [convert_to_conversation(d) for d in tqdm(dataset_dev) if len(d['aligment']) > 0 and len(d['aligment']) < 26]
    
    # dataset_train_new = [convert_to_conversation(d) for d in tqdm(dataset_train) if len(d['aligment']) > 0]
    # dataset_dev_new = [convert_to_conversation(d) for d in tqdm(dataset_dev) if len(d['aligment']) > 0]

    print("Loaded {} train prompts".format(len(dataset_train_new)))
    print("Loaded {} dev prompts".format(len(dataset_dev_new)))
    
    print(dataset_train_new[0])
    
    training_args = SFTConfig(
        output_dir="./model/verification/Qwen3-4B-Instruct-2507",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./model/verification/Qwen3-4B-Instruct-2507",
        logging_steps=1000,
        packing=False,
        save_total_limit=3,
        dataset_kwargs={'skip_prepare_dataset': True},
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        # eval_steps=500,  # Steps interval for evaluation
        # eval_strategy="steps",  # Strategy for evaluation
        # save_strategy="steps",  # Strategy for saving the model
        do_eval=False,  # Steps interval for evaluation
        eval_strategy="no",  # Strategy for evaluation
        save_steps=1000,  # Steps interval for saving
        # metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        # greater_is_better=False,  # Whether higher metric values are better
        # load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Gradient checkpointing settings
        gradient_accumulation_steps=4,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    )
    # training_args.remove_unused_columns = False  # Keep unused columns in dataset
    # peft_model = get_peft_model(model, peft_config)
    # peft_model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset_train_new,
        # eval_dataset=dataset_dev_new,
        # peft_config=peft_config,
        # formatting_func=formatting_prompts_func,
        tokenizer=processor,
    )

    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    # Step 6: Save the model
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    

def inference_model(data, processor, model):
    prompt = convert_to_conversation_infer(data)
    
    # print(prompt)
    text_inputs = [processor.apply_chat_template(prompt, tokenize=False, enable_thinking=False)]
    
    model_inputs = processor(
        text=text_inputs, return_tensors="pt", padding=True
    ).to("cuda")
    
    output_ids = model.generate(**model_inputs, max_new_tokens=10, pad_token_id=processor.eos_token_id)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


if __name__ == "__main__":
    fine_tune_model()
