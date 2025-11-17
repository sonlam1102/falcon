from tqdm import tqdm
import json

# from unsloth import FastLanguageModel, FastModel
import torch
# from accelerate import Accelerator
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from PIL import Image

def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    # print(examples)
    # raise Exception
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs
    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    # if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
    #     image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    # else:
    #     image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch


IMG_PATH = "/home/jnlp/sonlt/drive/data/"
def load_peft_model_vision(peft_model_name, device="auto", flash_attention=True, quantize=False):
    processor = AutoProcessor.from_pretrained(
        peft_model_name,
        model_max_length=8192,
        padding_side="left",
        truncation_side="left",
        token="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",
        trust_remote_code=True
    )
    
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    atten_type = "flash_attention_2" if flash_attention else "eager" 
    if quantize:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            peft_model_name,
            token="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",
            device_map=device,
            attn_implementation=atten_type,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        peft_model_name,
        token="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",
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation=atten_type,
    )

    return processor, model


def load_peft_model_vision_qwen3(peft_model_name, device="auto", flash_attention=True, quantize=False):
    processor = AutoProcessor.from_pretrained(
        peft_model_name,
        model_max_length=8192,
        padding_side="left",
        truncation_side="left",
        token="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",
        trust_remote_code=True
    )
    
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    atten_type = "flash_attention_2" if flash_attention else "eager" 
    if quantize:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            peft_model_name,
            token="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",
            device_map=device,
            attn_implementation=atten_type,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
        peft_model_name,
        token="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",
        device_map=device,
        torch_dtype=torch.bfloat16,
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
       Belows are list of evidence containing the image, the text evidence, and a sentence describe the consistency between text and image:
    """
    system_message2 = f"""
        Base on given evindece belows, let's determine the truthfulness of the claim as True, False or Not enough information. Response only one of three values: True, False, or Not enough information.
    """
    
    multimodal_content = []
    for s in sample['aligment']:
        if '/finfact/' in s['image']:
            new_path = IMG_PATH+"finfact/"+s['image'].split("/finfact/")[-1]
        else:
            new_path = IMG_PATH+"mocheg/"+s['image'].split("/mocheg/")[-1]
        
        multimodal_content.append({
            "type": "image",
            "image": new_path,
            "resized_height": 480, 
            "resized_width": 720
        }) 
        
        multimodal_content.append({
            "type": "text",
            "text": "TEXT EVIDENCE: " + s['text'],
        }) 
        
        multimodal_content.append({
            "type": "text",
            "text": "CONSISTENCY OF TEXT and IMAGE: " + s['alignment'][0] + s['alignment'][1].replace("assistant ", " In conclusion, "),
        })
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": multimodal_content,
        },
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message2}],
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
       Belows are list of evidence containing the image, the text evidence, and a sentence describe the consistency between text and image:
    """
    system_message2 = f"""
        Base on given evindece belows, let's determine the truthfulness of the claim as True, False or Not enough information. Response only one of three values: True, False, or Not enough information.
    """
    
    multimodal_content = []
    for s in sample['aligment']:
        if '/finfact/' in s['image']:
            new_path = IMG_PATH+"finfact/"+s['image'].split("/finfact/")[-1]
        else:
            new_path = IMG_PATH+"mocheg/"+s['image'].split("/mocheg/")[-1]
        
        multimodal_content.append({
            "type": "image",
            "image": new_path,
            "resized_height": 480, 
            "resized_width": 720
        }) 
        
        multimodal_content.append({
            "type": "text",
            "text": "TEXT EVIDENCE: " + s['text'],
        }) 
        
        multimodal_content.append({
            "type": "text",
            "text": "CONSISTENCY OF TEXT and IMAGE: " + s['alignment'][0] + s['alignment'][1].replace("assistant ", " In conclusion, "),
        })
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": multimodal_content,
        },
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message2}],
        },
    ]
    return conversation


def fine_tune_model():
    ## FINE-TUNE model
    # load model
    global processor, model
    processor, model = load_peft_model_vision("/data/huggingface_models/Qwen2.5-VL-3B-Instruct", flash_attention=True)

    with open("./data/train_aug_full_expl_2.json", "r") as f:
        dataset_train = json.load(f)
    f.close()
    
    with open("./data/dev_aug_full_expl_2.json", "r") as f:
        dataset_dev = json.load(f)
    f.close()
    
    dataset_train_new = [convert_to_conversation(d) for d in tqdm(dataset_train) if len(d['aligment']) > 0 and len(d['aligment']) < 6]
    dataset_dev_new = [convert_to_conversation(d) for d in tqdm(dataset_dev) if len(d['aligment']) and len(d['aligment']) < 6]

    print("Loaded {} train prompts".format(len(dataset_train_new)))
    print("Loaded {} dev prompts".format(len(dataset_dev_new)))

    print(dataset_train_new[0])
    
    training_args = SFTConfig(
        output_dir="./model/verification/Qwen2.5-VL-3B-Instruct",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./model/verification/Qwen2.5-VL-3B-Instruct",
        logging_steps=10000,
        packing=False,
        save_total_limit=3,
        dataset_kwargs={'skip_prepare_dataset': True},
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        eval_steps=500,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        # save_steps=100000,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Gradient checkpointing settings
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
        eval_dataset=dataset_dev_new,
        # peft_config=peft_config,
        # formatting_func=formatting_prompts_func,
        tokenizer=processor.tokenizer,
    )

    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    # Step 6: Save the model
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    


# def inference_model():
#     # processor, model = load_peft_model_vision("/data/huggingface_models/Qwen2.5-VL-3B-Instruct", flash_attention=True)
#     processor, model = load_peft_model_vision("./model/verification/Qwen2.5-VL-3B-Instruct/", flash_attention=True)
#     # model.load_adapter("./model/augmented_llms/Qwen2.5-VL-3B-Instruct-new/")
    
#     with open("./data/test_aug_full_expl_2.json", "r") as f:
#         dataset = json.load(f)  
#     f.close()
    
#     example = dataset[0]['alignment'][0]
#     prompt = convert_to_conversation_infer(example)
    
#     # print(prompt)
#     text_inputs = [processor.apply_chat_template(prompt, tokenize=False)]
#     image_inputs = [process_vision_info(prompt)[0]]
    
#     model_inputs = processor(
#         text=text_inputs, images=image_inputs, return_tensors="pt", padding=True
#     ).to("cuda")
    
#     output_ids = model.generate(**model_inputs, max_new_tokens=2048)
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_ids)
#     ]
    
#     output_text = processor.batch_decode(
#         generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     print(output_text)
    
#     return output_text[0]


def inference_model(data, processor, model):
    prompt = convert_to_conversation_infer(data)
    
    # print(prompt)
    text_inputs = [processor.apply_chat_template(prompt, tokenize=False)]
    image_inputs = [process_vision_info(prompt)[0]]
    
    model_inputs = processor(
        text=text_inputs, images=image_inputs, return_tensors="pt", padding=True
    ).to("cuda")
    
    output_ids = model.generate(**model_inputs, max_new_tokens=10)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


if __name__ == "__main__":
    fine_tune_model()
