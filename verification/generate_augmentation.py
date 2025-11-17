from finetune import *
from tqdm import tqdm


# IMG_PATH = "/home/sonlt/drive/data/"
# # IMG_PATH = "/home/s2320014/data/"
# def convert_to_conversation_infer(image_path, text):
#     system_message = """
#         Please generate a short paragraph describing the about the consistency of the image based on the given text following this template:
#         \n
#         <HYPOTHESIS>: Please determining whether the image is consistent with the text or not.
#         <EXPLANATION>: Explanation the aligment between the image hypothesis and the text.
#         <FINAL ANSWER>: Give one paragraph describing the consistency of the image and text based on the explanation.
#     """
    
#     conversation = [
#         {
#             "role": "system",
#             "content": [{"type": "text", "text": system_message}],
#         },
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image_path, "resized_height": 720, "resized_width": 1280},
#                 {"type": "text", "text": text},
#             ],
#         },
#     ]
#     return conversation


# def inference_model(image_path, text, processor, model):
#     prompt = convert_to_conversation_infer(image_path, text)
    
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
    
#     return output_text[0]
    
def make_explaination(data, processor, model):
    def make_image_description_evidence(image_explaination):
        expl = image_explaination.split("<EXPLANATION>")[-1]
        expl = expl.replace("<FINAL ANSWER>:", "")
        expl = expl.replace("     ", " ")
        expl = expl.replace("\n", "")
        expl = expl.replace(": ", "")

        hyp = image_explaination.split("<EXPLANATION>")[0]
        hyp = hyp.replace("\n", "")
        hyp = hyp.replace("<HYPOTHESIS>:", "")
        return expl, hyp
    
    def make_full_img_path(path):
        if "/mocheg/" in path:
            return IMG_PATH + "mocheg/" + path.split("/mocheg/")[-1] 
        else: 
            return IMG_PATH + "finfact/" + path.split("/finfact/")[-1]
    
    data_aug = []
    for d in tqdm(data):
        alignment = []
        if len(d['text_evidence']) > 0 and len(d['image_evidence']) > 0:
            for te in d['text_evidence']:
                for im in d['image_evidence']:
                    try:
                        align_data = make_image_description_evidence(inference_model(make_full_img_path(im), te, processor, model))
                        alignment.append({
                            "image": im,
                            "text": te,
                            "alignment": align_data
                        })
                    except Exception as e:
                        print(e)
            d['aligment'] = alignment
            data_aug.append(d)
        
    return data_aug


if __name__ == "__main__":
    # fine_tune_model()
    # inference_model()
    processor, model = load_peft_model_vision("./model/augmented_llms/Qwen2.5-VL-3B-Instruct-new-2/", flash_attention=True)
    
    with open("./data/mocheg_train.json", "r") as f:
        mocheg_train = json.load(f)
    f.close()
    
    with open("./data/mocheg_dev.json", "r") as f:
        mocheg_dev = json.load(f)
    f.close()
    
    with open("./data/mocheg_test.json", "r") as f:
        mocheg_test = json.load(f)
    f.close()
    
    with open("./data/finfact_train.json", "r") as f:
        finfact_train = json.load(f)
    f.close()
    
    with open("./data/finfact_dev.json", "r") as f:
        finfact_dev = json.load(f)
    f.close()
    
    with open("./data/finfact_test.json", "r") as f:
        finfact_test = json.load(f)
    f.close()
    
    train_full = mocheg_train + finfact_train
    print(len(train_full))
    
    dev_full = mocheg_dev + finfact_dev
    print(len(dev_full))
    
    test_full = mocheg_test + finfact_test
    print(len(test_full))
    
    
    train_aug = make_explaination(train_full, processor, model)
    with open('./data/train_aug_full_expl_2.json', 'w', encoding='utf-8') as f:
        json.dump(train_aug, f, ensure_ascii=False, indent=4)
    f.close()
    
    dev_aug = make_explaination(dev_full, processor, model)
    with open('./data/dev_aug_full_expl_2.json', 'w', encoding='utf-8') as f:
        json.dump(dev_aug, f, ensure_ascii=False, indent=4)
    f.close()
    
    
    test_aug = make_explaination(test_full, processor, model)
    with open('./data/test_aug_full_expl_2.json', 'w', encoding='utf-8') as f:
        json.dump(test_aug, f, ensure_ascii=False, indent=4)
    f.close()
