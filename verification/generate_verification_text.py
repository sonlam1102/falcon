from tqdm import tqdm
from finetune_verify_text import *
from sklearn.metrics import f1_score

def make_verification(data, processor, model):
    def clean_response(res):
        res = res.replace("<think>", "")
        res = res.replace("</think>", "")
        res = res.replace("\n", "")
        
        return res
    
    label_map = {
        'NEI': "Not enough information",
        'supported': "True",
        'refuted': "False"
    }
        
    ground_truth = []
    predict = []
    for d in tqdm(data):
        # if (len(d['aligment']) < 1 or len(d['aligment']) > 5):
        #     continue
        if (len(d['aligment'])) < 1:
            continue
        
        try:
            response = inference_model(d, processor, model)
            response = response.replace("assistant\n", "")
            response = response.replace("assistant**", "")
            clean_label = clean_response(response)
        except Exception as e:
            print(e)
            clean_label = "True"
        d['predict'] = clean_label
        
        predict.append(clean_label)
        ground_truth.append(label_map[d['label']])
    
    return predict, ground_truth, data


def process_test_data(data):
    mocheg = []
    finfact = []
    
    for d in data:
        if (len(d['aligment'])) < 1:
            continue
        
        if '/finfact/' in d['image_evidence'][0]:
            finfact.append(d)
        else:
            mocheg.append(d)
    
    return mocheg, finfact


if __name__ == "__main__":
    # processor, model = load_peft_model("./model/verification/Qwen3-4B-Instruct-2507-new-2/", flash_attention=True)
    # processor, model = load_peft_model("/data/huggingface_models/Qwen3-4B-Instruct-2507", flash_attention=True)
    # processor, model = load_peft_model("/data/huggingface_models/Qwen3-30B-A3B-Instruct-2507", flash_attention=True, quantize=True)
    # processor, model = load_peft_model("/data/huggingface_models/Qwen2.5-14B-Instruct", flash_attention=True)
    processor, model = load_peft_model("/data/huggingface_models/gpt-oss-20b", flash_attention=True, quantize=True)
    
    with open("./data/test_aug_full_expl_2.json", "r") as f:
        test = json.load(f)
    f.close()
    
    mocheg_test, finfact_test = process_test_data(test)
    print("Mocheg: {}".format(len(mocheg_test)))
    print("FinFact: {}".format(len(finfact_test)))
    
    lb2indx = {
        "Not enough information": 0,
        "True": 2,
        "False": 1
    }
    
    p, g, mocheg = make_verification(mocheg_test, processor, model)
    # # predict = [lb2indx[i] for i in p]
    # predict = []
    # for p_sample in p:
    #     try:
    #         predict.append(lb2indx[p_sample])
    #     except Exception as e:
    #         print(e)
    #         predict.append(2)

    # truth = [lb2indx[i] for i in g]
    
    with open('./results/verification/mocheg-gpt_oss_20.json', 'w', encoding='utf-8') as f:
        json.dump(mocheg, f, ensure_ascii=False, indent=4)
    f.close()
    
    # print(len(predict))
    # print(f1_score(truth, predict, average="micro"))
    # print(f1_score(truth, predict, average="macro"))
    
    
    print("-----------------")
    p, g, finfact = make_verification(finfact_test, processor, model)
    
    with open('./results/verification/finfact-gpt_oss_20.json', 'w', encoding='utf-8') as f:
        json.dump(finfact, f, ensure_ascii=False, indent=4)
    f.close()
    
    # # predict = [lb2indx[i] for i in p]
    # predict = []
    # for p_sample in p:
    #     try:
    #         predict.append(lb2indx[p_sample])
    #     except Exception as e:
    #         print(e)
    #         predict.append(2)
    
    # truth = [lb2indx[i] for i in g]
    
    # print(len(predict))
    # print(f1_score(truth, predict, average="micro"))
    # print(f1_score(truth, predict, average="macro"))
    
    