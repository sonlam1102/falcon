from fine_tune_text import *
from tqdm import tqdm
import copy
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

def make_explanation(data, processor, model):     
    ground_truth = []
    predict = []
    for d in tqdm(data):
        # if (len(d['aligment']) < 1 or len(d['aligment']) > 5):
        #     continue
        
        if len(d['aligment']) < 1:
            continue
        
        try:
            response = inference_model(d, processor, model)
        except Exception as e:
            print(e)
            response = ""
        
        d['predict_ruling'] = response
        ground_truth.append(d['ruling'])
        predict.append(response)
    
    return predict, ground_truth, data


def compute_bleu(grounds, preds):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'
        score += sentence_bleu([word_tokenize(g_t[i])],
                               word_tokenize(p_t[i]),
                               smoothing_function=SmoothingFunction().method3
                               )

    score /= len(grounds)
    return score


def compute_rouge(grounds, preds):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'

        temp_score = scorer.score(g_t[i], p_t[i])
        precision, recall, fmeasure = temp_score['rougeL']
        score = score + fmeasure

    score /= len(grounds)
    return score


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
    # processor, model = load_peft_model("./model/explanation_llms/Qwen3-4B-Instruct-2507/", flash_attention=True)
    # processor, model = load_peft_model("Qwen/Qwen3-30B-A3B-Instruct-2507", flash_attention=True, quantize=True)
    
    # processor, model = load_peft_model("Qwen/Qwen2.5-32B-Instruct", flash_attention=True, quantize=True)
    
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
    
    p, g, mocheg = make_explanation(mocheg_test, processor, model)
    with open('./results/explanation/mocheg-gpt_oss_20b.json', 'w', encoding='utf-8') as f:
        json.dump(mocheg, f, ensure_ascii=False, indent=4)
    f.close()
    
    # print("ROUGE-L: {}".format(compute_rouge(g, p)))
    # print("BLEU: {}".format(compute_bleu(g, p)))
    
    print("---------------")
    
    p, g, finfact = make_explanation(finfact_test, processor, model)
    with open('./results/explanation/finfact-gpt_oss_20b.json', 'w', encoding='utf-8') as f:
        json.dump(finfact, f, ensure_ascii=False, indent=4)
    f.close()
    # print("ROUGE-L: {}".format(compute_rouge(g, p)))
    # print("BLEU: {}".format(compute_bleu(g, p)))
