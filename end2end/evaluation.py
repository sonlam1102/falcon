import pandas as pd
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import bert_score
import copy
import json
import statistics

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


def compute_rouge(grounds, preds, type="rouge1"):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    scorer = rouge_scorer.RougeScorer([type], use_stemmer=True)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'

        temp_score = scorer.score(g_t[i], p_t[i])
        precision, recall, fmeasure = temp_score[type]
        score = score + fmeasure

    score /= len(grounds)
    return score


def compute_bertscore(grounds, preds):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'

    precision, recall, fmeasure = bert_score.score(p_t, g_t, lang="en", verbose=False)
    return fmeasure.mean().item()


def compute_meteor(grounds, preds):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'
        score += meteor_score([word_tokenize(g_t[i])],
                               word_tokenize(p_t[i]))

    score /= len(grounds)
    return score


def eval_verification_stage(mocheg_llms, mocheg_llms_vl, finfact_llms, finfact_llms_vl):
    lb2indx1 = {
        "NEI": 0,
        "supported": 2,
        "refuted": 1
    }

    lb2indx2 = {
        "Not enough information": 0,
        "True": 2,
        "False": 1
    }
    print("VERIFICATION TASK")
    mocheg_gold = mocheg_llms['ground_label'].tolist()
    mocheg_gold = [lb2indx1[i] for i in mocheg_gold]
    mocheg_pred = mocheg_llms['predict_label'].tolist()
    mocheg_pred = [lb2indx2[i] for i in mocheg_pred]
    
    mocheg_gold_vl = mocheg_llms_vl['ground_label'].tolist()
    mocheg_gold_vl = [lb2indx1[i] for i in mocheg_gold_vl]
    mocheg_pred_vl = mocheg_llms_vl['predict_label'].tolist()
    mocheg_pred_vl = [lb2indx2[i] for i in mocheg_pred_vl]

    
    finfact_gold = finfact_llms['ground_label'].tolist()
    finfact_gold = [lb2indx1[i] for i in finfact_gold]
    finfact_pred = finfact_llms['predict_label'].tolist()
    finfact_pred = [lb2indx2[i] for i in finfact_pred]
    
    finfact_gold_vl = finfact_llms_vl['ground_label'].tolist()
    finfact_gold_vl = [lb2indx1[i] for i in finfact_gold_vl]
    finfact_pred_vl = finfact_llms_vl['predict_label'].tolist()
    finfact_pred_vl = [lb2indx2[i] for i in finfact_pred_vl]
    
    print("mocheg F1: ")
    print(f1_score(mocheg_gold, mocheg_pred, average="micro")*100)
    print(f1_score(mocheg_gold, mocheg_pred, average="macro")*100)
    
    print("mocheg F1 vl: ")
    print(f1_score(mocheg_gold_vl, mocheg_pred_vl, average="micro")*100)
    print(f1_score(mocheg_gold_vl, mocheg_pred_vl, average="macro")*100)
    
    
    print("finfact F1: ")
    print(f1_score(finfact_gold, finfact_pred, average="micro")*100)
    print(f1_score(finfact_gold, finfact_pred, average="macro")*100)
    
    print("finfact F1 vl: ")
    print(f1_score(finfact_gold_vl, finfact_pred_vl, average="micro")*100)
    print(f1_score(finfact_gold_vl, finfact_pred_vl, average="macro")*100)


def eval_justification(mocheg_llms, mocheg_llms_vl, finfact_llms, finfact_llms_vl):
    print("JUSTIFICATION TASK")
    mocheg_gold = mocheg_llms['ground_ruiling'].fillna('').tolist()
    mocheg_pred = mocheg_llms['predict_ruiling'].fillna('').tolist()
    
    mocheg_gold_vl = mocheg_llms_vl['ground_ruiling'].fillna('').tolist()
    mocheg_pred_vl = mocheg_llms_vl['predict_ruiling'].fillna('').tolist()

    finfact_gold = finfact_llms['ground_ruiling'].fillna('').tolist()
    finfact_pred = finfact_llms['predict_ruiling'].fillna('').tolist()

    finfact_gold_vl = finfact_llms_vl['ground_ruiling'].fillna('').tolist()
    finfact_pred_vl = finfact_llms_vl['predict_ruiling'].fillna('').tolist()
    
    print("mocheg: ")
    print("ROUGE-1: {}".format(compute_rouge(mocheg_gold, mocheg_pred, type="rouge1")*100))
    print("ROUGE-2: {}".format(compute_rouge(mocheg_gold, mocheg_pred, type="rouge2")*100))
    print("ROUGE-L: {}".format(compute_rouge(mocheg_gold, mocheg_pred, type="rougeL")*100))
    print("BLEU: {}".format(compute_bleu(mocheg_gold, mocheg_pred)*100))
    print("METEOR: {}".format(compute_meteor(mocheg_gold, mocheg_pred)*100))
    print("BERTScore: {}".format(compute_bertscore(mocheg_gold, mocheg_pred)*100))
    
    print("Num empty pred: {}".format(mocheg_pred.count("")))
    len_justify_pred = [len(d.split()) for d in mocheg_pred]
    print("Average length: {}".format(statistics.mean(len_justify_pred)))
    
    print("mocheg vl: ")
    print("ROUGE-1: {}".format(compute_rouge(mocheg_gold_vl, mocheg_pred_vl, type="rouge1")*100))
    print("ROUGE-2: {}".format(compute_rouge(mocheg_gold_vl, mocheg_pred_vl, type="rouge2")*100))
    print("ROUGE-L: {}".format(compute_rouge(mocheg_gold_vl, mocheg_pred_vl, type="rougeL")*100))
    print("BLEU: {}".format(compute_bleu(mocheg_gold_vl, mocheg_pred_vl)*100))
    print("METEOR: {}".format(compute_meteor(mocheg_gold_vl, mocheg_pred_vl)*100))
    print("BERTScore: {}".format(compute_bertscore(mocheg_gold_vl, mocheg_pred_vl)*100))
    
    print("Num empty pred: {}".format(mocheg_pred_vl.count("")))
    len_justify_pred = [len(d.split()) for d in mocheg_pred_vl]
    print("Average length: {}".format(statistics.mean(len_justify_pred)))
    
    print("finfact: ")
    print("ROUGE-1: {}".format(compute_rouge(finfact_gold, finfact_pred, type="rouge1")*100))
    print("ROUGE-2: {}".format(compute_rouge(finfact_gold, finfact_pred, type="rouge2")*100))
    print("ROUGE-L: {}".format(compute_rouge(finfact_gold, finfact_pred, type="rougeL")*100))
    print("BLEU: {}".format(compute_bleu(finfact_gold, finfact_pred)*100))
    print("METEOR: {}".format(compute_meteor(finfact_gold, finfact_pred)*100))
    print("BERTScore: {}".format(compute_bertscore(finfact_gold, finfact_pred)*100))
    
    print("Num empty pred: {}".format(finfact_pred.count("")))
    len_justify_pred = [len(d.split()) for d in finfact_pred]
    print("Average length: {}".format(statistics.mean(len_justify_pred)))
    
    print("finfact vl: ")
    print("ROUGE-1: {}".format(compute_rouge(finfact_gold_vl, finfact_pred_vl, type="rouge1")*100))
    print("ROUGE-2: {}".format(compute_rouge(finfact_gold_vl, finfact_pred_vl, type="rouge2")*100))
    print("ROUGE-L: {}".format(compute_rouge(finfact_gold_vl, finfact_pred_vl, type="rougeL")*100))
    print("BLEU: {}".format(compute_bleu(finfact_gold_vl, finfact_pred_vl)*100))
    print("METEOR: {}".format(compute_meteor(finfact_gold_vl, finfact_pred_vl)*100))
    print("BERTScore: {}".format(compute_bertscore(finfact_gold_vl, finfact_pred_vl)*100))
    
    print("Num empty pred: {}".format(finfact_pred_vl.count("")))
    len_justify_pred = [len(d.split()) for d in finfact_pred_vl]
    print("Average length: {}".format(statistics.mean(len_justify_pred)))
    

def eval_time(data):
    time_retrival = data['time_retrieval'].sum()
    time_augmentation = data['time_augmentation'].sum()
    time_verification = data['time_verification'].sum()
    time_justification = data['time_explanation'].sum()
    
    total = time_retrival + time_augmentation + time_verification + time_justification
    time_per_sentence = total / len(data)
    
    print("Retrirval: {}".format(time_retrival))
    print("Augmentation: {}".format(time_augmentation))
    print("Verification: {}".format(time_verification))
    print("Justification: {}".format(time_justification))
    
    print("Total: {}".format(total))
    print("Time per sec: {}".format(time_per_sentence))

def data_evaluation():
    with open("./data/train_data_falcon.json", "r") as f:
        dataset_train = json.load(f)
    f.close()
    
    with open("./data/dev_data_falcon.json", "r") as f:
        dataset_dev = json.load(f)
    f.close()
    
    with open("./data/test_data_falcon.json", "r") as f:
        dataset_test = json.load(f)
    f.close()
    
    dataset_train_new = [d for d in dataset_train if len(d['aligment']) > 0 and len(d['aligment']) < 26]
    dataset_dev_new = [d for d in dataset_dev if len(d['aligment']) > 0 and len(d['aligment']) < 26]
    dataset_test_new = [d for d in dataset_test if len(d['aligment']) > 0]
    
    # with open('./data/train_data_falcon.json', 'w', encoding='utf-8') as f:
    #     json.dump(dataset_train_new, f, ensure_ascii=False, indent=4)
    # f.close()
    
    # with open('./data/dev_data_falcon.json', 'w', encoding='utf-8') as f:
    #     json.dump(dataset_dev_new, f, ensure_ascii=False, indent=4)
    # f.close()
    
    # with open('./data/test_data_falcon.json', 'w', encoding='utf-8') as f:
    #     json.dump(dataset_test_new, f, ensure_ascii=False, indent=4)
    # f.close()
    
    print(len(dataset_train_new))
    print(len(dataset_dev_new))
    print(len(dataset_test_new))
    
    print("-----mocheg-----")
    mocheg_train = [d for d in dataset_train_new if '/mocheg/' in d['image_evidence'][0]]
    mocheg_dev = [d for d in dataset_dev_new if '/mocheg/' in d['image_evidence'][0]]
    mocheg_test = [d for d in dataset_test_new if '/mocheg/' in d['image_evidence'][0]]
    
    print(len(mocheg_train))
    print(len(mocheg_dev))
    print(len(mocheg_test))
    
    print("-----finfact-----")
    finfact_train = [d for d in dataset_train_new if '/finfact/' in d['image_evidence'][0]]
    finfact_dev = [d for d in dataset_dev_new if '/finfact/' in d['image_evidence'][0]]
    finfact_test = [d for d in dataset_test_new if '/finfact/' in d['image_evidence'][0]]
    
    print(len(finfact_train))
    print(len(finfact_dev))
    print(len(finfact_test))
    

if __name__ == '__main__':
    mocheg_llms = pd.read_csv("./results/result_system_mocheg_has_sum.csv")
    finfact_llms = pd.read_csv("./results/result_system_finfact_has_sum.csv")
    
    mocheg_llms_vl = pd.read_csv("./results/result_system_mocheg_has_sum_vl.csv")
    finfact_llms_vl = pd.read_csv("./results/result_system_finfact_has_sum_vl.csv")
    
    # eval_verification_stage(mocheg_llms, mocheg_llms_vl, finfact_llms, finfact_llms_vl)
    eval_justification(mocheg_llms, mocheg_llms_vl, finfact_llms, finfact_llms_vl)
    
    # print("text LLM")
    # eval_time(mocheg_llms_vl)
    # print("------")
    # eval_time(finfact_llms_vl)
    
    # data_evaluation()