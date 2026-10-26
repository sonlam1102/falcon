from .modules import *
from sklearn.metrics import f1_score
import json
from tqdm import tqdm
import copy
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import pandas as pd
import bert_score
import time

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


MAP_LABEL_1 = {
    'False': 0,
    'NEI': 1,
    'True': 2,
    'Not enough information': 1,
    'not enough information': 1
}

MAP_LABEL_2 = {
    'refuted': 0,
    'NEI': 1,
    'supported': 2
}

def full_evidence():
    retrieval = Retrival()
    augmentation = Augmentation(vllm=USE_VLLM)
    verification = Verification(visual=False, vllm=USE_VLLM)
    explanation = Explanation(visual=False, vllm=USE_VLLM)
    
    with open("./data/mocheg_test.json", "r") as f:
        mocheg = json.load(f)
    f.close()
    mocheg = mocheg[0:100]
    with open("./data/finfact_test.json", "r") as f:
        finfact = json.load(f)
    f.close()
    finfact = finfact[0:100]
    
    print("======Mocheg====")
    ids_mocheg = []
    claim_mocheg = []
    ground_mocheg = []
    predict_mocheg = []
    ground_emocheg = []
    predict_emocheg = []
    
    for d in tqdm(mocheg):
        ids_mocheg.append(d['id'])
        claim_mocheg.append(d['claim'])
        
        claim, lst_text_evidence, lst_image_evidence = retrieval.run(d['claim'], is_summary=True)
        
        lst_augmentation = augmentation.run(d['claim'], lst_image_evidence, lst_text_evidence)[-1]
        predict_label = verification.run(d['claim'], lst_augmentation)[-1]
        predict_explanation, _ = explanation.run(d['claim'], predict_label, lst_augmentation)

        ground_mocheg.append(d['label'])
        ground_emocheg.append(d['ruling'] if d['ruling'] != "nan" else "")
        
        predict_mocheg.append(predict_label)
        predict_emocheg.append(predict_explanation)
    
    predict = [MAP_LABEL_1[i] for i in predict_mocheg]
    truth = [MAP_LABEL_2[i] for i in ground_mocheg]
    
    print("F1-Micro: {}".format(f1_score(truth, predict, average="micro")))
    print("F1-Macro: {}".format(f1_score(truth, predict, average="macro")))
    
    print("ROUGE-L: {}".format(compute_rouge(ground_mocheg, predict_emocheg)))
    print("BLEU: {}".format(compute_bleu(ground_emocheg, predict_emocheg)))
    print("BertScore: {}".format(compute_bertscore(ground_emocheg, predict_emocheg)))

    result_mocheg = pd.DataFrame({
        'id': ids_mocheg,
        'claim': claim_mocheg,
        'ground_label': ground_mocheg,
        'predict_label': predict_mocheg,
        'ground_ruiling': ground_emocheg,
        'predict_ruiling': predict_emocheg,
    })
    result_mocheg.to_csv("result_system_full_evidence_mocheg.csv", index=False)
    
    
    print("======FINFACT====")
    ids_finfact = []
    claim_finfact = []
    ground_finfact = []
    predict_finfact = []
    ground_efinfact = []
    predict_efinfact = []
    
    for d in tqdm(finfact):
        ids_finfact.append(d['id'])
        claim_finfact.append(d['claim'])
        
        claim, lst_text_evidence, lst_image_evidence = retrieval.run(d['claim'], is_summary=True)
        lst_augmentation = augmentation.run(d['claim'], lst_image_evidence, lst_text_evidence)[-1]
        predict_label = verification.run(d['claim'], lst_augmentation)[-1]
        predict_explanation, _ = explanation.run(d['claim'], predict_label, lst_augmentation)

        
        ground_finfact.append(d['label'])
        ground_efinfact.append(d['ruling'] if d['ruling'] != "nan" else "")
        
        predict_finfact.append(predict_label)
        predict_efinfact.append(predict_explanation)
    
    predict = [MAP_LABEL_1[i] for i in predict_finfact]
    truth = [MAP_LABEL_2[i] for i in ground_finfact]
    
    print("F1-Micro: {}".format(f1_score(truth, predict, average="micro")))
    print("F1-Macro: {}".format(f1_score(truth, predict, average="macro")))
    
    print("ROUGE-L: {}".format(compute_rouge(ground_efinfact, predict_efinfact)))
    print("BLEU: {}".format(compute_bleu(ground_efinfact, predict_efinfact)))
    print("BertScore: {}".format(compute_bertscore(ground_efinfact, predict_efinfact)))
    
    result_finfact = pd.DataFrame({
        'id': ids_finfact,
        'claim': claim_finfact,
        'ground_label': ground_finfact,
        'predict_label': predict_finfact,
        'ground_ruiling': ground_efinfact,
        'predict_ruiling': predict_efinfact,
    })
    result_finfact.to_csv("result_system_full_evidence_finfact.csv", index=False)


def no_evidence():
    retrieval = Retrival()
    augmentation = Augmentation(vllm=USE_VLLM)
    verification = Verification(visual=False, vllm=USE_VLLM)
    explanation = Explanation(visual=False, vllm=USE_VLLM)
    
    with open("./data/mocheg_test.json", "r") as f:
        mocheg = json.load(f)
    f.close()
    mocheg = mocheg[0:100]
    with open("./data/finfact_test.json", "r") as f:
        finfact = json.load(f)
    f.close()
    finfact = finfact[0:100]
    
    print("======Mocheg====")
    ids_mocheg = []
    claim_mocheg = []
    ground_mocheg = []
    predict_mocheg = []
    ground_emocheg = []
    predict_emocheg = []

    for d in tqdm(mocheg):
        ids_mocheg.append(d['id'])
        claim_mocheg.append(d['claim'])
        
        # claim, lst_text_evidence, lst_image_evidence = retrieval.run(d['claim'], is_summary=True)
        # lst_text_evidence = [""]
        # lst_image_evidence = [""]
        # lst_augmentation = augmentation.run(d['claim'], lst_image_evidence, lst_text_evidence)[-1]
        lst_augmentation = []
        predict_label = verification.run(d['claim'], lst_augmentation)[-1]
        predict_explanation, _ = explanation.run(d['claim'], predict_label, lst_augmentation)

        ground_mocheg.append(d['label'])
        ground_emocheg.append(d['ruling'] if d['ruling'] != "nan" else "")
        
        predict_mocheg.append(predict_label)
        predict_emocheg.append(predict_explanation)
    
    predict = [MAP_LABEL_1[i] for i in predict_mocheg]
    truth = [MAP_LABEL_2[i] for i in ground_mocheg]
    
    print("F1-Micro: {}".format(f1_score(truth, predict, average="micro")))
    print("F1-Macro: {}".format(f1_score(truth, predict, average="macro")))
    
    print("ROUGE-L: {}".format(compute_rouge(ground_mocheg, predict_emocheg)))
    print("BLEU: {}".format(compute_bleu(ground_emocheg, predict_emocheg)))
    print("BertScore: {}".format(compute_bertscore(ground_emocheg, predict_emocheg)))

    result_mocheg = pd.DataFrame({
        'id': ids_mocheg,
        'claim': claim_mocheg,
        'ground_label': ground_mocheg,
        'predict_label': predict_mocheg,
        'ground_ruiling': ground_emocheg,
        'predict_ruiling': predict_emocheg,
    })
    result_mocheg.to_csv("result_system_no_evidence_mocheg.csv", index=False)
    
    
    print("======FINFACT====")
    ids_finfact = []
    claim_finfact = []
    ground_finfact = []
    predict_finfact = []
    ground_efinfact = []
    predict_efinfact = []
    
    for d in tqdm(finfact):
        ids_finfact.append(d['id'])
        claim_finfact.append(d['claim'])
        
        # claim, lst_text_evidence, lst_image_evidence = retrieval.run(d['claim'], is_summary=True)
        # lst_text_evidence = []
        # lst_image_evidence = []
        # lst_augmentation = augmentation.run(d['claim'], lst_image_evidence, lst_text_evidence)[-1]
        lst_augmentation = []
        predict_label = verification.run(d['claim'], lst_augmentation)[-1]
        predict_explanation, _ = explanation.run(d['claim'], predict_label, lst_augmentation)

        
        ground_finfact.append(d['label'])
        ground_efinfact.append(d['ruling'] if d['ruling'] != "nan" else "")
        
        predict_finfact.append(predict_label)
        predict_efinfact.append(predict_explanation)
    
    predict = [MAP_LABEL_1[i] for i in predict_finfact]
    truth = [MAP_LABEL_2[i] for i in ground_finfact]
    
    print("F1-Micro: {}".format(f1_score(truth, predict, average="micro")))
    print("F1-Macro: {}".format(f1_score(truth, predict, average="macro")))
    
    print("ROUGE-L: {}".format(compute_rouge(ground_efinfact, predict_efinfact)))
    print("BLEU: {}".format(compute_bleu(ground_efinfact, predict_efinfact)))
    print("BertScore: {}".format(compute_bertscore(ground_efinfact, predict_efinfact)))
    
    result_finfact = pd.DataFrame({
        'id': ids_finfact,
        'claim': claim_finfact,
        'ground_label': ground_finfact,
        'predict_label': predict_finfact,
        'ground_ruiling': ground_efinfact,
        'predict_ruiling': predict_efinfact,
    })
    result_finfact.to_csv("result_system_no_evidence_finfact.csv", index=False)


def text_only_evidence():
    retrieval = Retrival()
    augmentation = Augmentation(vllm=USE_VLLM)
    verification = Verification(visual=False, vllm=USE_VLLM)
    explanation = Explanation(visual=False, vllm=USE_VLLM)
    
    with open("./data/mocheg_test.json", "r") as f:
        mocheg = json.load(f)
    f.close()
    mocheg = mocheg[0:100]
    with open("./data/finfact_test.json", "r") as f:
        finfact = json.load(f)
    f.close()
    finfact = finfact[0:100]
    
    print("======Mocheg====")
    ids_mocheg = []
    claim_mocheg = []
    ground_mocheg = []
    predict_mocheg = []
    ground_emocheg = []
    predict_emocheg = []
    
    for d in tqdm(mocheg):
        ids_mocheg.append(d['id'])
        claim_mocheg.append(d['claim'])
        
        claim, lst_text_evidence, lst_image_evidence = retrieval.run(d['claim'], is_summary=True)
        # blank image 
        lst_image_evidence = ["/home/sonlt/falcon/blank.png"]
        lst_augmentation = augmentation.run(d['claim'], lst_image_evidence, lst_text_evidence)[-1]
        predict_label = verification.run(d['claim'], lst_augmentation)[-1]
        predict_explanation, _ = explanation.run(d['claim'], predict_label, lst_augmentation)

        ground_mocheg.append(d['label'])
        ground_emocheg.append(d['ruling'] if d['ruling'] != "nan" else "")
        
        predict_mocheg.append(predict_label)
        predict_emocheg.append(predict_explanation)
    
    predict = [MAP_LABEL_1[i] for i in predict_mocheg]
    truth = [MAP_LABEL_2[i] for i in ground_mocheg]
    
    print("F1-Micro: {}".format(f1_score(truth, predict, average="micro")))
    print("F1-Macro: {}".format(f1_score(truth, predict, average="macro")))
    
    print("ROUGE-L: {}".format(compute_rouge(ground_mocheg, predict_emocheg)))
    print("BLEU: {}".format(compute_bleu(ground_emocheg, predict_emocheg)))
    print("BertScore: {}".format(compute_bertscore(ground_emocheg, predict_emocheg)))

    result_mocheg = pd.DataFrame({
        'id': ids_mocheg,
        'claim': claim_mocheg,
        'ground_label': ground_mocheg,
        'predict_label': predict_mocheg,
        'ground_ruiling': ground_emocheg,
        'predict_ruiling': predict_emocheg,
    })
    result_mocheg.to_csv("result_system_text_only_evidence_mocheg.csv", index=False)
    
    
    print("======FINFACT====")
    ids_finfact = []
    claim_finfact = []
    ground_finfact = []
    predict_finfact = []
    ground_efinfact = []
    predict_efinfact = []
    
    for d in tqdm(finfact):
        ids_finfact.append(d['id'])
        claim_finfact.append(d['claim'])
        
        claim, lst_text_evidence, lst_image_evidence = retrieval.run(d['claim'], is_summary=True)
        lst_image_evidence = ["/home/sonlt/falcon/blank.png"]
        lst_augmentation = augmentation.run(d['claim'], lst_image_evidence, lst_text_evidence)[-1]
        predict_label = verification.run(d['claim'], lst_augmentation)[-1]
        predict_explanation, _ = explanation.run(d['claim'], predict_label, lst_augmentation)
        
        ground_finfact.append(d['label'])
        ground_efinfact.append(d['ruling'] if d['ruling'] != "nan" else "")
        
        predict_finfact.append(predict_label)
        predict_efinfact.append(predict_explanation)
    
    predict = [MAP_LABEL_1[i] for i in predict_finfact]
    truth = [MAP_LABEL_2[i] for i in ground_finfact]
    
    print("F1-Micro: {}".format(f1_score(truth, predict, average="micro")))
    print("F1-Macro: {}".format(f1_score(truth, predict, average="macro")))
    
    print("ROUGE-L: {}".format(compute_rouge(ground_efinfact, predict_efinfact)))
    print("BLEU: {}".format(compute_bleu(ground_efinfact, predict_efinfact)))
    print("BertScore: {}".format(compute_bertscore(ground_efinfact, predict_efinfact)))
    
    result_finfact = pd.DataFrame({
        'id': ids_finfact,
        'claim': claim_finfact,
        'ground_label': ground_finfact,
        'predict_label': predict_finfact,
        'ground_ruiling': ground_efinfact,
        'predict_ruiling': predict_efinfact,
    })
    result_finfact.to_csv("result_system_text_only_evidence_finfact.csv", index=False)


def image_only_evidence():
    retrieval = Retrival()
    augmentation = Augmentation(vllm=USE_VLLM)
    verification = Verification(visual=True, vllm=USE_VLLM)
    explanation = Explanation(visual=True, vllm=USE_VLLM)
    
    with open("./data/mocheg_test.json", "r") as f:
        mocheg = json.load(f)
    f.close()
    mocheg = mocheg[0:100]
    with open("./data/finfact_test.json", "r") as f:
        finfact = json.load(f)
    f.close()
    finfact = finfact[0:100]
    
    print("======Mocheg====")
    ids_mocheg = []
    claim_mocheg = []
    ground_mocheg = []
    predict_mocheg = []
    ground_emocheg = []
    predict_emocheg = []
    
    for d in tqdm(mocheg):
        ids_mocheg.append(d['id'])
        claim_mocheg.append(d['claim'])
        
        claim, lst_text_evidence, lst_image_evidence = retrieval.run(d['claim'], is_summary=True)
        lst_text_evidence = [""]
        lst_augmentation = augmentation.run(d['claim'], lst_image_evidence, lst_text_evidence)[-1]
        predict_label = verification.run(d['claim'], lst_augmentation)[-1]
        predict_explanation, _ = explanation.run(d['claim'], predict_label, lst_augmentation)

        ground_mocheg.append(d['label'])
        ground_emocheg.append(d['ruling'] if d['ruling'] != "nan" else "")
        
        predict_mocheg.append(predict_label)
        predict_emocheg.append(predict_explanation)
    
    predict = [MAP_LABEL_1[i] for i in predict_mocheg]
    truth = [MAP_LABEL_2[i] for i in ground_mocheg]
    
    print("F1-Micro: {}".format(f1_score(truth, predict, average="micro")))
    print("F1-Macro: {}".format(f1_score(truth, predict, average="macro")))
    
    print("ROUGE-L: {}".format(compute_rouge(ground_mocheg, predict_emocheg)))
    print("BLEU: {}".format(compute_bleu(ground_emocheg, predict_emocheg)))
    print("BertScore: {}".format(compute_bertscore(ground_emocheg, predict_emocheg)))

    result_mocheg = pd.DataFrame({
        'id': ids_mocheg,
        'claim': claim_mocheg,
        'ground_label': ground_mocheg,
        'predict_label': predict_mocheg,
        'ground_ruiling': ground_emocheg,
        'predict_ruiling': predict_emocheg,
    })
    result_mocheg.to_csv("result_system_image_only_evidence_mocheg.csv", index=False)
    
    
    print("======FINFACT====")
    ids_finfact = []
    claim_finfact = []
    ground_finfact = []
    predict_finfact = []
    ground_efinfact = []
    predict_efinfact = []
    
    for d in tqdm(finfact):
        ids_finfact.append(d['id'])
        claim_finfact.append(d['claim'])
        
        claim, lst_text_evidence, lst_image_evidence = retrieval.run(d['claim'], is_summary=True)
        lst_text_evidence = [""]
        lst_augmentation = augmentation.run(d['claim'], lst_image_evidence, lst_text_evidence)[-1]
        predict_label = verification.run(d['claim'], lst_augmentation)[-1]
        predict_explanation, _ = explanation.run(d['claim'], predict_label, lst_augmentation)
        
        ground_finfact.append(d['label'])
        ground_efinfact.append(d['ruling'] if d['ruling'] != "nan" else "")
        
        predict_finfact.append(predict_label)
        predict_efinfact.append(predict_explanation)
    
    predict = [MAP_LABEL_1[i] for i in predict_finfact]
    truth = [MAP_LABEL_2[i] for i in ground_finfact]
    
    print("F1-Micro: {}".format(f1_score(truth, predict, average="micro")))
    print("F1-Macro: {}".format(f1_score(truth, predict, average="macro")))
    
    print("ROUGE-L: {}".format(compute_rouge(ground_efinfact, predict_efinfact)))
    print("BLEU: {}".format(compute_bleu(ground_efinfact, predict_efinfact)))
    print("BertScore: {}".format(compute_bertscore(ground_efinfact, predict_efinfact)))
    
    result_finfact = pd.DataFrame({
        'id': ids_finfact,
        'claim': claim_finfact,
        'ground_label': ground_finfact,
        'predict_label': predict_finfact,
        'ground_ruiling': ground_efinfact,
        'predict_ruiling': predict_efinfact,
    })
    result_finfact.to_csv("result_system_image_only_evidence_finfact.csv", index=False)


if __name__ == "__main__":
    full_evidence()
