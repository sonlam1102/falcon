from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import bert_score
import copy
import json

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


def make_eval(data):
    truth = []
    predicts = []
    for d in data:
        truth.append(d['ruling'] if d['ruling'] != "nan" else "")
        new_ruiling = d['predict_ruling'].replace("assistant\n", "")
        new_ruiling = new_ruiling.replace("usercontent\n", "")
        new_ruiling = new_ruiling.replace("\n", "")
        predicts.append(new_ruiling)
    
    print("ROUGE-1: {}".format(compute_rouge(truth, predicts, type="rouge1")*100))
    print("ROUGE-2: {}".format(compute_rouge(truth, predicts, type="rouge2")*100))
    print("ROUGE-L: {}".format(compute_rouge(truth, predicts, type="rougeL")*100))
    print("BLEU: {}".format(compute_bleu(truth, predicts)*100))
    print("METEOR: {}".format(compute_meteor(truth, predicts)*100))
    print("BERTScore: {}".format(compute_bertscore(truth, predicts)*100))

if __name__ == '__main__':
    print("qwen3vl_32b")
    with open("./results/explanation/mocheg-qwen3vl_32b.json", "r") as f:
        mocheg = json.load(f)
    f.close()
    print("Mocheg: \n")
    make_eval(mocheg)
    print("--------")
    print("FINFACT: \n")
    with open("./results/explanation/finfact-qwen3vl_32b.json", "r") as f:
        finfact = json.load(f)
    f.close()
    make_eval(finfact)
