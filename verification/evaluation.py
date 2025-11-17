import json
from sklearn.metrics import f1_score

def show_results(data):
    lb2indx = {
    "Not enough information": 0,
    "True": 2,
    "False": 1
    }

    label_map = {
        'NEI': "Not enough information",
        'supported': "True",
        'refuted': "False"
    }

    truth = []
    predicts = []
    
    for d in data:
        truth.append(label_map[d['label']])
        
        if 'Not enough information' in d['predict']:
            predict = 'Not enough information'
        elif 'True' in d['predict']:
            predict = 'True'
        elif 'False' in d['predict']:
            predict = 'False'
        else:
            predict = 'True'
        
        predicts.append(predict)
    
    new_predicts = [lb2indx[i] for i in predicts]
    new_truth = [lb2indx[i] for i in truth]
    
    print("-------")
    print(len(data))
    print(f1_score(new_truth, new_predicts, average="micro"))
    print(f1_score(new_truth, new_predicts, average="macro"))
    

if __name__ == '__main__':
    print("qwen3vl_32b")
    with open("./results/verification/mocheg-qwen3vl_32b.json", "r") as f:
        mocheg = json.load(f)
    f.close()
    print("Mocheg: \n")
    show_results(mocheg)
    print("--------")
    print("FINFACT: \n")
    with open("./results/verification/finfact-qwen3vl_32b.json", "r") as f:
        finfact = json.load(f)
    f.close()
    show_results(finfact)
    