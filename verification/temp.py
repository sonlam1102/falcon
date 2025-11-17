import json

def read_augmented_data(data):
    for d in data:
        if len(d['alignment']) > 0:
            for a in d['alignment']:
                temp = a['alignment'].split("\n\n\n\n")[-1]
                a['clean_alignment'] = temp.replace('<|eot_id|>', '').strip()
    return data


if __name__ == '__main__':
    # DEV
    with open("./data/mocheg_claim_llama3.2_dev_new.json", "r") as f:
        dev_org = json.load(f)
    f.close()
    
    # with open("./data/mocheg_claim_llama3.2_dev_system_sum.json", "r") as f:
    #     dev_sum = json.load(f)
    # f.close()
    
    # dev = dev_org + read_augmented_data(dev_sum)
    
    with open('./data/train_augmentation_refined.json', 'w', encoding='utf-8') as f:
        json.dump(dev_org, f, ensure_ascii=False, indent=4)
    f.close()
    
    # TEST 
    with open("./data/mocheg_claim_llama3.2_test_new.json", "r") as f:
        test_org = json.load(f)
    f.close()
    
    # with open("./data/mocheg_claim_llama3.2_test_system_sum.json", "r") as f:
    #     test_sum = json.load(f)
    # f.close()
    
    # test = test_org + read_augmented_data(test_sum)
    
    with open('./data/dev_augmentation_refined.json', 'w', encoding='utf-8') as f:
        json.dump(test_org, f, ensure_ascii=False, indent=4)
    f.close()
    
    # full = dev + test
    full = dev_org + test_org
    with open('./data/full_augmentation_refined.json', 'w', encoding='utf-8') as f:
        json.dump(full, f, ensure_ascii=False, indent=4)
    f.close()

    print(len(dev_org))
    print(len(test_org))
    print(len(full))
