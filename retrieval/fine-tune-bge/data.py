import json

if __name__ == '__main__':
    
    with open('./data/train_image_candidates.jsonl', "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    f.close()
    
    new_img_data = []
    for d in data:
        if len(d['positive_key']) == 1 and d['positive_key'][0] == "":
            continue
        new_img_data.append(d)
    
    with open('./data/train_image_candidates_new.jsonl', 'w', encoding="utf-8") as outfile:
        for entry in new_img_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')
    outfile.close()

    print("--Done--")
    
    
    # with open('./data/train_text_candidates.jsonl', "r", encoding="utf-8") as f:
    #     data = [json.loads(line) for line in f]
    # f.close()
    
    # print(data[0:2])
    
    # print("--Done--")
    
    pass 