from .modules import *
from sklearn.metrics import f1_score
import json
from tqdm import tqdm
import copy
import pandas as pd
import time
from PIL import Image


if __name__ == "__main__":
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

    # ids_mocheg = []
    # claim_mocheg = []
    # evidence_text_mocheg = []
    # evidence_image_mocheg = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bge_model = load_bge_model()
    image_model, processor = clip.load("ViT-L/14@336px", device=device)

    # cer_text = []
    # cer_image = []
    # cer_total = []
    # claims = []

    out_mocheg = []

    cer_text_avg_mocheg = 0
    cer_image_avg_mocheg = 0
    cer_total_avg_mocheg = 0
    
    for d in tqdm(mocheg):
        # ids_mocheg.append(d['id'])
        # claim_mocheg.append(d['claim'])
        claim, lst_text_evidence, lst_image_evidence = retrieval.run(d['claim'], is_summary=True)
        
        claim_emb = bge_model.encode(claim)['dense_vecs']
        text_evidence_embedding = bge_model.encode(lst_text_evidence)['dense_vecs']

        print(claim)
        image_evidence_embedding = None
        # clip_claim_encode = clip.tokenize([claim], truncate=True).to(device).cpu().detach().numpy()

        clip_claim_encode = image_model.encode_text(clip.tokenize([claim], truncate=True).to(device)).cpu().detach().numpy()
        for img in tqdm(lst_image_evidence):
            im_process = processor(Image.open(img)).unsqueeze(0).to(device)
            image_features = image_model.encode_image(im_process)
            if image_evidence_embedding is None:
                image_evidence_embedding = image_features.cpu().detach().numpy()
            else:
                image_evidence_embedding = np.append(image_evidence_embedding, image_features.cpu().detach().numpy(), axis=0)
        
        output_image = np.dot(clip_claim_encode, image_evidence_embedding.T) / (np.linalg.norm(clip_claim_encode) * np.linalg.norm(image_evidence_embedding))
        output_image = output_image[0]
        output_text = np.dot(claim_emb, text_evidence_embedding.T) / (np.linalg.norm(claim_emb) * np.linalg.norm(text_evidence_embedding))

        sim_text_value = sum(output_text)/len(output_text)
        sim_image_value = sum(output_image)/len(output_image)
        sim_text_value = sim_text_value.item()
        sim_image_value = sim_image_value.item()

        total_sim = 0.5*sim_text_value + 0.5*sim_image_value 

        cer_text_avg_mocheg += sim_text_value
        cer_image_avg_mocheg += sim_image_value
        cer_total_avg_mocheg += total_sim

        # cer_image.append(sim_image_value)
        # cer_text.append(sim_text_value)
        # cer_total.append(total_sim)
        # claims.append(claim)
        out_mocheg.append({
            "claim": claim,
            "cer_text": sim_text_value,
            "cer_image": sim_image_value,
            "cer": total_sim
        })

    with open("bechmark_mocheg_no_sum.json", "w", encoding="utf-8") as outfile:
        json.dump(out_mocheg, outfile, indent=4, ensure_ascii=False)
    outfile.close()

    print("====MOCHEG=====")
    print("CER_text: {}".format(cer_text_avg_mocheg / len(mocheg)))
    print("CER_image: {}".format(cer_image_avg_mocheg / len(mocheg)))
    print("CER: {}".format(cer_total_avg_mocheg / len(mocheg)))
    print("==========")

    out_finfact = []

    cer_text_avg_finfact = 0
    cer_image_avg_finfact = 0
    cer_total_avg_finfact = 0

    for d in tqdm(finfact):
        # ids_mocheg.append(d['id'])
        # claim_mocheg.append(d['claim'])
        claim, lst_text_evidence, lst_image_evidence = retrieval.run(d['claim'], is_summary=True)
        
        claim_emb = bge_model.encode(claim)['dense_vecs']
        text_evidence_embedding = bge_model.encode(lst_text_evidence)['dense_vecs']

        print(claim)
        image_evidence_embedding = None
        # clip_claim_encode = clip.tokenize([claim], truncate=True).to(device).cpu().detach().numpy()

        clip_claim_encode = image_model.encode_text(clip.tokenize([claim], truncate=True).to(device)).cpu().detach().numpy()
        for img in tqdm(lst_image_evidence):
            im_process = processor(Image.open(img)).unsqueeze(0).to(device)
            image_features = image_model.encode_image(im_process)
            if image_evidence_embedding is None:
                image_evidence_embedding = image_features.cpu().detach().numpy()
            else:
                image_evidence_embedding = np.append(image_evidence_embedding, image_features.cpu().detach().numpy(), axis=0)
        
        output_image = np.dot(clip_claim_encode, image_evidence_embedding.T) / (np.linalg.norm(clip_claim_encode) * np.linalg.norm(image_evidence_embedding))
        output_image = output_image[0]
        output_text = np.dot(claim_emb, text_evidence_embedding.T) / (np.linalg.norm(claim_emb) * np.linalg.norm(text_evidence_embedding))

        sim_text_value = sum(output_text)/len(output_text)
        sim_image_value = sum(output_image)/len(output_image)
        sim_text_value = sim_text_value.item()
        sim_image_value = sim_image_value.item()
        total_sim = 0.5*sim_text_value + 0.5*sim_image_value 

        cer_text_avg_finfact += sim_text_value
        cer_image_avg_finfact += sim_image_value
        cer_total_avg_finfact += total_sim

        # cer_image.append(sim_image_value)
        # cer_text.append(sim_text_value)
        # cer_total.append(total_sim)
        # claims.append(claim)
        out_finfact.append({
            "claim": claim,
            "cer_text": sim_text_value,
            "cer_image": sim_image_value,
            "cer": total_sim
        })

    with open("bechmark_finfact_no_sum.json", "w", encoding="utf-8") as outfile:
        json.dump(out_finfact, outfile, indent=4, ensure_ascii=False)
    outfile.close()

    print("====FINFACT=====")
    print("CER_text: {}".format(cer_text_avg_finfact / len(finfact)))
    print("CER_image: {}".format(cer_image_avg_finfact / len(finfact)))
    print("CER: {}".format(cer_total_avg_finfact / len(finfact)))
    print("==========")