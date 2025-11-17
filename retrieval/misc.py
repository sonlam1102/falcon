from FlagEmbedding import BGEM3FlagModel
import clip
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import heapq
from PIL import Image

def get_text_evidences_db(path):
    text_corpus_path = path + '/Corpus3.csv'
    text_evidence = pd.read_csv(text_corpus_path, low_memory=False)
    return text_evidence


def get_top_k(predict_output, top_k):
    lst_out_one_hot = np.zeros(len(predict_output))
    idx_top_k = heapq.nlargest(top_k, range(len(predict_output)), predict_output.take)

    i = 1
    for idx in idx_top_k:
        lst_out_one_hot[idx] = i
        i = i + 1

    return lst_out_one_hot

class MultimodalRetriever():
    def __init__(self, device):
        self._image_emb = None
        self._text_emb = None
        self._bm25 = None

        self._image_ids = None
        self._text_ids = None

        self._device = device

        self._bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device=device)
        self._visualize_clip_model, _ = clip.load("ViT-L/14@336px", device=device)
    
    
    def build_flag_embedding(self, emb_path):
        self._image_ids = np.load(emb_path + "/image_embedding_db_clip_id.npy")
        self._image_emb = torch.from_numpy(np.load(emb_path + "/image_embedding_db_clip.npy")).to(self._device)
        
        self._text_ids = np.load(emb_path + "/text_embedding_db_bge_id.npy")
        self._text_emb = torch.from_numpy(np.load(emb_path + "/text_embedding_db_bge.npy")).to(self._device)
        
        print(self._text_emb.shape)
        print(self._image_emb.shape)

        print("----Loaded evidence DB-----------")

    def get_evidence_db_ids(self):
        return self._text_ids, self._image_ids
    
    def set_evidence_db_ids(self, text_ids, image_ids):
        self._text_ids = text_ids
        self._image_ids = image_ids
    
    # BGE model
    def retrieve_text_similarity(self, query):
        claim_encode_text = self._bge_model.encode(query)['dense_vecs']
        claim_encode_text = np.expand_dims(claim_encode_text, axis=0)
        text_sim = F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self._device), self._text_emb)
        text_sim = text_sim.detach().cpu().numpy()

        return text_sim
    
    # CLIP model
    def retrieve_image_similarity_clip(self, query):
        text = clip.tokenize([query], truncate=True).to(self._device)
        claim_encode_text = self._visualize_clip_model.encode_text(text)
        image_sim = F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self._device), self._image_emb)
        image_sim = image_sim.detach().cpu().numpy()
        return image_sim
    
    def retrieve_evidence(self, query):
        lst_doc = get_top_k(self.retrieve_text_similarity(query), top_k=5)
        lst_image = get_top_k(self.retrieve_image_similarity_clip(query), top_k=5)
        
        lst_doc_ids = []
        lst_image_ids = []
        
        assert len(lst_doc) == len(self._text_ids)
        for i in range(0, len(lst_doc)):
            if lst_doc[i] > 0:
                lst_doc_ids.append(int(self._text_ids[i]))

        assert len(lst_image) == len(self._image_ids)
        for i in range(0, len(lst_image)):
            if lst_image[i] > 0:
                lst_image_ids.append(str(self._image_ids[i]))
        
        # print(lst_doc_ids)
        # print(lst_image_ids)
        
        return lst_doc_ids, lst_image_ids
        

def get_full_doc(lst_doc, text_evidence_db):
    doc_list = []
    for d_id in lst_doc:
        document = text_evidence_db.loc[text_evidence_db.relevant_document_id == d_id]['Origin Document'].values.tolist()[0]
        doc_list.append(document)
        
    return doc_list


def get_full_image(lst_image, image_db_path):
    image_list = []
    for i_id in lst_image:
        image = Image.open(image_db_path + "/" + i_id)
        image_list.append(image)
    
    return image_list


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_evidence_db = get_text_evidences_db("/home/sonlt/drive/data/mocheg")
    image_db_path = "/home/sonlt/drive/data/mocheg/images"
    
    retriever = MultimodalRetriever(device)
    retriever.build_flag_embedding("./model/encode_embedding")
    
    lst_rel_doc, lst_rel_img = retriever.retrieve_evidence(query="Photographs shared widely in September 2019 showed Greta Thunberg posing with George Soros and a member of Isis and 'aligning herself' with Antifa.")
    
    lst_doc = get_full_doc(lst_rel_doc, text_evidence_db)
    lst_image = get_full_image(lst_rel_img, image_db_path)
    
    print(lst_image)