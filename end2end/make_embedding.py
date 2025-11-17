import argparse

import torch
from FlagEmbedding import BGEM3FlagModel
import numpy as np

from visualized_bge import Visualized_BGE

from transformers import ViltImageProcessor
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch.nn as nn
import clip
from PIL import Image

from misc import get_text_evidences_db, get_image_evidences_db_path_only, get_text_evidences_sentence_db, get_image_evidences_db
from tqdm import tqdm, trange

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default="../data")
    parser.add_argument('--n_gpu', type=int, default=None)
    args = parser.parse_args()
    return args


def encoding_text_bge(model, textdb):
    lst_text_db = textdb['Origin Document'].values.tolist()
    lst_text_id = textdb['relevant_document_id'].values.tolist()
    loaders = torch.utils.data.DataLoader(lst_text_db, batch_size=4, shuffle=False)
    embedding = None
    embedding_id = np.array(lst_text_id)
    for d in tqdm(loaders):
        if embedding is None:
            embedding = model.encode(d)['dense_vecs']
        else:
            embedding = np.append(embedding, model.encode(d)['dense_vecs'], axis=0)

    print(embedding.shape)
    print(embedding_id.shape)
    np.save("./encode_embedding/text_embedding_db_bge.npy", embedding)
    np.save("./encode_embedding/text_embedding_db_bge_id.npy", embedding_id)


def encoding_sentence(model, sentenceDB):
    lst_sentence_db = sentenceDB['paragraph'].values.tolist()
    lst_sentence_ids = sentenceDB['2903-15073-0'].values.tolist()
    loaders = torch.utils.data.DataLoader(lst_sentence_db, batch_size=4, shuffle=False)

    embedding = None
    embedding_id = np.array(lst_sentence_ids)
    for d in tqdm(loaders):
        if embedding is None:
            embedding = model.encode(d)['dense_vecs']
        else:
            embedding = np.append(embedding, model.encode(d)['dense_vecs'], axis=0)

    print(embedding.shape)
    print(embedding_id.shape)
    np.save("./encode_embedding/sentence_embedding_db_bge.npy", embedding)
    np.save("./encode_embedding/sentence_embedding_db_bge_id.npy", embedding_id)


def encoding_image_clip(image_model, processor, imagedb, device):
    list_images = []
    lst_ids = []
    for im in imagedb:
        list_images.append(im[5])
        lst_ids.append(im[4])

    embedding_id = np.array(lst_ids)

    print("--Encoding by CLIP--")
    embedding = None
    for img in tqdm(list_images):
        im_process = processor(Image.open(img)).unsqueeze(0).to(device)
        image_features = image_model.encode_image(im_process)
        if embedding is None:
            embedding = image_features.cpu().detach().numpy()
        else:
            embedding = np.append(embedding, image_features.cpu().detach().numpy(), axis=0)

    print(embedding.shape)
    print(embedding_id.shape)
    np.save("./encode_embedding/image_embedding_db_clip.npy", embedding)
    np.save("./encode_embedding/image_embedding_db_clip_id.npy", embedding_id)


if __name__ == '__main__':
    args = parser_args()
    if args.n_gpu:
        device = torch.device("cuda:{}".format(args.n_gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_PATH = args.db_path
    # Original Models
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device=device)
    image_model, processor = clip.load("ViT-L/14@336px", device=device)

    text_evidences = get_text_evidences_db(DATA_PATH)
    print(len(text_evidences))
    encoding_text_bge(model, text_evidences)

    image_evidences = get_image_evidences_db_path_only(DATA_PATH)
    print(len(image_evidences))
    encoding_image_clip(image_model, processor, image_evidences, device)