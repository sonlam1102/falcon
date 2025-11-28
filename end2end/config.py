BASE_PATH = "/home/falcon/"
BASE_DATA_PATH = "/home/drive/data"

AUGMENTATION_MODEL = "sonlam1102/falcon_augmentation_vision"
VERIFICATION_MODEL = "sonlam1102/falcon_verify_vision"
VERIFICATION_MODEL_TEXT = "sonlam1102/falcon_verify_text"
EXPLANATION_MODEL = "sonlam1102/falcon_explanation_vision"
EXPLANATION_MODEL_TEXT = "sonlam1102/falcon_explanation_text"
RETRIVAL_MODEL_EMBEDDING = BASE_PATH + "/model/encode_embedding/"

SUMMARY_MODEL = "sonlam1102/falcon_summary_evidence"

BGE_MODEL = "BAAI/bge-m3"
CLIP_MODEL = "ViT-L/14@336px"

HF_TOKEN = ""

EVIDENCE_DB = BASE_DATA_PATH + "/data/mocheg"
IMAGE_DB = BASE_DATA_PATH + "/data/mocheg/images"

FLASH_ATTENTION = True
QUANTIZE = True
USE_VLLM = False