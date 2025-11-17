from .modules import *
from .config import *

if __name__ == "__main__":
    retrieval = Retrival()
    augmentation = Augmentation(vllm=USE_VLLM)
    verification = Verification(visual=False, vllm=USE_VLLM)
    explanation = Explanation(visual=False, vllm=USE_VLLM)
    
    sample_claim = "'It is the decision of the President,' not governors, to 'open up the states."
    
    claim, lst_text_evidence, lst_image_evidence = retrieval.run(sample_claim, is_summary=True)
    # print(lst_text_evidence)
    
    assert claim == sample_claim
    # print(lst_image_evidence)
    lst_augmentation = augmentation.run(claim, lst_image_evidence, lst_text_evidence)[-1]
    # print(lst_text_evidence)
    
    print("------")
    print(lst_augmentation)
    print("------")
    
    predict_label = verification.run(claim, lst_augmentation)[-1]
    
    print(predict_label)
    predict_explanation, _ = explanation.run(claim, predict_label, lst_augmentation)
    
    print(predict_explanation)