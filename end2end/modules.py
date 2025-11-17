from .misc import *
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from .config import *

class Augmentation:
    def __init__(self, vllm=False):
        self.processor, self.model = load_peft_model_vision(AUGMENTATION_MODEL, flash_attention=FLASH_ATTENTION, token=HF_TOKEN, quantize=QUANTIZE)
        # if vllm:
        #     self.processor, self.model = load_peft_model_vision_with_vllm(AUGMENTATION_MODEL, token=HF_TOKEN, quantize=QUANTIZE)
        # else:
        #     self.processor, self.model = load_peft_model_vision(AUGMENTATION_MODEL, flash_attention=FLASH_ATTENTION, token=HF_TOKEN, quantize=QUANTIZE)
        
        self._is_vllm = vllm   
        self.bge_model = load_bge_model()
        print("Successfully load Augmentation model and BGE model.")
    
    def _convert_to_conversation_infer(self, text, image):            
        system_message = """
            Please generate a short paragraph describing the about the consistency of the image based on the given text following this template:
            \n
            <HYPOTHESIS>: Please determining whether the image is consistent with the text or not.
            <EXPLANATION>: Explanation the aligment between the image hypothesis and the text.
            <FINAL ANSWER>: Give one paragraph describing the consistency of the image and text based on the explanation.
        """
        
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "resized_height": 720, "resized_width": 1280},
                    {"type": "text", "text": text},
                ],
            },
        ]
        return conversation

    def _inference_model(self, image_path, text):
        prompt = self._convert_to_conversation_infer(text, image_path)
        
        # print(prompt)
        text_inputs = [self.processor.apply_chat_template(prompt, tokenize=False)]
        image_inputs = [process_vision_info(prompt)[0]]
        
        model_inputs = self.processor(
            text=text_inputs, images=image_inputs, return_tensors="pt", padding=True
        ).to("cuda")
        
        # if self._is_vllm:
        #     sampling_params = SamplingParams(do_sample=False, max_new_tokens=2048)
        #     outputs = self.model.generate(model_inputs, sampling_params)
        #     print(len(outputs))
        #     for output in outputs:
        #         generated_text = output.outputs[0].text
            
        #     return generated_text
            
        output_ids = self.model.generate(**model_inputs, max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]
    
    def generate_augmentation(self, lst_image, lst_text):
        def make_image_description_evidence(image_explaination):
            expl = image_explaination.split("<EXPLANATION>")[-1]
            expl = expl.replace("<FINAL ANSWER>:", "")
            expl = expl.replace("     ", " ")
            expl = expl.replace("\n", "")
            expl = expl.replace(": ", "")

            hyp = image_explaination.split("<EXPLANATION>")[0]
            hyp = hyp.replace("\n", "")
            hyp = hyp.replace("<HYPOTHESIS>:", "")
            return expl, hyp
        
        alignment = []
        for te in lst_text:
            for im in lst_image:
                try:
                    align_data = make_image_description_evidence(self._inference_model(im, te))
                    alignment.append({
                        "image": im,
                        "text": te,
                        "alignment": align_data
                    })
                except Exception as e:
                    print(e)
        return alignment
    
    def quey(self, claim, lst_aligment):
        new_lst_aligment = [d['alignment'][0] + d['alignment'][1].replace("assistant ", " In conclusion, ") for d in lst_aligment]
        
        claim_encode_text = self.bge_model.encode([claim])['dense_vecs']
        claim_encode_aligment = self.bge_model.encode(new_lst_aligment)['dense_vecs']
        similarity = claim_encode_text @ claim_encode_aligment.T
        
        # print(similarity)
        # print("-----")
        # print(similarity.flatten())
        # print("******")
        
        top_k_aligment = get_top_k(similarity.flatten(), 5)
        # print(top_k_aligment)
        assert len(top_k_aligment) == len(lst_aligment)
        
        final_aligment = []
        for i in range(0, len(top_k_aligment)):
            if top_k_aligment[i] > 0:
                final_aligment.append(lst_aligment[i])
        
        return final_aligment
        
    
    def run(self, claim, lst_image, lst_text):
        # print("Performing augmentation: ")
        aligment = self.generate_augmentation(lst_image, lst_text)
        
        # print("Get most similar aligment: ")
        lst_final_aligment = self.quey(claim, aligment)
        
        return claim, lst_image, lst_text, lst_final_aligment
    

class Retrival:
    def __init__(self):
        self.bge_model = load_bge_model()
        self.clip_model, self.clip_device = load_clip_model()
        self.summary_tokenizer, self.summary_model = load_summary_model()
        print("LOADED BGE, CLIP and SUMMARY MODEL")
        
        self.text_evidence_db = get_text_evidences_db(EVIDENCE_DB)
        self.image_db_path = IMAGE_DB
        
        self._image_emb = None
        self._text_emb = None
        self._image_ids = None
        self._text_ids = None
        
        self._build_flag_embedding()
    
    def _build_flag_embedding(self):
        self._image_ids = np.load(RETRIVAL_MODEL_EMBEDDING + "/image_embedding_db_clip_id.npy")
        self._image_emb = torch.from_numpy(np.load(RETRIVAL_MODEL_EMBEDDING + "/image_embedding_db_clip.npy")).to(self.bge_model.device)
        
        self._text_ids = np.load(RETRIVAL_MODEL_EMBEDDING + "/text_embedding_db_bge_id.npy")
        self._text_emb = torch.from_numpy(np.load(RETRIVAL_MODEL_EMBEDDING + "/text_embedding_db_bge.npy")).to(self.bge_model.device)
        
        print(self._text_emb.shape)
        print(self._image_emb.shape)

        print("----Loaded evidence DB-----------")

    def get_evidence_db_ids(self):
        return self._text_ids, self._image_ids
    
    def set_evidence_db_ids(self, text_ids, image_ids):
        self._text_ids = text_ids
        self._image_ids = image_ids
    
    def retrieve_text_similarity(self, query):
        claim_encode_text = self.bge_model.encode(query)['dense_vecs']
        claim_encode_text = np.expand_dims(claim_encode_text, axis=0)
        text_sim = F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self.bge_model.device), self._text_emb)
        text_sim = text_sim.detach().cpu().numpy()

        return text_sim
    
    def retrieve_image_similarity_clip(self, query):
        text = clip.tokenize([query], truncate=True).to(self.clip_device)
        claim_encode_text = self.clip_model.encode_text(text)
        image_sim = F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self.clip_device), self._image_emb)
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
        
        return lst_doc_ids, lst_image_ids
    
    
    def do_sum(self, claim, text_evidence):
        command = "summarize: {} </s> {} </s>".format(claim, text_evidence)
        input = self.summary_tokenizer(command, return_tensors="pt", padding="max_length", max_length=1024, truncation=True, return_attention_mask=True)
        out_results = self.summary_model.generate(
                input_ids=input.input_ids.to(self.summary_model.device),
                attention_mask=input.attention_mask.to(self.summary_model.device),
                num_beams=3,
                max_length=512,
                min_length=0,
                no_repeat_ngram_size=2,
            )

        preds = [self.summary_tokenizer.decode(p, skip_special_tokens=True) for p in out_results]
        return preds[0]
    
    def run(self, claim, is_summary=True):
        lst_retrieved_doc_ids, lst_image_ids = self.retrieve_evidence(claim)
        lst_doc = get_full_doc(lst_retrieved_doc_ids, self.text_evidence_db)
        lst_image = get_full_image(lst_image_ids, self.image_db_path, return_image=False)
        
        if is_summary:
            # print("summarizing the evidence")
            lst_new_doc = [self.do_sum(claim, d) for d in lst_doc]
            return claim, lst_new_doc, lst_image
        else:
            return claim, lst_doc, lst_image
    
    
class Verification:
    def __init__(self, visual=True, vllm=False):
        if visual:
            self.processor, self.model = load_peft_model_vision(VERIFICATION_MODEL, flash_attention=FLASH_ATTENTION, token=HF_TOKEN, quantize=QUANTIZE)
            print("Successfully load Verification model visual.")
        else:
            self.processor, self.model = load_peft_model(VERIFICATION_MODEL_TEXT, flash_attention=FLASH_ATTENTION, token=HF_TOKEN, quantize=QUANTIZE)
            print("Successfully load Verification model text.")
            
        # if visual:
        #     if vllm:
        #         self.processor, self.model = load_peft_model_vision_with_vllm(VERIFICATION_MODEL, token=HF_TOKEN, quantize=QUANTIZE)
        #     else: 
        #         self.processor, self.model = load_peft_model_vision(VERIFICATION_MODEL, flash_attention=FLASH_ATTENTION, token=HF_TOKEN, quantize=QUANTIZE)
        #     print("Successfully load Verification model visual.")
        # else:
        #     if vllm:
        #         self.processor, self.model = load_peft_model_with_vllm(VERIFICATION_MODEL_TEXT, token=HF_TOKEN, quantize=QUANTIZE)
        #     else:
        #         self.processor, self.model = load_peft_model(VERIFICATION_MODEL_TEXT, flash_attention=FLASH_ATTENTION, token=HF_TOKEN, quantize=QUANTIZE)
        #     print("Successfully load Verification model text.")
        
        self._is_vllm = vllm
        self._is_visual = visual
        self._aligment = None
        
    def __convert_to_conversation_infer_visual(self, claim, lst_aligment): 
        system_message = f"""
        You are an assistant that help verifying the veracity of a claim based on the multimodal evidence including text and image.
        The claim is: {claim}
        Belows are list of evidence containing the image, the text evidence, and a sentence describe the consistency between text and image:
        """
        system_message2 = f"""
            Base on given evindece belows, let's determine the truthfulness of the claim as True, False or Not enough information. Response only one of three values: True, False, or Not enough information.
        """
        
        multimodal_content = []
        multimodal_aligmenent = []
        for s in lst_aligment:
            multimodal_content.append({
                "type": "image",
                "image": s['image'],
                "resized_height": 480, 
                "resized_width": 720
            }) 
            
            multimodal_content.append({
                "type": "text",
                "text": "TEXT EVIDENCE: " + s['text'],
            }) 
            
            tmp_aligment = {
                "type": "text",
                "text": "CONSISTENCY OF TEXT and IMAGE: " + s['alignment'][0] + s['alignment'][1].replace("assistant ", " In conclusion, "),
            }
            multimodal_content.append(tmp_aligment)
            multimodal_aligmenent.append(tmp_aligment)
            
            
        self._aligment = list(set([m['text'] for m in multimodal_aligmenent]))
        
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": multimodal_content,
            },
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message2}],
            },
        ]
        return conversation
    
    
    def __convert_to_conversation_infer_text(self, claim, lst_aligment): 
        system_message = f"""
        You are an assistant that help verifying the veracity of a claim based on the multimodal evidence including text and image.
        The claim is: {claim}
        Belows are list of evidence containing the image, the text evidence, and a sentence describe the consistency between text and image:
        """
        system_message2 = f"""
            Base on given evindece belows, let's determine the truthfulness of the claim as True, False or Not enough information. Response only one of three values: True, False, or Not enough information.
        """

        # multimodal_content = ""
        # for s in lst_aligment:
        #     multimodal_content += "TEXT EVIDENCE: " + s['text'] + "\n" + "CONSISTENCY OF TEXT and IMAGE: " + s['alignment'][0] + s['alignment'][1].replace("assistant ", " In conclusion, ") + "\n"
        
        multimodal_content = []
        for s in lst_aligment:
            multimodal_content.append("TEXT EVIDENCE: " + s['text'] + "\n" + "CONSISTENCY OF TEXT and IMAGE: " + s['alignment'][0] + s['alignment'][1].replace("assistant ", " In conclusion, ") + "\n")
        
        multimodal_content = list(set(multimodal_content))
        new_multimodal_content = ''.join(str(x) for x in multimodal_content)
        # print(new_multimodal_content)
        # print("-----")
        # raise Exception
    
        conversation = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": new_multimodal_content,
            },
            {
                "role": "system",
                "content": system_message2,
            },
        ]
        self._aligment = multimodal_content
        
        return conversation
    
    
    def _convert_to_conversation_infer(self, claim, lst_aligment):
        if self._is_visual:
            return self.__convert_to_conversation_infer_visual(claim, lst_aligment)
        else:
            return self.__convert_to_conversation_infer_text(claim, lst_aligment)
            
    
    def _inference_model(self, claim, lst_aligment):
        prompt = self._convert_to_conversation_infer(claim, lst_aligment)
        
        # print(prompt)
        text_inputs = [self.processor.apply_chat_template(prompt, tokenize=False)]
        image_inputs = [process_vision_info(prompt)[0]]
        
        if self._is_visual:
            model_inputs = self.processor(
                text=text_inputs, images=image_inputs, return_tensors="pt", padding=True
            ).to("cuda")
        else:
            model_inputs = self.processor(
                text=text_inputs, return_tensors="pt", padding=True
            ).to("cuda")
        
        # if self._is_vllm:
        #     sampling_params = SamplingParams(do_sample=False, max_new_tokens=2048)
        #     outputs = self.model.generate(model_inputs, sampling_params)
        #     print(len(outputs))
        #     for output in outputs:
        #         generated_text = output.outputs[0].text
            
        #     return generated_text
        
        output_ids = self.model.generate(**model_inputs, max_new_tokens=100)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]
    
    
    def make_verification(self, claim, lst_aligment):
        response = self._inference_model(claim, lst_aligment)
        return response.replace("assistant\n", "")
    
    def run(self, claim, lst_aligment):
        label = self.make_verification(claim, lst_aligment)
        
        return claim, self._aligment, label
    
    
class Explanation:
    def __init__(self, visual=True, vllm=False):
        if visual:
            self.processor, self.model = load_peft_model_vision(EXPLANATION_MODEL, flash_attention=FLASH_ATTENTION, token=HF_TOKEN, quantize=QUANTIZE)
            print("Successfully load Explanation model visual.")
        else:
            self.processor, self.model = load_peft_model(EXPLANATION_MODEL_TEXT, flash_attention=FLASH_ATTENTION, token=HF_TOKEN, quantize=QUANTIZE)
            print("Successfully load Explanaiton model text.")
        
        # if visual:
        #     if vllm:
        #         self.processor, self.model = load_peft_model_vision_with_vllm(EXPLANATION_MODEL, token=HF_TOKEN, quantize=QUANTIZE)
        #     else: 
        #         self.processor, self.model = load_peft_model_vision(EXPLANATION_MODEL, flash_attention=FLASH_ATTENTION, token=HF_TOKEN, quantize=QUANTIZE)
        #     print("Successfully load Verification model visual.")
        # else:
        #     if vllm:
        #         self.processor, self.model = load_peft_model_with_vllm(EXPLANATION_MODEL_TEXT, token=HF_TOKEN,quantize=QUANTIZE)
        #     else:
        #         self.processor, self.model = load_peft_model(EXPLANATION_MODEL_TEXT, flash_attention=FLASH_ATTENTION, token=HF_TOKEN, quantize=QUANTIZE)
        #     print("Successfully load Verification model text.")
        
        self._is_vllm = vllm
        self._is_visual = visual
        self._aligment = None
    
    def __convert_to_conversation_infer_visual(self, claim, predict_label, lst_aligment):    
        system_message = f"""
            You are an assistant that help explaning the veracity of a verified claim based on the multimodal evidence including text and image.
            The claim is: {claim}
            Belows are list of evidence containing the image, the text evidence, and a sentence describe the consistency between text and image:
        """
        
        system_message2 = f"""
            Base on given evindece belows, knowing that the claim is determined as {predict_label}, let's generate a paragraph to explain the truthfulness of the claim.
        """
        
        multimodal_content = []
        multimodal_aligment = []
        for s in lst_aligment:
            multimodal_content.append({
                "type": "image",
                "image": s['image'],
                "resized_height": 480, 
                "resized_width": 720
            }) 
            
            multimodal_content.append({
                "type": "text",
                "text": "TEXT EVIDENCE: " + s['text'],
            }) 
            
            tmp_aligment = {
                "type": "text",
                "text": "CONSISTENCY OF TEXT and IMAGE: " + s['alignment'][0] + s['alignment'][1].replace("assistant ", " In conclusion, "),
            }
            
            multimodal_content.append(tmp_aligment)
            multimodal_aligment.append(tmp_aligment)
        
        self._aligment = list(set([m['text'] for m in multimodal_aligment]))
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": multimodal_content,
            },
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message2}],
            },
        ]
        return conversation
    
    
    def __convert_to_conversation_infer_text(self, claim, predict_label, lst_aligment):    
        system_message = f"""
            You are an assistant that help explaning the veracity of a verified claim based on the multimodal evidence including text and image.
            The claim is: {claim}
            Belows are list of evidence containing the image, the text evidence, and a sentence describe the consistency between text and image:
        """
        
        system_message2 = f"""
            Base on given evindece belows, knowing that the claim is determined as {predict_label}, let's generate a paragraph to explain the truthfulness of the claim.
        """
        
        # multimodal_content = ""
        # for s in lst_aligment:
        #     multimodal_content += "TEXT EVIDENCE: " + s['text'] + "\n" + "CONSISTENCY OF TEXT and IMAGE: " + s['alignment'][0] + s['alignment'][1].replace("assistant ", " In conclusion, ") + "\n"
        
        multimodal_content = []
        for s in lst_aligment:
            multimodal_content.append("TEXT EVIDENCE: " + s['text'] + "\n" + "CONSISTENCY OF TEXT and IMAGE: " + s['alignment'][0] + s['alignment'][1].replace("assistant ", " In conclusion, ") + "\n")
        
        multimodal_content = list(set(multimodal_content))
        new_multimodal_content = ''.join(str(x) for x in multimodal_content)
        # print(new_multimodal_content)
        # print("-----")
        # raise Exception
        
        conversation = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": new_multimodal_content,
            },
            {
                "role": "system",
                "content": system_message2,
            },
        ]
        self._aligment = multimodal_content
        return conversation
    
    
    def _convert_to_conversation_infer(self, claim, predict_label, lst_aligment):
        if self._is_visual:
            return self.__convert_to_conversation_infer_visual(claim, predict_label, lst_aligment)
        else:
            return self.__convert_to_conversation_infer_text(claim, predict_label, lst_aligment)
    
    
    def _inference_model(self, claim, predict_label, lst_aligment):
        prompt = self._convert_to_conversation_infer(claim, predict_label, lst_aligment)
        
        # print(prompt)
        text_inputs = [self.processor.apply_chat_template(prompt, tokenize=False)]
        image_inputs = [process_vision_info(prompt)[0]]
        
        if self._is_visual:
            model_inputs = self.processor(
                text=text_inputs, images=image_inputs, return_tensors="pt", padding=True
            ).to("cuda")
        else:
            model_inputs = self.processor(
                text=text_inputs, return_tensors="pt", padding=True
            ).to("cuda")
        
        # if self._is_vllm:
        #     sampling_params = SamplingParams(do_sample=False, max_new_tokens=2048)
        #     outputs = self.model.generate(model_inputs, sampling_params)
        #     print(len(outputs))
        #     for output in outputs:
        #         generated_text = output.outputs[0].text
            
        #     return generated_text
        
        output_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]
    
    def make_explanation(self, claim, predict_label, lst_aligment):
        response = self._inference_model(claim, predict_label, lst_aligment)
        # print("test====")
        # print(response.replace("assistant\n", ""))
        # print("end_test====")
        return response.replace("assistant\n", "")
    
    def run(self, claim, predict_label, lst_aligment):
        ruling = self.make_explanation(claim, predict_label, lst_aligment)
        
        return ruling, self._aligment
