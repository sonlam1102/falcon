from flask import Flask, render_template, request, jsonify, abort, make_response
from end2end.modules import *
import base64

def load_fact_check_models():
    global retrieval
    global augmentation
    global verification
    global explanation
    
    retrieval = Retrival()
    augmentation = Augmentation()
    verification = Verification(visual=False)
    explanation = Explanation(visual=False)
    
    print("Load done")


def dump_response():
    lst_text_evidence = ["a", "b", "c", "d", "e"]
    white_im_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    new_base54img = [white_im_base64, white_im_base64, white_im_base64, white_im_base64, white_im_base64]
    response_data = jsonify({
        "claim": "test_claim",
        "lst_evidence": lst_text_evidence,
        "lst_image_evidence": new_base54img,
        # "lst_augmentation": new_lst_augmentation,
        "label": "True",
        "ruiling": "hehehe"
    })

    return response_data
    
app = Flask(__name__)
load_fact_check_models()

@app.route('/')
def main_page():
    return render_template('base.html')

@app.route('/check_claim', methods=['POST', 'GET'])
def check_claim():
    if request.method == 'POST':
        data = request.get_json()
        claim = data.get("claim")
        
        # example: COVID 19 is not kill people as WHO announced
        print(claim)
        
        claim, lst_text_evidence, lst_image_evidence = retrieval.run(claim, is_summary=True)
        print("Done retieval")
        lst_augmentation = augmentation.run(claim, lst_image_evidence, lst_text_evidence)[-1]
        print("Done augmentation")
        _, new_lst_augmentation, predict_label = verification.run(claim, lst_augmentation)
        print("Done verify")
        
        predict_explanation, _ = explanation.run(claim, predict_label, lst_augmentation)
        print("Done explanation")

        new_base54img = []
        for li in lst_image_evidence:
            print(li)
            with open(li, "rb") as image_file:
                img_b64 = base64.b64encode(image_file.read())
                new_base54img.append(img_b64.decode('utf-8'))
            image_file.close()
        
        response_data = jsonify({
            "claim": claim,
            "lst_evidence": lst_text_evidence,
            "lst_image_evidence": new_base54img,
            # "lst_augmentation": new_lst_augmentation,
            "label": predict_label,
            "ruiling": predict_explanation
        })
        
        # response_data = dump_response()
        return make_response(response_data, 200)
    else:
        abort(405)

# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()