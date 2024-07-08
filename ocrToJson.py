from paddleocr import PaddleOCR
import json
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from flask import *  


paddleocr = PaddleOCR(lang="en",ocr_version="PP-OCRv4",show_log = False,use_gpu=True)

def paddle_scan(paddleocr,img_path_or_nparray):
    result = paddleocr.ocr(img_path_or_nparray,cls=True)
    result = result[0]
    boxes = [line[0] for line in result]       #boundign box
    txts = [line[1][0] for line in result]     #raw text
    scores = [line[1][1] for line in result]   # scores
    return  txts, result

# perform ocr scan
receipt_texts, receipt_boxes = paddle_scan(paddleocr,"imageReceipt.jpg")
print(50*"--","\ntext only:\n",receipt_texts)
print(50*"--","\nocr boxes:\n",receipt_boxes)


from transformers import AutoTokenizer, AutoModelForCausalLM
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("mychen76/mistral7b_ocr_to_json_v1")
model = AutoModelForCausalLM.from_pretrained("mychen76/mistral7b_ocr_to_json_v1")
model = model.to(device)
prompt=f"""### Instruction:
You are POS receipt data expert, parse, detect, recognize and convert following receipt OCR image result into structure receipt data object.
Don't make up value not in the Input. Output must be a well-formed JSON object.```json

### Input:
{receipt_boxes}

### Output:
"""

# with torch.inference_mode():
#     inputs = tokenizer(prompt,return_tensors="pt",truncation=True).to(device)
#     outputs = model.generate(**inputs, max_new_tokens=512)
#     result_text = tokenizer.batch_decode(outputs)[0]
#     print(result_text)


app = Flask(__name__)

@app.route('/')   
def hello_world():
    return 'Hello World'
  
@app.route('/parse', methods = ['POST'])   
def success():   
    if request.method == 'POST':   
        receipt_texts, receipt_boxes = paddle_scan(paddleocr,"imageReceipt.jpg")
        prompt=f"""### Instruction:
            You are POS receipt data expert, parse, detect, recognize and convert following receipt OCR image result into structure receipt data object.
            Don't make up value not in the Input. Output must be a well-formed JSON object.```json

            ### Input:
            {receipt_boxes}

            ### Output:
            """
        with torch.inference_mode():
            inputs = tokenizer(prompt,return_tensors="pt",truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=512).to(device)
            result_text = tokenizer.batch_decode(outputs)[0]
            print(result_text)
        return result_text
  
if __name__ == '__main__':   
    app.run(port=5001)