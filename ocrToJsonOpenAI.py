from paddleocr import PaddleOCR
import json
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from flask import *  

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field
from typing import Dict
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 



def paddle_scan(paddleocr,img_path_or_nparray):
    result = paddleocr.ocr(img_path_or_nparray,cls=True)
    result = result[0]
    boxes = [line[0] for line in result]       #boundign box
    txts = [line[1][0] for line in result]     #raw text
    scores = [line[1][1] for line in result]   # scores
    return  txts, result

# with torch.inference_mode():
#     inputs = tokenizer(prompt,return_tensors="pt",truncation=True).to(device)
#     outputs = model.generate(**inputs, max_new_tokens=512)
#     result_text = tokenizer.batch_decode(outputs)[0]
#     print(result_text)

class TableData(BaseModel):
    """Data model for Receipt OCR image result information."""
    
    items: Dict = Field()


app = Flask(__name__)

@app.route('/')   
def hello_world():
    return 'Hello World'
  
@app.route('/parse', methods = ['POST'])   
def success():   
    if request.method == 'POST':   
        f = request.files['files'] 
        f.save(f.filename) 
        paddleocr = PaddleOCR(lang="en",ocr_version="PP-OCRv4",show_log = False,use_gpu=True)
        receipt_texts, receipt_boxes = paddle_scan(paddleocr, f.filename)
        print(receipt_boxes)
        api_key = os.getenv("openAPIKey")
        azure_endpoint = "https://hackaithon-qa-10-swn.openai.azure.com/"
        api_version = "2024-02-15-preview"

        prompt = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are POS receipt data expert, parse, detect, recognize and convert following receipt OCR image result into structure receipt data object.\n"
                        "Don't make up value not in the Input. Output must be a well-formed JSON object.```json"
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "Receipt OCR image result : \n" "------\n" + str(receipt_boxes) + "\n------"
                    ),
                ),
            ]
        )


        
        llm = AzureOpenAI(
            engine="hackaithon-qa-10-swn",
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

        
        # program = OpenAIPydanticProgram.from_defaults(
        #     output_cls=TableData,
        #     llm=llm,
        #     prompt=prompt,
        #     verbose=True,
        # )

        print("calling openAi")
        resp = llm.complete(prompt.get_template())
        print(resp)
        # output = program(email_msg_content=email_content)
        return jsonify(json.loads(resp.text))
  
  
if __name__ == '__main__':   
    app.run(port=5002)