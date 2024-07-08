from distutils.log import debug 
from fileinput import filename 
from flask import *  
from llama_parse import LlamaParse
import logging
import sys, json, os

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage

from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

import nltk
nltk.download('averaged_perceptron_tagger')

from pydantic import BaseModel, Field
from typing import List


from llama_index.core import download_loader
from llama_index.readers.file import UnstructuredReader


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

app = Flask(__name__)   

class Item(BaseModel):
    item_name: str = Field(description="Item Name")
    quamtity: str = Field(description="Quantity")
    price: str = Field(description="Price")

class EmailData(BaseModel):
    """Data model for email extracted information."""
    
    items: List[Item] = Field(
        description="List of Items described in email having list of items ordred, quantity and price"
    )
    order_no: str = Field(description="Order No")
    
    order_status: str = Field(description="Order Status")
    order_placed_date: str = Field(description="Order placed at")
    order_delievered_date: str = Field(description="Order Delivered at")
    sender_email_id: str = Field(description="Email Id of the email sender.")
    email_date_time: str = Field(description="Date and time of email")
    

@app.route('/')   
def hello_world():
    return 'Hello World'
  
@app.route('/parse', methods = ['POST'])   
def success():   
    if request.method == 'POST':   
        f = request.files['files'] 
        f.save(f.filename) 
        loader = UnstructuredReader()

        # For eml file
        eml_documents = loader.load_data(f.filename)
        email_content = eml_documents[0].text

        api_key = os.getenv("openAPIKey")
        azure_endpoint = "https://hackaithon-qa-10-swn.openai.azure.com/"
        api_version = "2024-02-15-preview"

        prompt = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are an expert assitant for extracting insights from email in JSON format. \n"
                        "You extract data and returns it in JSON format, according to provided JSON schema, from given email message. \n"
                        "REMEMBER to return extracted data only from provided email message."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "Email Message: \n" "------\n" "{email_msg_content}\n" "------"
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

    
        program = OpenAIPydanticProgram.from_defaults(
            llm=llm,
            prompt=prompt,
            verbose=True,
        )

        output = program(email_msg_content=email_content)
        return json.dumps(output.dict(), indent=2)     
  
if __name__ == '__main__':   
    app.run(port=5001)