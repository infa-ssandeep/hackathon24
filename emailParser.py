import logging
import sys, json
import os
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage

from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


import nltk
nltk.download('averaged_perceptron_tagger')

from pydantic import BaseModel, Field
from typing import List

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
    
from llama_index.core import download_loader
from llama_index.readers.file import UnstructuredReader

# Initialize the UnstructuredReader
loader = UnstructuredReader()

# For eml file
eml_documents = loader.load_data("Your Swiggy order was successfully delivered.eml")
email_content = eml_documents[0].text
print("\n\n Email contents")
print(email_content)

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
    output_cls=EmailData,
    llm=llm,
    prompt=prompt,
    verbose=True,
)

output = program(email_msg_content=email_content)
print("Output JSON From .eml File: ")
print(json.dumps(output.dict(), indent=2))