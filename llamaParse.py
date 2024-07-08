from llama_parse import LlamaParse
import json, os

from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

parser = LlamaParse(
    api_key = os.getenv("llamaParseKey"),  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="text"  # "markdown" and "text" are available
)

documents = parser.get_json_result("./Capture.png")

rows = documents[0]['pages'][0]['items'][0]['rows']


# dictified = markdown_to_json.dictify(documents[0].text)
# print(dictified)
      
# jsonified = markdown_to_json.jsonify(documents[0].text)
# print(jsonified)

import logging
import sys, json

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


import nltk
nltk.download('averaged_perceptron_tagger')

from pydantic import BaseModel, Field
from typing import List, Dict


class TableData(BaseModel):
    """Data model for table json information."""
    
    table: List[Dict[str, str]] = Field(
        description="List of rows of tables represented as key value pair where key is column header and value is the cell value"
    )


api_key = os.getenv("openAPIKey")
azure_endpoint = "https://hackaithon-qa-10-swn.openai.azure.com/"
api_version = "2024-02-15-preview"

prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assitant for converting a json representing table to anothet json. \n"
                "In the given json text mostly first object is column headers and others objects represents rows. Sometimes column headers can be present in second or third object as well \n"
                "You extract data and returns it in JSON format, according to provided JSON schema, from given text. \n"
                "REMEMBER to return extracted data only from provided text."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Json Table: \n" "------\n" "{jsonTable}\n" "------"
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
    output_cls=TableData,
    llm=llm,
    prompt=prompt,
    verbose=True,
)

output = program(jsonTable=rows)
print("Output JSON From .eml File: ")
print(json.dumps(output.dict(), indent=2))
