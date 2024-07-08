from distutils.log import debug 
from fileinput import filename 
from flask import *  
from llama_parse import LlamaParse
import os
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

app = Flask(__name__)   
  
@app.route('/')   
def hello_world():
    return 'Hello World'
  
@app.route('/parse', methods = ['POST'])   
def success():   
    if request.method == 'POST':   
        f = request.files['file'] 
        f.save(f.filename) 
        parser = LlamaParse(
            api_key = os.getenv("llamaParseKey"),  # can also be set in your env as LLAMA_CLOUD_API_KEY
            result_type="text"  # "markdown" and "text" are available
        )
    documents = parser.get_json_result(f.filename)
    rows = documents[0]['pages'][0]['items'][0]['rows']    
    return rows
  
if __name__ == '__main__':   
    app.run(debug=True)