"""
https://www.baseten.co/blog/build-a-chatbot-with-llama-2-and-langchain/
"""
import os
import tomllib
from pathlib import Path
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp, HuggingFaceHub
from langchain.prompts import PromptTemplate

ABSOLUTE_PATH = Path().resolve()
MODELS_PATH = Path("models/GGUF_models")
FULL_MODELS_PATH = ABSOLUTE_PATH.joinpath(MODELS_PATH)

with open(Path().resolve().parent / "config/chatbot.toml", "rb") as c:
    config = tomllib.load(c)
    
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def build_llm_chain(template: str,
                    input_variables=["context","question"],
                    type = "llamacpp",) -> LLMChain:

    if type == "llamacpp":
        llm = LlamaCpp(
            model_path=FULL_MODELS_PATH.joinpath(config["llamacpp"]["model_path"]).as_posix,
            n_ctx=config["llamacpp"]["n_ctx"],
            n_gpu_layers=config["llamacpp"]["n_gpu_layers"],
            n_batch=config["llamacpp"]["n_batch"],
            callback_manager=callback_manager,
            temperature = config["llamacpp"]["temperature"],
            max_tokens = config["llamacpp"]["max_tokens"],
            n_parts=1,
            verbose=False,  # Verbose is required to pass to the callback manager
        )
    elif type == "hf":
        llm = HuggingFaceHub(
            huggingfacehub_api_token=os.getenv("hf_token"),
            repo_id=config["huggingfacehub"]["model_id"],
            model_kwargs={
                "temperature":config["huggingfacehub"]["temperature"], 
                "max_length":config["huggingfacehub"]["max_length"]
            },
            callback_manager = callback_manager,
            verbose=False
        )
    else:
        raise ValueError(f"Not a valid type: {type}") 
    llm_chain = LLMChain(prompt = PromptTemplate(template=template, input_variables=input_variables), 
                         llm=llm,
                         )
    return llm_chain

def prompt_model(llm_chain, question, context):
    response = llm_chain.predict(context=context, question=question)
    print(response)