"""
https://www.baseten.co/blog/build-a-chatbot-with-llama-2-and-langchain/
"""
from pathlib import Path
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from loader import load_pdf

ABSOLUTE_PATH = Path().resolve()
MODELS_PATH = Path("models/GGUF_models")
FULL_MODELS_PATH = ABSOLUTE_PATH.joinpath(MODELS_PATH)


template = """Context: {context}

Based on Context provide me answer for following question
Question: {question}

Tell me the information about the fact. The answer should be from context only
do not use general knowledge to answer the query"""

prompt_template = PromptTemplate(template=template, input_variables=["context","question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 27
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llama2_path = FULL_MODELS_PATH.joinpath("Llama2/llama-2-13b-chat.Q5_K_M.gguf")

llm = LlamaCpp(
    model_path=str(llama2_path),
    n_ctx=6000,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    temperature = 0.9,
    max_tokens = 4095,
    n_parts=1,
    verbose=True,  # Verbose is required to pass to the callback manager
)

llm_chain = LLMChain(prompt=prompt_template, llm=llm)

db = load_pdf()
question = "How many questions are in this paper?"
context = db.query("Questions in this paper")
context = [text.page_content for text in context]
llm_chain.predict(context=" ".join(context), question=question)
