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

Based on the context of the document, please provide me an answer for following question: {question}

Use all of the information that is given to you. Do not make up or use any other information that is outside of this document.
"""

prompt_template = PromptTemplate(template=template, input_variables=["context","question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 27
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llama2_path = FULL_MODELS_PATH.joinpath("Llama2/llama2-13b-psyfighter2.Q5_K_M.gguf")

llm = LlamaCpp(
    model_path=str(llama2_path),
    n_ctx=6000,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    temperature = 0.9,
    max_tokens = 2096,
    n_parts=1,
    verbose=True,  # Verbose is required to pass to the callback manager
)

llm_chain = LLMChain(prompt=prompt_template, llm=llm)

db = load_pdf()
question = "Give me some benefits of simulated annealing over hill climbing"
context = db.query("Simulated annealing", 3)
context = " ".join([text.page_content for text in context])
response = llm_chain.predict(context=context, question=question)
print(f"Question: {question}")
print(f"Answer: {response}")
