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

n_gpu_layers = 27
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def build_template(template: str) -> PromptTemplate:
    prompt_template = PromptTemplate(template=template, input_variables=["context","question"])
    return prompt_template

def build_llm_chain(prompt_template: PromptTemplate, model_path: str = "Llama2/llama2-13b-psyfighter2.Q5_K_M.gguf") -> LLMChain:
    llama2_path = FULL_MODELS_PATH.joinpath(model_path)

    llm = LlamaCpp(
        model_path=str(llama2_path),
        n_ctx=6000,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        temperature = 0.9,
        max_tokens = 2096,
        n_parts=1,
        verbose=False,  # Verbose is required to pass to the callback manager
    )

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    return llm_chain

def prompt_model(llm_chain, question, context):
    print(f"Question: {question}")
    llm_chain.predict(context=context, question=question)
    # print(f"Answer: {response}")