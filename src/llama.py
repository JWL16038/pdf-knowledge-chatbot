"""
https://www.baseten.co/blog/build-a-chatbot-with-llama-2-and-langchain/
"""
from pathlib import Path
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

ABSOLUTE_PATH = Path().resolve()
MODELS_PATH = Path("models/GGUF_models")
FULL_MODELS_PATH = ABSOLUTE_PATH.joinpath(MODELS_PATH)


template = """
Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer.
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 27
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llama2_path = FULL_MODELS_PATH.joinpath("Llama2/llama-2-13b-chat.Q5_K_M.gguf")

llm = LlamaCpp(
    model_path=str(llama2_path),
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What's 1 divided by 0? Wrong answers only"
llm_chain.run(question)