import llama
from loader import load_pdf
from database import Database

template_text = """Context: {context}

Based on the context of the document, please provide me an answer for following question: {question}

Use all of the information that is given to you. Do not make up or use any other information that is outside of this document.
"""

def run():
    db = load_pdf()
    prompt_temp = llama.build_template(template_text)
    llm_chain = llama.build_llm_chain(prompt_temp)
    question = "Is greedy search always optimal? State true or false and explain why."
    context = db.query("Greedy search", 3)
    llama.prompt_model(llm_chain, question=question, context=context)

if __name__ == "__main__":
    run()