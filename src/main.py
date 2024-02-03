import model
from loader import load_documents

template_text = """Context: {context}

Based on the context of the document, please provide me an answer for following question: {question}

Use all of the information that is given to you. Do not make up or use any other information that is outside of this document.
"""

def run():
    db = load_documents()
    print(db.get_num_documents())
    llm_chain = model.build_llm_chain(template_text, type="hf")
    # Prompt context
    context_prompt = input("Enter query to search in db: ")
    context = db.query(context_prompt, 3)
    # Prompt question 
    # E.g "Can you describe to me the importance of conducting heuristic evaluation for UX as if you're talking to a teenager with no technical knowledge?"
    question = input("Enter question to ask: ")
    model.prompt_model(llm_chain, question=question, context=context)

if __name__ == "__main__":
    run()