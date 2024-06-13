from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_openai.chat_models import ChatOpenAI
import openai
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import random
import fitz
import json
from RAG import *


def load_pdf_content(filename):
    doc = fitz.open(filename)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


chunkSize = 2500
pdf_filename = "source.pdf"

# Load and extract text from the PDF document
pdf_text = load_pdf_content(pdf_filename)

# Create a Document object with the extracted text
doc = Document(page_content=pdf_text, metadata={'source': pdf_filename})

# Split the document into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunkSize, chunk_overlap=20)
documents = text_splitter.split_documents([doc])
for i, section in enumerate(documents):
    section.metadata['section_index'] = i

# Create DataFrame from documents
df = pd.DataFrame([(d.page_content, d.metadata)
                  for d in documents], columns=["text", "meta"])


complexity = ["Make a minor spelling mistake in two words of the output question", "Add a random word somewhere in the output question",
              "Make a major spelling mistake in a less important word of the output question", "Omit an unimportant word from the output question", "Switch the order of two adjacent words in the output question"]


def generate_question(llm_purpose, style, complexity_id):
    transform = [f"""You are an expert at writing questions. 
Your task is to re-write questions that will be used to evaluate the following agent:
- Model description: {llm_purpose}  

Respect the following rules to reformulate the question:
- Apply the following adjustment: {complexity[complexity_id]}

The user will provide a single question.
You will return the modified question.
You must output only the question, without any other wrapping text. Make sure you only return the question itself.
""",
                 f"""You are an expert at writing questions. 
Your task is to re-write questions that will be used to evaluate the following agent:
- Model description: {llm_purpose}  

Respect the following rules to reformulate the question:
- The re-written question should not be longer than the original question by up to 10 to 15 words. 
- The re-written question should be more elaborated than the original. 
- The re-written question should be more difficult to handle for AI models but it must be understood and answerable by humans.
- Add one or more constraints / conditions to the question.
- Apply the following adjustment: {complexity[complexity_id]}

The user will provide a single question.
You will return the modified question.
You must output only the question, without any other wrapping text. Make sure you only return the question itself.
""",
                 f"""You are an expert at rewriting questions.
Your task is to re-write questions that will be used to evaluate the following agent:
- Agent description: {llm_purpose}  

Your task is to add situational context about the user inside the question. 
Create a situational context which would make sense as exigence for the question being asked.
Please respect the following rules to generate the question:
- Add exactly one sentence to the question in order to incorporate the situational context. This can be as simple as just a greeting, or an exigence for the question.
- The question combined with the added sentence must sound plausible and coming from a real human user.
- The original question and answer should be preserved.
- The question must be self-contained and understandable by humans. 
- Apply the following adjustment: {complexity[complexity_id]}

The user will provide a single question.
You will return the modified question.
You must output only the question and the added situational sentence, without any other wrapping text. Make sure you only return the question and situational sentence.
"""

                 ]

    # Select a random entry
    random_entry = df.sample().iloc[0]
    entry_text = random_entry["text"]
    entry_index = random_entry["meta"]["section_index"]

    # Extract an objective fact from the text
    prompt = """You are a powerful auditor, your role is to generate question & answer pair from a context paragraph.

    The agent model you are auditing is the following:
    - Agent description: {llm_purpose}

    Your question must be related to a provided context.  
    Please respect the following rules to generate the question:
    - The answer to the question should be found inside the provided context
    - The question must be self-contained
    - The question should be answerable in a concise manner

    The user will provide the context, consisting of a paragraph of informative text.
    You will return the question and the precise answer to the question based exclusively on the provided context in the most recent message.
    You must output a single JSON object with keys 'question' and 'answer', without any other wrapping text or markdown.Make sure you only return valid JSON. """

    input_ex = "[Document(page_content='property if anyone who uses your car:\n•\nLeft any removable electronic equipment\nor removable in-car entertainment inside \na locked car where it could be seen.\n•\nLeft any property in an open or convertible\ncar outside of a locked boot or locked \nglove compartment.\n8 We won’t cover loss or damage caused by \ntheft or attempted theft of your car if any \nsecurity device fitted to your car by the \nmanufacturer is not operational when \nyour car is left unattended.\nPage 16\nTheft of car keys \nWe’ll cover your stolen car keys.\nThis cover is included with: \nEssentials\nComprehensive\nComprehensive Plus\nThis cover is not included with: \nThird Party, Fire and Theft\nWhat we’ll do\nWe’ll replace your stolen car keys and the locks \nthey fit, including any locksmith charges.\nYou must take all reasonable steps to protect \nyour car keys from theft.\nIf your keys are stolen, you’ll need to pay the \ntheft excess. You’ll need to report this theft to \nthe police and provide us with the crime \nreference number.', metadata={'source': 'source.pdf'})"
    output_ex = """{
        "question": "What policies is key theft not covered on?",
        "answer": "Key theft is not covered on Third Party, Fire and Theft policies."
    } """

    examples = []
    examples.append({"role": "user", "content": input_ex})
    examples.append({"role": "assistant", "content": output_ex})

    messages = [
        {
            "role": "system",
                    "content": prompt
        }
    ]
    messages.extend(examples)

    messages.append({"role": "user", "content": entry_text})

    initial_response = llm.invoke(
        messages
    )

    question = json.loads(initial_response.content)["question"]
    answer = json.loads(initial_response.content)["answer"]
    # Set up the transformation messages for the API
    t_messages = [{
        "role": "system",
        "content": transform[style]
    }, {"role": "user", "content": question}]

    t_response = llm.invoke(
        t_messages
    )

    return entry_text, entry_index, question, t_response.content, answer


def generate_set(num_questions, llm_purpose):
    results_df = pd.DataFrame(
        columns=["entry_used", "question", "t_question", "correct_answer", "type"])
    n_imp = num_questions // 4
    real = num_questions - n_imp
    for _ in range(n_imp):
        response = llm.invoke(
            input="Write a short question about a random topic").content
        new_row = pd.DataFrame([{
            "entry_used": "",
            "entry_index": -1,
            "question": response,
            "t_question": response,
            "correct_answer": "I am not aware of that topic",
            "type": 3
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    for _ in range(real):
        style = random.randint(0, 2)
        complexity_id = random.randint(0, 3)

        entry_used, entry_index, question, t_question, correct_answer = generate_question(
            llm_purpose, style, complexity_id)

        new_row = pd.DataFrame([{
            "entry_used": entry_used,
            "entry_index": entry_index,
            "question": question,
            "t_question": t_question,
            "correct_answer": correct_answer,
            "type": style
        }])

        results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df
