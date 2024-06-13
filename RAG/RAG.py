from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import fitz
import openai
# embeddings_llm = OllamaEmbeddings(model="all-minilm")
# llm = Ollama(model="tinyllama", base_url = 'http://localhost:11434')
api_key = ''
openai.api_key = api_key


embeddings_llm = OpenAIEmbeddings(
    model="text-embedding-3-small", api_key=api_key)
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)


def load_pdf_content(filename):
    doc = fitz.open(filename)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


def setChain(chunkSize, prompt_input):
    # Define the local filename
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
    vector_index = FAISS.from_documents(documents, embeddings_llm)

    retriever = vector_index.as_retriever()

    prompt = ChatPromptTemplate.from_template(prompt_input)

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain
