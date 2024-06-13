import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from RAG import *


def calculate_brevity(original_answer, query_answer):
    original_length = len(original_answer.split())
    query_length = len(query_answer.split())

    if query_length <= original_length:
        return 1
    elif query_length >= 3 * original_length:
        return 0
    else:
        return 1-(query_length - original_length) / (2 * original_length)

# Function to calculate the recall score using OpenAI embeddings


def calculate_sim(original_embedding, query_answer):

    query_embedding = embeddings_llm.embed_query(query_answer)

    # Calculate cosine similarity
    similarity = cosine_similarity(
        np.array(original_embedding).reshape(1, -1),  np.array(query_embedding).reshape(1, -1))
    return similarity[0][0]


def mostly_identical(chunkSize, llm_texts, text2_index):
    if text2_index == -1.0:
        return True
    for i in llm_texts:
        print(abs(int(i.metadata["section_index"]) *
              int(chunkSize) - float(text2_index) * 2500))
        if abs(int(i.metadata["section_index"]) * int(chunkSize) - float(text2_index) * 2500) < 5000:
            return True
    return False


def evaluate_llm(chunkSize, chain, df):
    llm_answers = []
    llm_t_answers = []
    llm_contexts = []
    brevity_scores = []
    recall_scores = []
    robustness_scores = []
    limitation_scores = []
    context_matching_scores = []

    for index, row in df.iterrows():
        simple_answer = chain.invoke({"input": str(row["question"])})
        comp_answer = chain.invoke({"input": str(row["t_question"])})
        style = str(row["type"])

        original_answer = str(row["correct_answer"])

        o_context = str(row["entry_used"])

        contents = [doc.page_content for doc in simple_answer["context"]]

    # Concatenate all contents into a single string
        concatenated_content = ''.join(contents)
        llm_context = concatenated_content

        original_embedding = embeddings_llm.embed_query(original_answer)
        idk_embedding = embeddings_llm.embed_query(
            "I am not aware of that topic")

        brevity_score = calculate_brevity(
            row['correct_answer'], simple_answer["answer"])
        recall_score = calculate_sim(
            original_embedding, simple_answer["answer"])
        robustness_score = calculate_sim(
            original_embedding, comp_answer["answer"])

        print("")
        print((calculate_sim(idk_embedding, simple_answer["answer"]) > .7))

        if ((calculate_sim(idk_embedding, comp_answer["answer"]) > .7) ^ (style == '3')):
            limitation_score = 0
        else:
            limitation_score = 1

        context_matching_score = 1 if mostly_identical(
            chunkSize, simple_answer["context"], row["entry_index"]) else 0
        llm_answers.append(simple_answer["answer"])
        llm_t_answers.append(comp_answer["answer"])
        llm_contexts.append(llm_context)
        brevity_scores.append(brevity_score)
        recall_scores.append(recall_score)
        robustness_scores.append(robustness_score)
        limitation_scores.append(limitation_score)
        context_matching_scores.append(context_matching_score)

    # Add the calculated scores to the DataFrame
    df['LLM Answers'] = llm_answers
    df['LLM T Answers'] = llm_t_answers
    df['LLM Contexts'] = llm_contexts

    df['Brevity'] = brevity_scores
    df['Recall'] = recall_scores
    df['Robustness'] = robustness_scores
    df['Knowledge Bounding'] = limitation_scores
    df['Context Matching'] = context_matching_scores

    return df
