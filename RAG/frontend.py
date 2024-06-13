import streamlit as st
from streamlit_chat import message
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from RAG import *
from testing import *
from evaluation import *


csv_file = 'settings.csv'
settings = pd.read_csv(csv_file)

# Access the value of var1 (assuming it's in the first row and first column)


# Initialize session state for chat and table
if "messages" not in st.session_state:
    st.session_state.messages = []
if "table" not in st.session_state:
    st.session_state.table = [["Prompt", "Chunk Size", "Recall",
                               "Robustness", "Brevity", "Knowledge Bounding", "Context Matching"]]
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = settings.at[0, 'chunk']
if "prompt" not in st.session_state:
    st.session_state.prompt = settings.at[0, 'prompt']
if "qs" not in st.session_state:
    st.session_state.qs = settings.at[0, 'qs']
if "purp" not in st.session_state:
    st.session_state.purp = settings.at[0, 'purp']

retrieve = setChain(3000, """
  Answer the following question based on the provided context and your internal knowledge.
  Give priority to context and if you are not sure then say you are not aware of topic:

  <context>
  {context}
  </context>

  Question: {input}
  """)
# Function to add a new row to the table


def add_row(prompt, chunk):

    chain = setChain(chunk, prompt)

    # Load the DataFrame from the CSV file
    testset = pd.read_csv('output.csv')

    evaluation = evaluate_llm(chunk, chain, testset)

    csv_file_path = 'eval.csv'
    evaluation.to_csv(csv_file_path, index=False)
    # Filter the DataFrame to include only the specified columns

    df_filtered = evaluation[['Recall', 'Robustness',
                             'Brevity', 'Knowledge Bounding', 'Context Matching']]

    # Compute the sum of each column
    column_means = df_filtered.mean(axis=0)

    means_list = column_means.tolist()

    # Create the final list with "a" and "b" as the first two entries, followed by prompt, chunk, and column sums
    final_list = [prompt, chunk] + means_list
    st.session_state.table.append(final_list)

# Function to handle sending a message


def regen(num_questions, purpose):
    questions = generate_set(num_questions, purpose)
    csv_file_path = 'output.csv'
    questions.to_csv(csv_file_path, index=False)
    st.session_state.table = [st.session_state.table[0]]


def save_settings(prompt, chunk, qs, purp):
    data = [[prompt, chunk, qs, purp]]

# Create a DataFrame from the list
    df = pd.DataFrame(data, columns=['prompt', 'chunk', 'qs', 'purp'])

# Specify the CSV file name
    csv_file = 'settings.csv'

# Write DataFrame to CSV
    df.to_csv(csv_file, index=False)


def send_message():
    if st.session_state.user_input:
        st.session_state.messages.append(
            {"role": "user", "content": st.session_state.user_input})
        response = retrieve.invoke(
            {"input": st.session_state.user_input})["answer"]
        st.session_state.messages.append({"role": "bot", "content": response})
        st.session_state.user_input = ""  # Clear the input after sending


# Chatbot Interface
st.title("Manual Testing")

# Create the text input widget with a callback for when the user presses enter
user_input = st.text_input("You: ", key="user_input", on_change=send_message)

# Display chat messages with unique keys
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")
    else:
        message(msg["content"], key=f"bot_{i}")

# Table Interface

# Top section with two columns
col1, col2 = st.columns(2)

with col1:
    st.title("Model Alteration")
    input1 = st.text_area("Prompt", key="prompt")
    input2 = st.number_input('Chunk size', min_value=100,
                             max_value=3000, step=1, key="chunk_size")
    if st.button("Evaluate Chain"):
        add_row(st.session_state.prompt, st.session_state.chunk_size)

with col2:
    st.title("Generate Testset")
    input3 = int_val = st.slider(
        'Number of questions', min_value=1, max_value=40, step=1, key="qs")
    input4 = st.text_input("Model Purpose", key="purp")
    if st.button("Generate Testset"):
        regen(st.session_state.qs, st.session_state.purp)

if st.button("Save Settings"):
    save_settings(st.session_state.prompt, st.session_state.chunk_size,
                  st.session_state.qs, st.session_state.purp)
st.subheader("Summary Statistics")

# Create a DataFrame from the table data (excluding the header)
df = pd.DataFrame(
    st.session_state.table[1:], columns=st.session_state.table[0])

# Ensure consistent data types for all columns


# Display the table with headers and numbered rows
st.dataframe(df)


# Plot the graph
fig, ax = plt.subplots()
unique_prompts = df['Prompt'].unique()
colors = sns.color_palette('hsv', len(unique_prompts))

for i, prompt in enumerate(unique_prompts):
    prompt_data = df[df['Prompt'] == prompt]
    ax.scatter(prompt_data['Chunk Size'], prompt_data[['Recall', 'Robustness',
                                                      'Brevity', 'Knowledge Bounding', 'Context Matching']].sum(axis=1),
               label=f'Prompt {i+1}', color=colors[i])

ax.set_xlabel('Chunk Size')
ax.set_ylabel('Performance')
ax.set_title('Chunk Size vs. Performance Average')
ax.legend(title='Prompt')

# Display the graph
st.pyplot(fig)
