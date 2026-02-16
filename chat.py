from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain_qdrant import QdrantVectorStore

load_dotenv()

#setting the command to retrieve data

#embeddings:
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")

#retrieving data from vector database
data = QdrantVectorStore.from_existing_collection(
    collection_name="beginning-nodejs",
    embedding=embeddings,
    url="http://localhost:6333"
)

#get user prompt
user_prompt = input("Ask a question about nodejs:")

#get the context:
search_results = data.similarity_search(query = user_prompt)

print(search_results)
context = "\n".join([f"Page Content: {result.page_content}\n Page Number: {result.metadata['page']}\n Source: {result.metadata['source']}\n" for result in search_results])

print(context)

#setting up system prompt
SYSTEM_PROMPT = """
You are a helpful nodejs assistanat. you use the following information provided in the context and answer user's question.
You will answer the questions only based on the context provided. Any question asked outside the context should be answered with "I don't know".

Context:
{context}

Rules:
- Always use the context provided to answer the question
- Answer each question with page number, location if available in the context
- Response should be in json format with the following structure:
{{
    "answer": "your answer here",
    "source": "page number, location or any other source information"
}}

Example 1:
Q: What is the capital of France?
A: I don't know

Example 2:
Q: How to install Nodejs?
A: Node.js now provides 
installers for Windows as well as Mac OS X, and it can be installed in the same way as any other application on these 
platforms (Figure 1-1). You can download Node.js installers from http://nodejs.org/download/.

In the example above, the answer is based on the context provided and includes the source information (Figure 1-1 and the URL).

Output format:
{
    "answer": "your answer here",
    "source": "page number, location or any other source information"
}
"""


#call open ai to answer questions
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role":"system",
        "content": SYSTEM_PROMPT
    },
    {
        "role":"user",
        "content": user_prompt
    }]
)

print(response.choices[0].message.content)
