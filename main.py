from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
parser = StrOutputParser()


def document_loader(url):
  try :
    loader = YoutubeLoader.from_youtube_url(url, add_video_info = False )
    docs = loader.load()
    return docs[0].page_content
  except TranscriptsDisabled:
    print("No captions available for this video.")
s = document_loader(input("Enter Youtube Video URL: "))

if not s:
    raise ValueError("No transcript available, cannot continue.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([s])
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt = PromptTemplate(
    input_variables = ['context', 'question'],
    template = """
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """

)

# question = input("Enter your Query: ")
# retrieved_docs    = retriever.invoke(question)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    "context": RunnablePassthrough() | retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.2)
main_chain = parallel_chain | prompt | llm | parser
query = input("Enter your Query: ")

answer = main_chain.invoke(query)
print(answer)