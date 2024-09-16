import os
import google.generativeai as genai
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import GoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory

class CustomerSupportQA:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = self.create_vectorstore()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa = ConversationalRetrievalChain.from_llm(
            GoogleGenerativeAI(model="gemini-pro", temperature=0),
            self.vectorstore.as_retriever(),
            memory=self.memory
        )

    def create_vectorstore(self):
        knowledge_base_dir = "knowledge_base"
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = []

        for filename in os.listdir(knowledge_base_dir):
            loader = TextLoader(os.path.join(knowledge_base_dir, filename))
            documents = loader.load()
            texts.extend(text_splitter.split_documents(documents))

        return FAISS.from_documents(texts, self.embeddings)

    def get_response(self, query):
        result = self.qa({"question": query})
        return result['answer']

customer_support_qa = CustomerSupportQA()