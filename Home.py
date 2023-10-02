import utils
import streamlit as st
from streaming import StreamHandler
import toml

from langchain.llms import OpenAI
from langchain.chains import ConversationChain,ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


st.set_page_config(page_title="Banana IA ChatBot", page_icon="⭐")
st.header('Banana IA ChatBot')
st.write('Hola. Soy el nuevo asistente virtual de Banana Computer. Si me haces las preguntas adecuadas podré darte información de los productos que tenemos en nuestro catálogo y asesorarte sobre como realizar tus compras en nuestra web.')

class ContextChatbot:

    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"

    def getTemplate(self):
        general_system_template = r""" 
        Eres un amable asistente comercial para los productos de marca Apple en una tienda en línea de Banana Computer. Utilizas un tono profesional, pero respondes de manera alegre. Sé conciso y muestra los precios en euro (€) en formato europeo ( con el símbolo detrás de la cifra ). Prioriza los productos en oferta. Si mencionas un producto con un enlace, puedes proporcionar el enlace del producto al cliente. Si no sabes la respuesta, di que no sabes. Habla en español.
        Nunca digas el stock que hay de un producto.
        ----
        {context}
        ----
        """
        general_user_template = "{question}"
        messages = [
                    SystemMessagePromptTemplate.from_template(general_system_template),
                    HumanMessagePromptTemplate.from_template(general_user_template)
        ]
        qa_prompt = ChatPromptTemplate.from_messages( messages )

        return qa_prompt

    def loadCSV(self):
        from urllib import request

        config = toml.load(".streamlit/config.toml")
        remote_url = config["data"]["BANANA_REMOTE_URL"]
        local_file = 'catalogo.csv'
        request.urlretrieve(remote_url, local_file)

        loader = CSVLoader('./' + local_file, csv_args={
            "delimiter": ";",
            "quotechar": '|',
            'fieldnames': ['codigo','name', 'precio', 'breadcrumb','descripcion','stock','enlace','en_oferta','precio_anterior']
        }, source_column="name")

        data = loader.load()

        return data

    @st.cache_resource
    def setup_chain(_self):

        qa_prompt = _self.getTemplate()
        data = _self.loadCSV()
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        llm = OpenAI(model_name=_self.openai_model, temperature=0)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(data, embeddings)
        retriever = vectorstore.as_retriever()

        #chain = ConversationChain(llm=llm, memory=memory, verbose=True)
        #chain = ConversationalRetrievalChain.from_llm(llm,retriever,memory=memory,verbose=True, combine_docs_chain_kwargs={'prompt': qa_prompt})
        chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True, combine_docs_chain_kwargs={'prompt': qa_prompt})

        return chain
    
    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Pregúntame lo que desees!")
        if user_query:
            utils.display_msg(user_query, 'user')
            
            st_cb = StreamHandler(st.empty())
            response = chain.run(user_query, callbacks=[st_cb])
            utils.display_msg(response, 'assistant')

if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()
