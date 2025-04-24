import streamlit as st
from streamlit_option_menu import option_menu
from  PyPDF2 import PdfReader
import PyPDF2 as pdf
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  ##vectorstore db
from langchain_google_genai import GoogleGenerativeAIEmbeddings  ##Vector Embedding techniques
from langchain_google_genai import ChatGoogleGenerativeAI
from youtube_transcript_api import YouTubeTranscriptApi
from PIL import Image
import io
import base64
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()     ## load all the environment variables

##------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##load the GROQ and Google API Key from the .env file
groq_api_key=os.getenv('GROQ_API_KEY')
llm=ChatGroq(groq_api_key=groq_api_key,model_name='llama3-8b-8192')

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
## PDF Q&A Application

# prompt  template for pdf Q&A app
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate answer based on the question,if the respone is not in provided context just say,"Answer is not available in the context",don't provide the wrong answer. 
<context>
{context}
<context>
Questions:{input}
Answer:
"""
)


def vector_embedding(Docs):
    if 'vectors' not in st.session_state:
        os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")   ##vector embedding layer
        st.session_state.docs=Docs
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.docs_chunks=st.session_state.text_splitter.split_documents(st.session_state.docs)  ##splitting the documents
        st.session_state.vectors=FAISS.from_documents(st.session_state.docs_chunks,st.session_state.embeddings)   # storing the documents chunks
        st.write('Vector Store DB IS READY')  


def pdfs_qa_app():
    st.title("DOCUMENT Q&A")
    uploaded_file=st.file_uploader("Upload Your PDF Files",type="pdf",help="Please upload the pdf",accept_multiple_files=True,key="qa_pdf_uploader")
    submit1 = st.button("Documents Embedding")
    
    if submit1:
        if uploaded_file:
            Docs=[]
            for pdf in uploaded_file:
                pdf_reader=PdfReader(pdf)
                for page in pdf_reader.pages:
                    Docs.append(Document(page_content=str(page.extract_text())))
            vector_embedding(Docs)        
        else:
            raise FileNotFoundError("No file uploaded")
    else:
        st.write("Please upload the documents.")
            
            
    query=st.text_input("What do you want to ask from the document..")
    submit2=st.button('Submit')
    if submit2:
        if query:
            document_chain=create_stuff_documents_chain(llm,prompt)
            retriever=st.session_state.vectors.as_retriever()
            retrieval_chain=create_retrieval_chain(retriever,document_chain)
            response=retrieval_chain.invoke({'input':query})
            st.write(response['answer'])
            # With a streamlit expander
            with st.expander("Document Similarity Search"):
                #Find the relevant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        else:
            st.write("Please paste the question what you want to ask from the document")

## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## PDF SUMMARIZER

# Prompt template for text summarization
prompt_summarizer=PromptTemplate.from_template("""You are a text summarizer.Write a concise and short summary of the following speech.
Speech: {text}

 """)


def generate_summary(text,prompt):
    chain=prompt|llm
    return chain.invoke({"text":text})


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def pdf_summarizer():
    st.title("DOCUMENT SUMMARIZER")
    uploaded_file=st.file_uploader("Upload Your PDF Files",type="pdf",help="Please upload the pdf",accept_multiple_files=True,key="summarizer_pdf_uploader")
    submit=st.button("Submit")
    if submit:
        if uploaded_file:
            raw_text=get_pdf_text(uploaded_file)
            chunk_text=get_text_chunks(raw_text)
            for text in chunk_text:
                st.write((generate_summary(text,prompt_summarizer)).content)
        else:
            raise FileNotFoundError("No file uploaded")

## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Conversational Q&A Chatbot

def get_chatmodel_response(question):
    if question:
        st.session_state['flowmessages'].append(HumanMessage(content=question))
        response = llm.invoke(st.session_state['flowmessages'])
        st.session_state['flowmessages'].append(AIMessage(content=response.content))
    return ""


def chatbot():
    st.title('Conversational Q&A Chatbot')
    st.header("Hey, Let's Chat")
    
    # Initialize chat history
    if 'flowmessages' not in st.session_state:
        st.session_state.flowmessages = [
            SystemMessage(content="You are an AI assistant")
        ]
    
    # Display chat messages - identical to chat_input() style
    for message in st.session_state.flowmessages:
        if isinstance(message, SystemMessage):
            continue
        
        # Use native chat message styling
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.write(message.content)
    
    # Create identical-looking chat input without using chat_input()
    chat_container = st.empty()
    user_input = chat_container.text_input(
        "Type your message...", 
        key=f"chat_input_{len(st.session_state.flowmessages)}",
        label_visibility="collapsed",
        placeholder="Type a message..."
    )
    
    # Process input on enter key or button press
    if user_input:
        get_chatmodel_response(user_input)
        
        # Clear input and rerun to refresh
        chat_container.text_input(
            "Type your message...", 
            value="",
            key=f"chat_input_{len(st.session_state.flowmessages)}_clear",
            label_visibility="collapsed",
            placeholder="Type a message..."
        )
        st.rerun()

## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Resume ATS

prompt_resume=PromptTemplate.from_template("""
Hey Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of tech field,software engineering,data science ,data analyst
and big data engineer. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide 
best assistance for improving thr resumes. Assign the percentage Matching based 
on Job description and
the missing keywords with high accuracy
resume:{text}
description:{jd}

I want the response in three different single string having the structure
{{"Job Description Match":"%",
   "Missing Keywords":"[]",
   "Profile Summary":""
}}

""")

def get_repsonse(prompt,text,jd):
    chain=prompt | llm
    return chain.invoke({'text':text,'jd':jd})

def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

def resume_ATS():
    st.title("RESUME APPLICATION TRACKING SYSTEM")
    st.header("Improve Your Resume")
    jd=st.text_area("Paste the Job Description")
    uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="Please uplaod the pdf")
    submit = st.button("Submit")
    if submit:
        if uploaded_file:
            if jd:
                text=input_pdf_text(uploaded_file)
                response=get_repsonse(prompt_resume,text,jd)
                st.subheader(response.content)
            else:
                st.write("Please paste the Job Description")
        else:
            raise FileNotFoundError("No file uploaded")

## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## YouTube Trancript Generator

prompt_transcript=PromptTemplate.from_template("""You are Yotube video summarizer. You will be taking the transcript text: {transcript_text}
and summarizing the entire video and providing the important summary in points
within 250 words. Please provide the summary of the text given here:  """)

## getting the summary based on Prompt
def generate_content(prompt,transcript_text):
    chain=prompt|llm
    return chain.invoke({'transcript_text':transcript_text})

## getting the transcript data from yt videos
def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e

def yt_transcript():
    st.title("YOUTUBE TRANSCRIPT SUMMARY GENERATOR")
    youtube_link = st.text_input("Enter YouTube Video Link:")
    if youtube_link:
        video_id = youtube_link.split("=")[1]
        print(video_id)
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)
    
    if youtube_link:
        if st.button("Get Detailed Notes"):
            transcript_text=extract_transcript_details(youtube_link)
            if transcript_text:
                summary=generate_content(prompt_transcript,transcript_text)
                st.markdown("## Detailed Notes:")
                st.write(summary.content)
    else:
        st.write("Give the valid you tube link.")

##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Health Management Application

def get_gemini_repsonse(image_b64):
    os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")
    response = llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "You are an expert in nutritionist where you need to see the food items from the image and calculate the total calories, also provide the details of every food items with calories in the following format: \n 1. Item 1 - no of calories \n 2. Item 2 - no of calories \n ---- \n ---- \n Finally you can also mention whether the food is healthy or not"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        ]
                        )
                        ]
                        )
    return response


def health():
    st.title("HEALTH ASSISTANT")
    uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
    submit=st.button('Tell me about the total calories')
    if submit:
        if uploaded_file is not None:
            byteImgIO = io.BytesIO()
            image = Image.open(uploaded_file)
            image.save(byteImgIO, "PNG")
            byteImgIO.seek(0)
            byteImg = byteImgIO.read()
            st.image(image, caption="Uploaded Image.", use_container_width=True)
            image_b64 = base64.b64encode(byteImg).decode("utf-8")
            response=get_gemini_repsonse(image_b64)
            st.write(response.content)
        else:
            raise FileNotFoundError("No file uploaded")

##------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Image Detector

def generate_response(image_b64):
    os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")
    response = llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        ]
                        )
        ]
    )
    return response

def img_detector():
    st.title("VISION ASSISTANT")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    submit=st.button('Detect the image')
    if submit:
        if uploaded_file is not None:
            byteImgIO = io.BytesIO()
            image = Image.open(uploaded_file)
            image.save(byteImgIO, "PNG")
            byteImgIO.seek(0)
            byteImg = byteImgIO.read()
            st.image(image, caption="Uploaded Image.", use_container_width=True)
            image_b64 = base64.b64encode(byteImg).decode("utf-8")
            response=generate_response(image_b64)
            st.write(response.content)
        else:
            raise FileNotFoundError("No file uploaded")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  # Language and Sentiment Analyzer
prompt_analyze=PromptTemplate.from_template(''' You are an advanced language model specializing in comprehensive sentiment and language analysis. Your task is to analyze the input {text} and provide two distinct outputs:
1. Sentiment Analysis:
               Identify the sentiment expressed in the text. Categorize the sentiment using the following specific emotion labels:
                                               Joy/Happiness
                                               Sadness
                                               Anger
                                               Fear
                                               Surprise
                                               Disgust
                                               Frustration
                                               Excitement
                                               Anxiety
                                               Hope
               If the text expresses a mix of emotions, identify the most prominent one.
               If the text is neutral or objective, state Sentiment-Neutral.
2. Language Analysis:
               Identify the current language of the input and provide translated text into English and original full name of the language of the input.''')

def Analyzer():
    st.title("Language and Sentiment Analyzer")
    input_text=st.text_input("Enter the text:")
    analyze= st.button("Analyze")
    if analyze:
        chain=prompt_analyze | llm
        response=chain.invoke({'text':input_text})
        st.write(response.content)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Streamlit app
st.set_page_config(layout='wide',page_title="GENERATIVE AI APPLICATIONS")
st.title("GENERATIVE AI APPLICATIONS")
selected=option_menu(
    menu_title="Menu",
    options=["Conversational Q&A","Language and Sentiment Analyzer","Document Q&A","Document Summarizer","Vision Assistant","Resume Application Tracking System","YouTube Transcript Summary Generator","Health Assistant"],
    icons=["app","app","app","app","app","app","app","app"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

if selected == "Document Q&A":
    pdfs_qa_app()
if selected=="Document Summarizer":
    pdf_summarizer()
if selected=="Resume Application Tracking System":
    resume_ATS()
if selected=="YouTube Transcript Summary Generator":
    yt_transcript()
if selected=="Health Assistant":
    health()
if selected=="Vision Assistant":
    img_detector()
if selected=="Language and Sentiment Analyzer":
    Analyzer()
if selected=="Conversational Q&A":
    chatbot()
if selected != "Conversational Q&A":
    if 'flowmessages' in st.session_state:
        del st.session_state['flowmessages']
        st.rerun()

if selected != "Document Q&A":
    vector_db_keys = ['vectors', 'embeddings', 'docs', 'text_splitter', 'docs_chunks']
    for key in vector_db_keys:
        if key in st.session_state:
            del st.session_state[key]