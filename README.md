# Generative AI Applications
## Introduction:
We are building some Generative AI Applications for creating some new contents.This system will provide a streamlit based user interface for user and gives the response to the user.

Mainly,We have implemented some generative ai applications based on some pre-trained models:
#### 1.	Conversational Q&A
#### 2. Language and Sentiment Analyzer
#### 3.	Document Q&A
#### 4.	Document Summarizer
#### 5. Vision Assistant
#### 6.	Resume Application Tracking System
#### 7.	YouTube Transcript Summary Generator
#### 8.	Health Assistant
This is an end to end LLM project using langchain framework(which is specially useful  for developing applications powered by language model) based on some pretrained open source  LLM models which are:

•	llama3-8b-8192(developed by MetaAI),model type=Chat

•	Gemini-1.5-flash(developed by Google),model type=Chat,Vision and audio



## 1. Conversational Q&A:
### Objective:
  This is a python application that allows us to search any queries or question based on any topics.It will give the response to our queries and it also remember the previous conversation in a particular session.
### How It Works:
![conversational Q A](https://github.com/user-attachments/assets/1edbd5e0-57a7-4021-b650-1bef8caccfac)

The application follows these steps to provide responses to our questions:
1.	Input Query Reading: The app reads our given query.
2.	Response Generation: Given query text and the past conversations are passed to the chat model(llama3-8b-8192),which generates a response based on the given query text and the past conversation.


## 2. Language and Sentiment Analyzer:
### Objective:
  This is a python application that allows us to analyze any text in any language and convert the given text to english and also analyze the sentiment in english.
### How It Works:
![WhatsApp Image 2025-04-23 at 02 09 10_81d0af22](https://github.com/user-attachments/assets/e7885b8b-7e7a-4fd3-b16f-36d9c2e3e468)

The application follows these steps to provide responses based on the input_text:
1.	Input text Reading: The app reads our given given input text.
2.	Response Generation: Given input text and a prompt template are passed to the chat model(llama3-8b-8192),which generates a response based on the prompt template.





## 3. Document Q&A:
### Objective:
  This is a python application that allows us to chat with multiple PDF documents.We can ask questions about the pdfs using natural language,and the application will provide relevant responses based on the 
  context of the documents.This apps utilizes a language model to generate accurate answers to our queries.Please note that the app will only respond to questions related to the loaded PDFs.
### How It Works:
![WhatsApp Image 2024-07-22 at 16 36 14_e57112d5](https://github.com/user-attachments/assets/2bf2ee99-5523-458b-a972-f1e70638af13)

The application follows these steps to provide responses to our questions:	
1.	PDF Loading: The app reads multiple PDF documents and extracts their text content.
2.	Converting into Langchain Document:  These extracted texts are converted into langchain document.
3.	Document  Chunking:  The extracted langchain document  is divided into smaller chunks of texts(in the form of langchain document) that can be processed effectively.
4.	Embedding Model: The application utilizes a embedding model(embedding-001) to generate vector representations (embeddings) of those text chunks(which are in the form of langchain document) and store those chunks in a vector database FAISS that is also provided by langchain framework.
5.	Similarity Matching: When we ask a query, the app compares it with the text chunks(which are in the form of langchain document)  and identifies the most semantically similar ones.
6.	Response Generation: The selected chunks are passed to the chat model(llama3-8b-8192), which generates a response based on the input prompt and the relevant content of the PDFs.



## 4. Document Summarizer:
### Objective:
  This is a python application that allows us to summarize multiple PDF documents simultaneously.
### How It Works:
![WhatsApp Image 2024-07-22 at 16 36 14_666eb5cb](https://github.com/user-attachments/assets/a043834c-7f56-4540-80b1-78cd526d7fd5)

The application follows these steps to provide summary of our uploaded pdfs:
1.	PDF Loading: The app reads multiple PDF documents and extracts their text content.
2.	Text Chunking: The extracted text is divided into smaller chunks of text that can be processed effectively.
3.	Response Generation: The text chunks are passed to the chat model(llama3-8b-8192), which generates the summary responses one by one based on the number of text chunks and the input prompt.



## 5. Vision Assistant:
### Objective:
  This is a python application that allows us to detect the every articles in an image.
### How It Works:
![WhatsApp Image 2024-07-25 at 23 44 07_49c9a204](https://github.com/user-attachments/assets/7fb297f6-0052-4971-b30c-407d5b49d9a4)

The application follows these steps to detect the image:
1.	Image Loading: The app reads the uploaded image and coverts into base64 encoded string.
2.	Human Message Creation: Langchain provides a human message function,that combines the base64 encoded string and the text or command for describing the uploaded image into a single message.
3.	Response Generation: Created message is passed into the large language model(gemini-1.5-flash) which generates the text about the every articles in the uploaded image.



## 6. Resume Application Tracking System:
### Objective:
  This is a python application that allows us to track our resume or cv based on any job description i.e. how much our resume or cv is eligible for a given job description.
### How It Works:
![WhatsApp Image 2024-07-22 at 16 36 13_25ea90a9](https://github.com/user-attachments/assets/022ed932-da25-48fe-9c8f-d6769e8b23a6)

The application follows these steps to track our resume or cv based on the given job description:
1.	Reading Job Description: The app reads the job description in the form of text.
2.	PDF Loading: The app reads our resume or cv PDF documents and extracts their text content.
3.	Response Generation: The extracted texts from the pdf document and the job description texts are passed to the chat model(llama3-8b-8192), which generates the response in a particular structured way based on the input prompt.



## 7. YouTube Transcript Summary Generator:
### Objective:
This is a python application that allows us to generate the summarized transcription for a you tube video.
### How It Works:
![WhatsApp Image 2024-07-22 at 16 36 13_3fc10be2](https://github.com/user-attachments/assets/9d8986c0-afca-411a-9ebd-e68704c63dba)

The application follows these steps to generate summarized transcription based on the you tube video:
1.	Reading YouTube video URL : The app reads the You Tube Video URL in the form of text.
2.	Extracting the Video ID and Generating the Transcript Text: Extract the video id from the text url and passed into the youtube_transcript _api function to generate the full transcript text of the you tube video.
3.	Response Generation: Extracted transcript text is passed into the chat model(llama3-8b-8192),which generates the summarized transcripted text in a structured way based on the input prompt.



## 8. Health Assistant:
### Objective:
  This is a python application that allows us to generate the total calories of a given food image and also tells us whether the food is healthy or not.
### How It Works:
![WhatsApp Image 2024-07-22 at 16 36 13_81743ad4](https://github.com/user-attachments/assets/e56930fc-0e95-49b4-b543-daa941d1368c)

The application follows these steps to generate the total calories of a given food image:
1.	Food Image Loading: The app reads the uploaded food image and coverts into base64 encoded string.
2.	Human Message Creation: Langchain provides a human message function,that combines the base64 encoded string and the text or command for determining the calories of the uploaded food image into a single message.
3.	Response Generation: Created message is passed into the language model(Gemini-1.5-flash) which generates the response about the calories and the healthiness of the uploaded food image.



# Dependencies and Installation

1. Navigate to the project directory:

```bash
  cd Generative-AI
```
2. Activate the conda environment:
```bash
  activate condaenv
```
3. Create a virtual environment in your local machine using:

```bash
  conda create -p venv python==3.10 -y
```
4. Activate the virtual environment:
```bash
  conda activate venv/
```
5. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
6. Acquire these api keys through [GroqCloud](https://console.groq.com/playground),[Google AI Studio](https://aistudio.google.com/app/apikey), then put them in .env file and keep this file in secret

```bash
  GROQ_API_KEY="your_api_key_here"
  GOOGLE_API_KEY="your_api_key_here"
```

# Usage
1. Ensure that we have installed the required dependencies.
2. Run app.py by executing:
```bash
streamlit run app.py

```
3. The application will launch in our default browser,displaying the user interface.

## References

[![LangChain](https://img.shields.io/badge/LangChain-02E6B8?style=for-the-badge&logo=langchain&logoColor=black)](https://python.langchain.com/docs/introduction/)

## Deployment
This application is deployed with streamlit: 

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://generative-ai-djtdny7kvpwanggwngfcme.streamlit.app/)
