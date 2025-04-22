# Generative AI Applications
## Introduction:
We are building some Generative AI Applications for creating some new contents.This system will provide a streamlit based user interface for user and gives the response to the user.

Mainly,We have implemented 7 generative ai applications based on some pre-trained model:
#### 1.	Search Engine
#### 2.	Document Q&A APPLICATION
#### 3.	Document Summarizer Application
#### 4. Vision Assistant
#### 5.	Resume Application Tracking System
#### 6.	YouTube Transcript Summary Generator
#### 7.	Health Assistant
This is an end to end LLM project using langchain framework(which is specially useful  for developing applications powered by language model) based on some pretrained open source  LLM models which are:

•	llama3-8b-8192(developed by MetaAI),model type=Chat

•	Gemini-1.5-flash(developed by Google),model type=Chat,Vision and audio



## 1. Search Engine:
### Objective:
  The search engine is a python application that allows us to search any queries or question based on any topics.It will give the response to our queries.
### How It Works:
![image](https://github.com/user-attachments/assets/ec5df313-383e-4c88-9b7d-58f4c00a1c9a)

The application follows these steps to provide responses to our questions:
1.	Input Query Reading: The app reads our given query.
2.	Response Generation: Given query text and an input prompt are passed to the chat model(llama3-8b-8192),which generates a response based on the given query text and the input prompt.



## 2. Document Q&A Application:
### Objective:
  This is a python application that allows us to chat with multiple PDF documents.We can ask questions about the pdfs using natural language,and the application will provide relevant responses based on the 
  context of the documents.This apps utilizes a language model to generate accurate answers to our queries.Please note that the app will only respond to questions related to the loaded PDFs.
### How It Works:
![image](https://github.com/user-attachments/assets/bd0f0d3c-b148-45fc-8a95-db3dd0066384)

The application follows these steps to provide responses to our questions:	
1.	PDF Loading: The app reads multiple PDF documents and extracts their text content.
2.	Converting into Langchain Document:  These extracted texts are converted into langchain document.
3.	Document  Chunking:  The extracted langchain document  is divided into smaller chunks of texts(in the form of langchain document) that can be processed effectively.
4.	Embedding Model: The application utilizes a embedding model(embedding-001) to generate vector representations (embeddings) of those text chunks(which are in the form of langchain document) and store those chunks in a vector database FAISS that is also provided by langchain framework.
5.	Similarity Matching: When we ask a query, the app compares it with the text chunks(which are in the form of langchain document)  and identifies the most semantically similar ones.
6.	Response Generation: The selected chunks are passed to the chat model(llama3-8b-8192), which generates a response based on the input prompt and the relevant content of the PDFs.



## 3. Document Summarizer Application:
### Objective:
  This is a python application that allows us to summarize multiple PDF documents simultaneously.
### How It Works:
![image](https://github.com/user-attachments/assets/03031372-e8e0-42c7-a041-092b757aff22)

The application follows these steps to provide summary of our uploaded pdfs:
1.	PDF Loading: The app reads multiple PDF documents and extracts their text content.
2.	Text Chunking: The extracted text is divided into smaller chunks of text that can be processed effectively.
3.	Response Generation: The text chunks are passed to the chat model(llama3-8b-8192), which generates the summary responses one by one based on the number of text chunks and the input prompt.



## 4. Vision Assistant:
### Objective:
  This is a python application that allows us to detect the every articles in an image.
### How It Works:
![image](https://github.com/user-attachments/assets/6df16b92-a781-43e6-8eb0-e1f259d9c01f)

The application follows these steps to detect the image:
1.	Image Loading: The app reads the uploaded image and coverts into base64 encoded string.
2.	Human Message Creation: Langchain provides a human message function,that combines the base64 encoded string and the text or command for describing the uploaded image into a single message.
3.	Response Generation: Created message is passed into the large language model(gemini-1.5-flash) which generates the text about the every articles in the uploaded image.



## 5. Resume Application Tracking System:
### Objective:
  This is a python application that allows us to track our resume or cv based on any job description i.e. how much our resume or cv is eligible for a given job description.
### How It Works:
![image](https://github.com/user-attachments/assets/61c349ed-ecf9-4dc1-9cda-e6939d83868e)

The application follows these steps to track our resume or cv based on the given job description:
1.	Reading Job Description: The app reads the job description in the form of text.
2.	PDF Loading: The app reads our resume or cv PDF documents and extracts their text content.
3.	Response Generation: The extracted texts from the pdf document and the job description texts are passed to the chat model(llama3-8b-8192), which generates the response in a particular structured way based on the input prompt.



## 6. YouTube Transcript Summary Generator:
### Objective:
This is a python application that allows us to generate the summarized transcription for a you tube video.
### How It Works:
![image](https://github.com/user-attachments/assets/5b8e7f15-e3ae-481d-8cfb-7d63aa15f313)

The application follows these steps to generate summarized transcription based on the you tube video:
1.	Reading YouTube video URL : The app reads the You Tube Video URL in the form of text.
2.	Extracting the Video ID and Generating the Transcript Text: Extract the video id from the text url and passed into the youtube_transcript _api function to generate the full transcript text of the you tube video.
3.	Response Generation: Extracted transcript text is passed into the chat model(llama3-8b-8192),which generates the summarized transcripted text in a structured way based on the input prompt.



## 7. Health Assistant:
### Objective:
  This is a python application that allows us to generate the total calories of a given food image and also tells us whether the food is healthy or not.
### How It Works:
![image](https://github.com/user-attachments/assets/eb63a8b2-13d6-4b49-9030-8f449eabd088)

The application follows these steps to generate the total calories of a given food image:
1.	Food Image Loading: The app reads the uploaded food image and coverts into base64 encoded string.
2.	Human Message Creation: Langchain provides a human message function,that combines the base64 encoded string and the text or command for determining the calories of the uploaded food image into a single message.
3.	Response Generation: Created message is passed into the language model(Gemini-1.5-flash) which generates the response about the calories and the healthiness of the uploaded food image.



# Dependencies and Installation

1. Navigate to the project directory:

```bash
  cd genai app
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

- [Langchain Documentation](https://python.langchain.com/v0.2/docs/introduction/)
