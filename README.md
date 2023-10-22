
# Chat PDF App 

A chatbot application that uses OpenAI's language model to answer questions about PDF documents

## Technologies Used
  - Streamlit: The user interface is built with Streamlit, making it easy to create web apps with Python.

  - LangChain: LangChain is used for text processing, text splitting, and handling embeddings.

  - OpenAI LLM: The application integrates with OpenAI's powerful LLM model for natural language processing and understanding.


## Features
  - Document Upload: Users can upload PDF documents, and the app extracts text from them for further analysis.

 - Text Chunking: The extracted text is divided into manageable text chunks, allowing for efficient processing.

 - OpenAI Embeddings: The app leverages OpenAI's GPT-3-powered embeddings to represent the text data for more meaningful interactions.

 - Question-Answering: Users can ask questions related to the content of the uploaded documents, and the chatbot provides answers based on the document's contents.

 - Easy Setup: With clear instructions and built-in settings, deploying and using the LLM Chat App is straightforward.


## Getting Started
Follow these steps to set up and run the LLM Chat App on your own system:

1. Clone this repository using the following command:
   ```sh
   git clone https://github.com/Mayuresh2703/Chat-PDF-App

2. Install the required Python packages using:
    ```sh
    pip install -r requirements.txt

3. Set up your OpenAI API key:

 - Visit the OpenAI Platform to create an account or log in if you already have one.
 - Follow the instructions on the OpenAI platform to obtain your API key. You can find detailed documentation on how to create an API key in the OpenAI API Documentation.
 - Once you have your API key, set it as an environment variable named OPENAI_API_KEY in your environment or in the application as specified in the project's setup instructions.

4. Run the app using:
    ```sh
    streamlit run app.py


## Usage
 - Upload a PDF document.

 - Ask questions related to the document.

 - The chatbot will provide answers based on the document's content.

## OpenAI API Key

This project relies on the OpenAI API for natural language processing and understanding. To use this application, you need to obtain an OpenAI API key.

1. Visit the [OpenAI Platform](https://platform.openai.com/) to create an account or log in if you already have one.

2. Follow the instructions on the OpenAI platform to obtain your API key. You can find detailed documentation on how to create an API key in the [OpenAI API Documentation](https://platform.openai.com/docs/guides/authentication).

3. Once you have your API key, set it as an environment variable named `OPENAI_API_KEY` in your environment or in the application as specified in the project's [setup instructions](#getting-started).

Please keep your API key confidential and do not share it publicly. If you plan to collaborate on this project with others, consider using environment variables or other secure methods for key management.

## Deployment on AWS

This application is hosted on Amazon Web Services (AWS). You can  refer the gif video by visiting the following URL: GIF:(https://www.veed.io/view/3408e84c-a536-4cb6-bd73-3f184d773539)

## Video Demo

To get a detailed demonstration of how this project works, watch our video  on YouTube:

[Watch the Video Demo on YouTube](https://youtu.be/B2T6hgMn_y8)

Don't forget to like and subscribe to our YouTube channel for more exciting content!



## Acknowledgements

 - [How to Build a ChatGPT-Powered PDF Assistant with Langchain and Streamlit | Step-by-Step Tutorial ](https://www.youtube.com/watch?v=RIWbalZ7sTo)
 - [How to Deploy GPT-3 Streamlit App on AWS EC2](https://www.youtube.com/watch?v=904cW9lJ7LQ)

