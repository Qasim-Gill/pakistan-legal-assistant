# **Pakistan Legal Assistant 🇵🇰**

## **About the Project**
The **Pakistan Legal Assistant** is an AI-powered chatbot designed to answer queries related to Pakistani laws. It provides accurate, detailed, and multilingual responses in **English** and **Urdu**, making legal information accessible to everyone.

This project was developed as part of the **ULEFUSA - APTECH Generative AI Training Hackathon (Jan-Feb 2025)**.

---

## **Features**
- **Multilingual Support**: Answers legal questions in both **English** and **Urdu**.
- **Context-Aware Responses**: Retrieves relevant legal documents and provides structured answers.
- **User-Friendly Interface**: Simple and intuitive chat interface powered by **Gradio**.
- **Fast and Accurate**: Uses state-of-the-art AI models for quick and reliable responses.

---

## **How It Works**
1. **User Query**: The user asks a legal question in English or Urdu.
2. **Language Detection**: The system detects the language of the query.
3. **Document Retrieval**: Relevant legal documents are retrieved from a vector store.
4. **Response Generation**: The system generates a detailed and structured response in the same language as the query.
5. **Output Delivery**: The response is displayed in the chat interface.

---

## **Technologies Used**
Here’s a breakdown of the technologies and models used in this project and their roles:

| **Technology/Model**                     | **Role**                                                                 |
|------------------------------------------|-------------------------------------------------------------------------|
| **Python**                               | The primary programming language used for the entire project.           |
| **Gradio**                               | Used to create the user-friendly chat interface.                        |
| **FAISS**                                | Efficient vector store for document retrieval and similarity search.    |
| **Llama3-8b-8192 (Groq API)**            | Large Language Model (LLM) used for generating detailed responses.      |
| **Helsinki-NLP/opus-mt-en-ur**           | Translation model for converting English context into Urdu.             |
| **Multilingual MiniLM-L12-v2**           | Multilingual embeddings model for understanding and retrieving text.    |
| **LangChain**                            | Framework for retrieval-augmented generation (RAG) and prompt engineering. |
| **Hugging Face Transformers**            | Library for multilingual embeddings and translation models.             |
| **PyPDF2**                               | Used to extract text from PDF documents.                                |

---

## **Live Demo**
Try the **Pakistan Legal Assistant** live:  
👉 [Live Demo on Hugging Face](https://huggingface.co/spaces/qasimgill/pakistan-laws-chatbot)

---

## **Project Slides**
Check out the project slides for a detailed overview:  
📑 [Slides on Canva](https://www.canva.com/design/DAGf6Tcfxlw/Ol3DBn87apFvkWD8_HQ1nw/edit)

---

## **Demo Video**
Watch the demo video to see the project in action:  
🎥 [Demo Video on Google Drive](https://drive.google.com/file/d/1epTrewdQRgtVKX0ktTSIfDgMXtgthEn9/view?usp=sharing)

---

## **Installation**
To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/pakistan-legal-assistant.git
   cd pakistan-legal-assistant
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Create a `.env` file and add your Groq API key:
     ```plaintext
     GROQ_API_KEY=your_groq_api_key_here
     ```

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the App**:
   - Open your browser and go to `http://127.0.0.1:7860`.

---

## **Usage**
1. Enter your legal question in the chat interface.
2. The system will retrieve relevant legal documents and generate a detailed response.
3. Ask follow-up questions or explore example queries provided.

---

## **Example Queries**
- **English**: "What is the punishment for theft?"
- **Urdu**: "چوری کی سزا کیا ہے؟"
- **English**: "Explain Section 302 of PPC."
- **Urdu**: "PECA 2016 کے تحت سائبر کرائم کی سزائیں؟"

---

## **Contributors**
- **Ali Rayan** - Project Lead
- **Muhammad Qasim** - Backend Developer
- **Yousuf** - Backend Developer
- **Alisha Tariq** - Slides and Video

---

## **License**
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
- **ULEFUSA - APTECH - ICodeGuru - ASPIRE PAKISTAN - UET** for hosting the Generative AI Training Hackathon.
- **Hugging Face** for providing the platform to deploy the app.
- **Groq** for the fast and efficient language model API.

---

## **Contact**
For questions or feedback, feel free to reach out:  
📧 **Email**: mq77gill@gmail.com  

---

## **Support the Project**
If you find this project useful, please give it a ⭐ on GitHub!

---

### **Detailed Model Descriptions**
Here’s a deeper dive into the models and their roles:

1. **Python**:
   - The backbone of the project, used for scripting, logic, and integrating all components.

2. **Gradio**:
   - Provides a simple and interactive chat interface for users to interact with the system.

3. **FAISS**:
   - A highly efficient library for similarity search, used to retrieve relevant legal documents based on user queries.

4. **Llama3-8b-8192 (Groq API)**:
   - A powerful language model used to generate detailed and structured responses based on the retrieved context.

5. **Helsinki-NLP/opus-mt-en-ur**:
   - A translation model specifically trained for English-to-Urdu translation. It translates the retrieved English context into Urdu for Urdu queries.

6. **Multilingual MiniLM-L12-v2**:
   - A multilingual embeddings model used to convert text into numerical vectors, enabling the system to understand and retrieve text in multiple languages.

7. **LangChain**:
   - A framework for building retrieval-augmented generation (RAG) pipelines. It handles document retrieval, prompt engineering, and response generation.

8. **Hugging Face Transformers**:
   - A library that provides pre-trained models for embeddings and translation, used to power the multilingual capabilities of the system.

9. **PyPDF2**:
   - A library for extracting text from PDF documents, used to process legal documents stored in PDF format.

