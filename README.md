# Document Q&A with GPT-4o-mini

A Streamlit-based RAG (Retrieval-Augmented Generation) application that uses OpenAI's GPT-4o-mini to chat with and get insights from your documents.

## Features

- **Document Upload**: Upload and process Word documents (.docx) and Excel files (.xlsx, .xls).
- **Document Understanding**: The app provides the full document content to the model for deep understanding.
- **Chat Interface**: Ask questions about your documents and get answers with references back to the source.
- **Chat History**: Save, manage, rename, and export multiple chat sessions.
- **Document Management**: View, preview, summarize, and delete uploaded documents.
- **Document Summaries**: Get AI-generated summaries of your documents.
- **Search Functionality**: Search within your documents and see results with context.
- **Custom Instructions**: Customize the AI system prompt to change how it responds.
- **Model Parameter Control**: Adjust temperature and max tokens for responses.
- **Chat Export**: Export your conversations to text files.
- **Tabbed Interface**: Easily navigate between chat, document management, and settings.
- **Quick Document Access**: View document summaries from the sidebar.
- **Search Commands**: Use "search:" command in the chat to search documents directly.

## Setup

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd [repository-directory]
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a .env file**:
   Copy the `.env.example` file to `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload Documents**:
   - Go to the "Document Management" tab to upload .docx or Excel files.
   - Click "Process Document" to extract text and generate a summary.

2. **Start a Chat**:
   - Click "New Chat" in the sidebar to create a new chat session.
   - Select any previously created chat to continue a conversation.

3. **Ask Questions**:
   - Type your questions related to the uploaded documents in the chat input.
   - The AI will analyze the documents and provide answers with references to the original documents.

4. **Search Documents**:
   - Type "search: [term]" in the chat to search for specific terms in your documents.
   - Use the search box in the Document Management tab for more advanced searching.

5. **Document Features**:
   - View document summaries in the sidebar or Document Management tab.
   - Preview document content to see what's been extracted.
   - Delete documents that are no longer needed.

6. **Chat Management**:
   - Rename chats for better organization.
   - Export chat conversations to text files.
   - Clear all chat history if needed.

7. **Customize Settings**:
   - Adjust system prompts to change how the AI formats responses.
   - Modify model parameters like temperature and max tokens.

## How It Works

Unlike traditional RAG systems that use vector stores, this application:

1. Extracts the full text content from uploaded documents
2. Provides the document content directly to the GPT-4o-mini model as context
3. Allows document summarization and searching for better information access
4. Instructs the model to reference specific sections in its responses
5. Maintains conversation context for follow-up questions

This approach allows the model to develop a deeper understanding of the documents and provide more accurate answers with proper source attribution.

## Limitations

- The app is designed for moderate-sized documents. Very large documents may be truncated in the context window.
- Currently supports only Word (.docx) and Excel (.xlsx, .xls) files.
- Requires an OpenAI API key with access to the GPT-4o-mini model.

## Requirements

- Python 3.7+
- Streamlit
- OpenAI
- Pandas
- Plotly
- Pillow
- python-dotenv

## License

MIT 
