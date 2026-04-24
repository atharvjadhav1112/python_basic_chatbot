# PDF RAG Chatbot with Groq

A powerful Retrieval-Augmented Generation (RAG) chatbot built with Gradio and Groq's fast language models. Upload PDF documents and ask questions about their content with streaming responses.

## Features

- 📄 **PDF Upload & Indexing**: Upload PDF documents and automatically extract and chunk text
- 💬 **RAG-Powered Chat**: Ask questions about your documents with context-aware answers
- ⚡ **Fast Streaming**: Real-time streaming responses using Groq's optimized LLMs
- 🎨 **User-Friendly Interface**: Clean web UI built with Gradio
- 🔍 **Semantic Search**: Retrieve relevant document chunks based on query similarity
- 📊 **Multiple Models**: Support for various Groq models (Llama, Mixtral, etc.)

## Prerequisites

- Python 3.8+
- Groq API key (get one free at [console.groq.com](https://console.groq.com/keys))

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/atharvjadhav1112/python_basic_chatbot.git
   cd python_basic_chatbot
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requiremets.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env` (or create a `.env` file)
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

## Configuration

Edit the `.env` file to customize:

```env
# Your Groq API key
GROQ_API_KEY=your_api_key_here

# Model selection (available models: llama-3.1-70b-versatile, llama-3.1-8b-instant, etc.)
GROQ_MODEL=llama-3.1-70b-versatile

# Model parameters
GROQ_TEMPERATURE=0.7
GROQ_TOP_P=0.95
GROQ_MAX_OUTPUT_TOKENS=2048
```

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open in browser**
   - The app will automatically open at `http://localhost:7860`

3. **Using the chatbot**
   - Upload a PDF using the "PDF" button on the left
   - Click "Index this PDF" to process the document
   - Ask questions in the chat box
   - The chatbot will search your documents and provide answers with sources

## How It Works

### Architecture

1. **Document Ingestion**: PDFs are uploaded and text is extracted
2. **Chunking**: Text is split into overlapping chunks for better context preservation
3. **Retrieval**: Queries are tokenized and matched against document chunks using simple semantic similarity
4. **Generation**: Retrieved chunks are combined with the user query and sent to Groq's LLM
5. **Streaming**: Responses are streamed in real-time for better UX

### Key Components

- **`chunk_text()`**: Splits documents into manageable chunks with overlap
- **`retrieve()`**: Finds the most relevant chunks for a query
- **`RagChatbot`**: Main class managing the RAG pipeline and API communication
- **`build_ui()`**: Creates the Gradio interface

## Dependencies

- `groq>=0.4.0` - Groq SDK for API access
- `gradio>=4.0.0` - Web UI framework
- `pypdf>=4.0.0` - PDF text extraction
- `python-dotenv>=1.0.0` - Environment variable management

## Troubleshooting

### API Key Issues
- Ensure `GROQ_API_KEY` is set in your `.env` file
- Get a free key from [console.groq.com/keys](https://console.groq.com/keys)

### Model Decommissioned Error
- Check [Groq's model deprecations page](https://console.groq.com/docs/deprecations)
- Update `GROQ_MODEL` in `.env` to an available model

### PDF Extraction Issues
- Ensure PDFs contain selectable text (not scanned images)
- For scanned PDFs, consider using OCR tools first

## Available Groq Models

Check the [Groq documentation](https://console.groq.com/docs/models) for the latest available models:
- `llama-3.1-70b-versatile` - Large, powerful model (default)
- `llama-3.1-8b-instant` - Smaller, faster model
- Other models available based on your plan

## Advanced Features

### Customizing Chunk Settings
In `app.py`, modify these constants:
```python
CHUNK_SIZE = 900  # Characters per chunk
CHUNK_OVERLAP = 150  # Overlap between chunks
TOP_K = 6  # Number of chunks to retrieve
```

### Changing System Instruction
Modify the `SYSTEM_INSTRUCTION` variable in `app.py` to customize the chatbot's behavior:
```python
SYSTEM_INSTRUCTION = "Your custom instruction here..."
```

## API Rate Limits

Groq has generous free tier limits. Monitor your usage in the [Groq console](https://console.groq.com).

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests.

## Support

For issues or questions:
- Check the [Groq documentation](https://console.groq.com/docs)
- Open an issue on [GitHub](https://github.com/atharvjadhav1112/python_basic_chatbot/issues)

## Disclaimer

This is a demonstration project. For production use, consider:
- Implementing proper authentication
- Adding rate limiting
- Using more sophisticated retrieval methods (embeddings, vector databases)
- Adding error handling and logging
- Securing your API keys properly

---

**Built with ❤️ using Groq's fast APIs**