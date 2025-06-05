# 📊 Chat with CSV (LLM)

A powerful Streamlit application that allows you to upload CSV files and ask questions in natural language to get intelligent insights using fine-tuned Language Learning Models (LLM).

## 🌟 Features

- **Natural Language Querying**: Ask questions like "What is the average age?" and get precise answers
- **Advanced CSV Processing**: Robust data cleaning, type inference, and preprocessing
- **LLM Integration**: Uses fine-tuned open-source models for intelligent query understanding
- **SQL Query Generation**: Automatically converts natural language to SQL for complex analysis
- **Interactive Visualizations**: Dynamic charts and graphs using Plotly
- **Conversation History**: Keep track of your queries and responses
- **Real-time Data Preview**: Live preview of your data with statistics

## 🚀 Quick Start

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd chat-with-csv-llm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

### Usage

1. **Upload CSV**: Click "Choose a CSV file" in the sidebar
2. **Ask Questions**: Type natural language questions in the chat interface
3. **Get Answers**: Receive intelligent responses with precise numerical results
4. **Explore Data**: Use the data preview and visualization features

## 💡 Example Queries

- "What is the average age?" → Returns precise average (e.g., 34.2)
- "How many records are there?" → Returns total count
- "What's the maximum salary?" → Returns highest value
- "Show me the sum of revenue" → Returns total sum
- "Give me a summary of the data" → Comprehensive overview

## 🏗️ Architecture

### Core Components

1. **CSVProcessor**: Advanced CSV handling with:
   - Multi-encoding support (UTF-8, Latin-1, CP1252, ISO-8859-1)
   - Automatic data type inference
   - Data cleaning and preprocessing
   - In-memory SQLite database creation

2. **LLMQueryProcessor**: Natural language understanding with:
   - Intent classification (aggregation, filter, comparison, summary)
   - Column extraction from queries
   - SQL query generation
   - Fine-tuned model integration

3. **ChatCSVApp**: Streamlit interface with:
   - Interactive chat interface
   - Real-time data visualization
   - Conversation history management
   - Responsive UI design

### LLM Integration

The application uses fine-tuned open-source models:
- **Primary Model**: Microsoft DialoGPT-medium for conversational understanding
- **Fallback**: GPT-2 for basic text generation
- **Custom Processing**: Intent classification and query parsing algorithms

## 🛠️ Advanced Features

### Data Processing
- **Smart Type Detection**: Automatically detects and converts data types
- **Date Recognition**: Intelligent datetime parsing
- **Missing Value Handling**: Comprehensive null value analysis
- **Encoding Support**: Multiple character encoding detection

### Query Understanding
- **Intent Classification**: Understands aggregation, filtering, comparison queries
- **Column Mapping**: Intelligently maps query terms to dataset columns  
- **SQL Generation**: Converts natural language to optimized SQL queries
- **Error Handling**: Graceful error management with helpful feedback

### Visualization
- **Dynamic Charts**: Auto-generated histograms and distributions
- **Interactive Plots**: Plotly-powered interactive visualizations
- **Statistical Summaries**: Comprehensive dataset statistics
- **Real-time Updates**: Live data preview and statistics

## 📊 Supported Query Types

### Aggregation Queries
- Average/Mean: "What is the average age?"
- Sum/Total: "What's the total revenue?"
- Count: "How many customers are there?"
- Maximum: "What's the highest score?"
- Minimum: "What's the lowest price?"

### Summary Queries
- "Give me a summary of the data"
- "Describe the dataset"
- "Show me the statistics"

### Filter Queries
- "Show me records where age > 30"
- "Find customers from New York"

## 🔧 Configuration

### Model Configuration
You can modify the LLM model in the `LLMQueryProcessor` class:

```python
model_name = "microsoft/DialoGPT-medium"  # Change this to your preferred model
```

### Database Configuration
The application uses SQLite in-memory database for fast query execution:

```python
self.sql_connection = sqlite3.connect(':memory:')
```

## 📝 Requirements

- Python 3.8+
- Streamlit 1.28.0+
- PyTorch 2.0.0+
- Transformers 4.30.0+
- Pandas 1.5.0+
- See `requirements.txt` for complete list

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Model Loading Errors:**
- Ensure you have sufficient RAM (8GB+ recommended)
- Check internet connection for model download
- Try using CPU-only version if GPU issues occur

**CSV Processing Errors:**
- Verify CSV format and encoding
- Check for special characters in column names
- Ensure file size is under 200MB for optimal performance

**Query Understanding Issues:**
- Use specific column names in queries
- Try rephrasing questions with keywords like "average", "count", "sum"
- Check data preview to understand column structure

## 🙏 Acknowledgments

- Hugging Face Transformers for LLM integration
- Streamlit for the amazing web framework
- Plotly for interactive visualizations
- Open-source community for model contributions

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Review the documentation

---

**Made with ❤️ using Streamlit and Open Source LLMs**