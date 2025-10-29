# 🤖 HR Chatbot

A multilingual HR assistant powered by **FastAPI**, **FAISS**, and **Hugging Face** — designed to handle employee queries, document searches, and feedback seamlessly.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green?logo=fastapi)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

---

## 📚 Table of Contents
- [🚀 Clone the Repository](#-clone-the-repository)
- [🐍 Create Virtual Environment](#-create-virtual-environment)
- [📦 Install Dependencies](#-install-dependencies)
- [⚙️ Set Up Environment Variables](#️-set-up-environment-variables)
- [📁 Prepare Data Files](#-prepare-data-files)
- [🧱 Build Vector Database](#-build-vector-database)
- [▶️ Run the Application](#️-run-the-application)
- [🖥️ API Endpoints](#️-api-endpoints)
- [📂 Project Structure](#-project-structure)
- [🔧 Configuration](#-configuration)
- [🌐 Deployment](#-deployment)
- [🧪 Testing](#-testing)
- [🛠️ Development](#️-development)
- [📊 Monitoring](#-monitoring)
- [🐛 Troubleshooting](#-troubleshooting)
- [🔐 Security](#-security)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)
- [👥 Authors](#-authors)
- [🙏 Acknowledgments](#-acknowledgments)
- [📞 Support](#-support)
- [🗺️ Roadmap](#️-roadmap)
- [📈 Version History](#-version-history)

---

## 🚀 **Clone the Repository**

# HR Chatbot - Multilingual AI Assistant

An intelligent HR assistant chatbot supporting **English**, **Bangla**, and **Banglish** with context-aware conversations, employee information lookup, and a comprehensive knowledge base.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

- **Multilingual Support** - English, Bangla, and Banglish (Romanized Bangla)
- **Context-Aware** - Maintains conversation history and context
- **Knowledge Base** - FAISS vector search for HR policies and information
- **Employee Lookup** - Secure employee information retrieval with Gmail verification
- **Rating System** - User feedback and rating dashboard
- **Secure** - Email verification and data protection
- **Fast** - Built with FastAPI and efficient vector search

---

## Use Cases

- Leave policies and procedures
- Salary and benefits information
- Office hours and holidays
- Employee directory lookup
- HR policy questions
- General workplace queries

---

## Architecture

```
┌─────────────────┐
│   Frontend      │  HTML/CSS/JavaScript
│   (Chat UI)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI       │  REST API Server
│   Backend       │
└────────┬────────┘
         │
         ├──────► Hugging Face API (Llama-3.2-3B)
         │
         ├──────► FAISS Vector Database
         │
         └──────► Employee JSON Database
```

---

## Prerequisites

- Python 3.9 or higher
- Hugging Face account and API token
- 8GB RAM (minimum)
- 20GB storage

---

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/yourusername/hr-chatbot.git
cd hr-chatbot
```

### Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt

ollama pull (model you like )
```

### Set Up Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
INDEX_PATH=vectorstores/db_faiss/index.faiss
TEXTS_PATH=vectorstores/db_faiss/texts.npy
```

### Prepare Data Files

Ensure these files exist:

```
data/
├── General HR Queries.pdf
└── employees.json

vectorstores/db_faiss/
├── index.faiss
└── texts.npy
```

### Build Vector Database (First Time Only)

```bash
# Edit paths in create_db.py to match your setup
python create_db.py
```

### Run the Application

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Visit: `http://localhost:8000`

---

## Installation

### Option 1: Manual Installation

```bash
pip install fastapi uvicorn python-dotenv
pip install openai sentence-transformers faiss-cpu
pip install spacy pydantic numpy
pip install deep-translator pdfminer.six PyMuPDF
ollama pull (model you like )
```

### Option 2: Using requirements.txt

```bash
pip install -r requirements.txt
ollama pull (model you like )
```

### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "model": "Llama-3.2-3B",
  "features": ["chat", "ratings", "multilingual"]
}
```

### Chat

```http
POST /chat
Content-Type: application/json

{
  "message": "What is the leave policy?",
  "chat_history": []
}
```

**Response:**

```json
{
  "response": "At Acme AI, employees are entitled to..."
}
```

### Submit Rating

```http
POST /rate
Content-Type: application/json

{
  "rating": 5,
  "feedback": "Very helpful!"
}
```

### Get Ratings

```http
GET /ratings
GET /ratings/stats
```

---

## Project Structure

```
hr-chatbot/
├── api_server.py              # FastAPI application
├── backend.py                 # Core chatbot logic
├── bangla_handler.py          # Multilingual processing
├── employee_lookup.py         # Employee database queries
├── ratings_manager.py         # Rating system
├── create_db.py              # FAISS database builder
├── pdf_reader.py             # PDF text extraction
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
├── data/
│   ├── General HR Queries.pdf
│   ├── employees.json
│   └── ratings.json
├── vectorstores/
│   └── db_faiss/
│       ├── index.faiss
│       └── texts.npy
└── static/
    ├── index.html            # Chat interface
    └── ratings_dashboard.html # Ratings dashboard
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL=` |(http://localhost:11434) | Required |
| `OLLAMA_BASE_URL=` |model name | Required |
| `INDEX_PATH` | Path to FAISS index | `vectorstores/db_faiss/index.faiss` |
| `TEXTS_PATH` | Path to text chunks | `vectorstores/db_faiss/texts.npy` |

### Employee Data Format

`employees.json`:

```json
[
  {
    "name": "John Doe",
    "email": "john.doe@acmeai.tech",
    "position": "Team Lead",
    "blood_group": "A+",
    "table": "1-A"
  }
]
```

---

## Deployment

### Production with Systemd

Create `/etc/systemd/system/hr-chatbot.service`:

```ini
[Unit]
Description=HR Chatbot API
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/hr-chatbot
Environment="PATH=/path/to/hr-chatbot/venv/bin"
ExecStart=/path/to/hr-chatbot/venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable hr-chatbot
sudo systemctl start hr-chatbot
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN ollama pull (model you like)
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t hr-chatbot .
docker run -p 8000:8000 --env-file .env hr-chatbot
```

### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  hr-chatbot:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./vectorstores:/app/vectorstores
    env_file:
      - .env
    restart: always
```

Run with:

```bash
docker-compose up -d
```

---

## Testing

### Test the Backend

```bash
python backend.py
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the office hours?", "chat_history": []}'
```

### View Ratings

```bash
python view_ratings.py
```

---

## Development

### Adding New HR Knowledge

1. Update `data/General HR Queries.pdf`
2. Rebuild FAISS index:
   ```bash
   python create_db.py
   ```

### Adding Employees

1. Update `data/employees.json`
2. Restart the application

### Customizing Responses

Edit the prompt in `backend.py` (line ~120) to customize bot behavior.

---

## Monitoring

### View Logs

```bash
# If using systemd
sudo journalctl -u hr-chatbot -f

# If running manually
tail -f app.log
```

### Access Ratings Dashboard

Open in browser:

```
http://localhost:8080/ratings_dashboard.html
```

### Check Application Status

```bash
# Using systemd
sudo systemctl status hr-chatbot

# Using curl
curl http://localhost:8000/health
```

---

## Troubleshooting

### Issue: "Module not found"

```bash
pip install -r requirements.txt
ollama pull (model you like )
```

### Issue: "FAISS index not found"

```bash
python create_db.py
```

### Issue: "ollama API error"

- download ollama in the PC
- then install a ollama model you like

### Issue: Port already in use

```bash
# Find process
lsof -i :8000

# Kill it
kill -9 <PID>
```

### Issue: Permission denied

```bash
# Fix file permissions
sudo chown -R $USER:$USER /path/to/hr-chatbot
chmod -R 755 /path/to/hr-chatbot
```

---

## Security

- Never commit `.env` file to Git
- Use environment variables for sensitive data
- Enable CORS only for trusted domains in production
- Implement rate limiting for API endpoints
- Use HTTPS in production (SSL/TLS)
- Keep dependencies updated
- Restrict file permissions on sensitive files

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Authors

- ** Probir Saha Shohom.** - *Initial work*

---

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Hugging Face](https://huggingface.co/) - AI models and inference
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Sentence Transformers](https://www.sbert.net/) - Embedding models

---


## Roadmap

- [ ] Add voice input support
- [ ] Implement advanced analytics
- [ ] Add more language support
- [ ] Create mobile application
- [ ] Integrate with Slack/Teams
- [ ] Add document upload capability

---

## Version History

- **v2.0** - Multilingual support, context awareness, rating system
- **v1.0** - Initial release with basic chatbot functionality

---

