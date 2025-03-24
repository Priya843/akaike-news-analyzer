# Akaike News Sentiment Analyzer

**Company news sentiment analyzer with TTS.**

## Overview

Akaike News Sentiment Analyzer is a web-based tool that extracts the latest company-related news, analyzes sentiment, compares article tone, and generates Hindi audio summaries. It uses a FastAPI backend for data processing and a Streamlit frontend for a user-friendly interface.

## Features

- **News Extraction:** Scrapes at least 10 unique news articles from Bing News using BeautifulSoup.
- **Sentiment Analysis:** Classifies articles as positive, negative, or neutral using VADER.
- **Comparative Analysis:** Compares article sentiment and topics to derive insights on how the company's news coverage varies.
- **Text-to-Speech:** Summarizes news content and converts the summary into Hindi speech using gTTS.
- **API-based Architecture:** Frontend communicates with the FastAPI backend via APIs.
- **Deployment Ready:** Configured for deployment on Hugging Face Spaces.

## Installation

### Prerequisites

- Python 3.8 or later
- pip

### Setup Steps

1. **Clone the Repository:**

    ```bash
    git clone [https://github.com/your-username/akaike-news-analyzer.git](https://github.com/your-username/akaike-news-analyzer.git)
    cd akaike-news-analyzer
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download NLTK and SpaCy Resources:**

    ```bash
    python -m nltk.downloader vader_lexicon punkt
    python -m spacy download en_core_web_sm
    ```

### Running the Application Locally

1. **Start the FastAPI Server:**

    ```bash
    python -m uvicorn api:app --reload
    ```

2. **Launch the Streamlit Frontend:**

    ```bash
    streamlit run app.py
    ```

    Then open your browser at: `http://localhost:8501`

### API Endpoint

- `POST /analyze/`

    **Request Body:**

    ```json
    {
        "company": "Tesla"
    }
    ```

    **Sample Response:**

    ```json
    {
        "Company": "Tesla",
        "Articles": [
            {
                "Title": "Tesla reports record-breaking sales",
                "Summary": "Tesla announced their Q4 earnings, surpassing all expectations...",
                "Sentiment": "Positive",
                "Topics": ["Tesla", "Q4", "Earnings"],
                "link": "[https://www.example.com/article1](https://www.example.com/article1)",
                "date": "March 23, 2025"
            },
            {
                "Title": "Tesla faces regulatory scrutiny over autopilot",
                "Summary": "Regulatory bodies are investigating Tesla's autopilot feature...",
                "Sentiment": "Negative",
                "Topics": ["Tesla", "Regulation", "Autonomous"],
                "link": "[https://www.example.com/article2](https://www.google.com/search?q=https://www.example.com/article2)",
                "date": "March 22, 2025"
            },
            {
                "Title": "Tesla unveils new innovative battery technology",
                "Summary": "Tesla showcased their latest advancements in battery technology...",
                "Sentiment": "Positive",
                "Topics": ["Tesla", "Innovation", "Battery"],
                "link": "[https://www.example.com/article3](https://www.google.com/search?q=https://www.example.com/article3)",
                "date": "March 21, 2025"
            },
            {
                "Title": "Lawsuit filed against Tesla for safety concerns",
                "Summary": "A class action lawsuit has been filed against Tesla...",
                "Sentiment": "Negative",
                "Topics": ["Tesla", "Lawsuit", "Safety"],
                "link": "[https://www.example.com/article4](https://www.google.com/search?q=https://www.example.com/article4)",
                "date": "March 20, 2025"
            },
            {
                "Title": "Tesla stock surges after positive market analysis",
                "Summary": "Market analysts predict strong growth for Tesla stock...",
                "Sentiment": "Positive",
                "Topics": ["Tesla", "Stock", "Market"],
                "link": "[https://www.example.com/article5](https://www.google.com/search?q=https://www.example.com/article5)",
                "date": "March 19, 2025"
            },
            {
                "Title": "Tesla expands its charging network",
                "Summary": "Tesla announced the expansion of its supercharger network...",
                "Sentiment": "Positive",
                "Topics": ["Tesla", "Charging", "Network"],
                "link": "[https://www.example.com/article6](https://www.google.com/search?q=https://www.example.com/article6)",
                "date": "March 18, 2025"
            },
            {
                "Title": "Tesla recalls vehicles due to software glitch",
                "Summary": "Tesla issued a recall for certain models due to a software issue...",
                "Sentiment": "Negative",
                "Topics": ["Tesla", "Recall", "Software"],
                "link": "[https://www.example.com/article7](https://www.google.com/search?q=https://www.example.com/article7)",
                "date": "March 17, 2025"
            },
            {
                "Title": "Tesla partners with renewable energy company",
                "Summary": "Tesla has entered a partnership to enhance renewable energy solutions...",
                "Sentiment": "Positive",
                "Topics": ["Tesla", "Renewable", "Energy"],
                "link": "[https://www.example.com/article8](https://www.google.com/search?q=https://www.example.com/article8)",
                "date": "March 16, 2025"
            },
            {
                "Title": "Tesla announces plans for new Gigafactory",
                "Summary": "Tesla revealed plans to build a new Gigafactory in Europe...",
                "Sentiment": "Positive",
                "Topics": ["Tesla", "Gigafactory", "Expansion"],
                "link": "[https://www.example.com/article9](https://www.google.com/search?q=https://www.example.com/article9)",
                "date": "March 15, 2025"
            },
            {
                "Title": "Tesla faces criticism over environmental impact",
                "Summary": "Environmental groups criticize Tesla's manufacturing processes...",
                "Sentiment": "Negative",
                "Topics": ["Tesla", "Environment", "Manufacturing"],
                "link": "[https://www.example.com/article10](https://www.google.com/search?q=https://www.example.com/article10)",
                "date": "March 14, 2025"
            }
        ],
        "Comparative Sentiment Score": {
            "Sentiment Distribution": {
                "Positive": 6,
                "Neutral": 0,
                "Negative": 4
            },
            "Coverage Differences": [
                {
                    "Comparison": "Article 1 highlights Tesla's record sales, while Article 2 discusses regulatory issues.",
                    "Impact": "The first article boosts confidence, while the second raises concerns about challenges ahead."
                },
                {
                    "Comparison": "Article 3 emphasizes innovation, whereas Article 4 focuses on lawsuits against Tesla.",
                    "Impact": "The contrasting coverage shows the balance of excitement and risk around the company."
                }
            ],
            "Topic Overlap": {
                "Common Topics": ["Tesla"],
                "Unique Topics in Article 1": ["Sales", "Market"],
                "Unique Topics in Article 2": ["Regulation", "Autonomous"]
            }
        },
        "Final Sentiment Analysis": "Tesla's latest news coverage is mostly positive.",
        "Audio": "news_hindi.mp3"
    }
    ```

### Project Structure

```bash
akaike-news-analyzer/
├── api.py           # FastAPI backend
├── app.py           # Streamlit frontend
├── requirements.txt # Python dependencies
├── space.yaml      # Hugging Face Spaces configuration
└── README.md        # Project documentation
