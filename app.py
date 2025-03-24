import streamlit as st
import requests
import threading
import uvicorn
import time

# ----------------- FastAPI Code -----------------
from fastapi import FastAPI
from pydantic import BaseModel
import requests as req  # Use a different alias to avoid conflict with Streamlit's requests
from bs4 import BeautifulSoup
from gtts import gTTS
import os
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from googletrans import Translator

# Download required NLTK data for sentiment analysis and NER
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def extract_topics_nltk(text):
    """
    Extracts key topics from the given text using NLTK's named entity recognition.
    It focuses on entities of type PERSON, ORGANIZATION, GPE, and LOCATION.
    Returns up to 3 unique topics.
    """
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(tagged)
    topics = []
    for subtree in tree:
        if hasattr(subtree, 'label'):
            if subtree.label() in ['PERSON', 'ORGANIZATION', 'GPE', 'LOCATION']:
                entity = " ".join(word for word, tag in subtree.leaves())
                if len(entity) >= 3 and not entity.isnumeric():
                    topics.append(entity)
    # Remove duplicates and return up to 3 topics
    filtered = []
    seen = set()
    for t in topics:
        if t not in seen:
            seen.add(t)
            filtered.append(t)
    return filtered[:3]

# Define the FastAPI app (named 'api' to avoid conflict with Streamlit's UI code)
api = FastAPI()

class CompanyRequest(BaseModel):
    company: str

def fetch_news(company):
    """
    Fetches at least 10 news articles related to a given company from Bing News.
    Extracts Title, Summary, Link, and Date.
    """
    url = f"https://www.bing.com/news/search?q={company}+reuters"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    try:
        response = req.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        articles = []
        seen_links = set()
        news_cards = soup.select("div.news-card, div.t_t")
        for card in news_cards:
            title_tag = card.select_one("a.title")
            summary_tag = card.select_one(".snippet")
            date_tag = card.select_one(".source .time")
            if title_tag and title_tag["href"] not in seen_links:
                title = title_tag.text.strip()
                link = title_tag["href"]
                summary = summary_tag.text.strip() if summary_tag else "No summary available."
                date = date_tag.text.strip() if date_tag else "Date not available"
                articles.append({
                    "Title": title,
                    "Summary": summary,
                    "link": link,
                    "date": date
                })
                seen_links.add(link)
            if len(articles) >= 10:
                break
        return articles
    except Exception:
        return []

def analyze_sentiment(text):
    """
    Performs sentiment analysis on the given text using VADER.
    Returns 'Positive', 'Negative', or 'Neutral'.
    """
    scores = sia.polarity_scores(text)
    if scores['compound'] > 0.05:
        return "Positive"
    elif scores['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"

def comparative_analysis(articles):
    """
    Generates a comparative sentiment analysis using broader comparisons among the top 5 articles.
    For every unique pair among the top 5 articles, it computes a sentiment difference and selects the top 2 pairs.
    Also computes topic overlap from the first 2 articles.
    """
    sentiment_mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
    top_articles = articles[:5] if len(articles) >= 5 else articles
    pairs = []
    n = len(top_articles)
    for i in range(n):
        for j in range(i+1, n):
            val_i = sentiment_mapping.get(top_articles[i]["Sentiment"], 0)
            val_j = sentiment_mapping.get(top_articles[j]["Sentiment"], 0)
            diff = abs(val_i - val_j)
            pairs.append((i, j, diff))
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    coverage_differences = []
    count = 0
    for (i, j, diff) in pairs:
        if count >= 2:
            break
        artA = top_articles[i]
        artB = top_articles[j]
        comparison = f"Article {i+1} highlights {artA['Title']}, while Article {j+1} discusses {artB['Title']}."
        s1 = artA["Sentiment"]
        s2 = artB["Sentiment"]
        if s1 == "Positive" and s2 == "Negative":
            impact = "The first article boosts confidence, while the second raises concerns about challenges ahead."
        elif s1 == "Negative" and s2 == "Positive":
            impact = "The first article underscores issues, whereas the second suggests potential recovery."
        elif s1 == s2:
            if s1 == "Positive":
                impact = "Both articles reflect a positive outlook, reinforcing optimism."
            elif s1 == "Negative":
                impact = "Both articles highlight challenges, indicating a pessimistic view."
            elif s1 == "Neutral":
                impact = "Both articles maintain a neutral tone, suggesting balanced reporting."
            else:
                impact = "Both articles share a similar sentiment."
        elif (s1 == "Positive" and s2 == "Neutral") or (s1 == "Neutral" and s2 == "Positive"):
            impact = "One article is optimistic while the other is measured, indicating cautious positivity."
        elif (s1 == "Negative" and s2 == "Neutral") or (s1 == "Neutral" and s2 == "Negative"):
            impact = "One article underscores challenges while the other is measured, suggesting a tempered outlook."
        else:
            impact = "The articles offer contrasting perspectives."
        coverage_differences.append({"Comparison": comparison, "Impact": impact})
        count += 1

    if len(articles) >= 2:
        topics1 = set(articles[0].get("Topics", []))
        topics2 = set(articles[1].get("Topics", []))
        common_topics = list(topics1.intersection(topics2))
        unique1 = list(topics1 - topics2)
        unique2 = list(topics2 - topics1)
    else:
        common_topics, unique1, unique2 = [], [], []
    
    topic_overlap = {
        "Common Topics": common_topics,
        "Unique Topics in Article 1": unique1,
        "Unique Topics in Article 2": unique2
    }
    
    return {
        "Sentiment Distribution": Counter(a["Sentiment"] for a in articles),
        "Coverage Differences": coverage_differences,
        "Topic Overlap": topic_overlap
    }

def generate_hindi_news_summary(articles):
    """
    Aggregates article titles and summaries, summarizes the content using LexRank,
    translates the summary into Hindi, and cleans up the text.
    """
    combined_text = " ".join([a["Title"] + ". " + a["Summary"] for a in articles])
    parser = PlaintextParser.from_string(combined_text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentences_count=3)
    english_summary = " ".join(str(s) for s in summary_sentences)
    if not english_summary.strip():
        sents = sent_tokenize(combined_text)
        english_summary = " ".join(sents[:3])
    try:
        translator = Translator()
        hindi_summary = translator.translate(english_summary, dest='hi').text
    except Exception:
        hindi_summary = english_summary
    hindi_summary = " ".join(hindi_summary.split())
    if not hindi_summary.endswith("."):
        hindi_summary += "."
    return hindi_summary

def text_to_speech(text, filename="news_hindi.mp3"):
    """
    Converts the given text into Hindi speech using gTTS and saves it as an MP3 file.
    """
    try:
        tts = gTTS(text=text, lang="hi", slow=False)
        tts.save(filename)
        return filename
    except Exception:
        return None

@api.post("/analyze/")
async def analyze_news_sentiments(data: CompanyRequest):
    """
    API Endpoint that:
      - Fetches news articles,
      - Performs sentiment analysis,
      - Extracts topics using NLTK's NE chunker,
      - Generates coverage differences for broader article comparisons,
      - Creates a Hindi summary and converts it to TTS audio.
    """
    company = data.company
    articles = fetch_news(company)
    if not articles:
        return {"error": "No articles found."}
    
    for article in articles:
        article["Sentiment"] = analyze_sentiment(article["Summary"])
        # Use NLTK-based topic extraction
        article["Topics"] = extract_topics_nltk(article["Summary"])
    
    comp_analysis = comparative_analysis(articles)
    hindi_summary = generate_hindi_news_summary(articles)
    audio_filename = text_to_speech(hindi_summary)
    
    pos_count = comp_analysis["Sentiment Distribution"].get("Positive", 0)
    neg_count = comp_analysis["Sentiment Distribution"].get("Negative", 0)
    final_sentiment = (
        f"{company}'s latest news coverage is mostly "
        f"{'positive' if pos_count > neg_count else 'negative'}."
    )
    
    return {
        "Company": company,
        "Articles": articles,
        "Comparative Sentiment Score": comp_analysis,
        "Final Sentiment Analysis": final_sentiment,
        "Audio": audio_filename if audio_filename else "Audio generation failed"
    }

# ----------------- End of FastAPI Code -----------------

# Function to run the FastAPI server in a background thread
def run_api():
    # Note: "app:api" assumes this file is named app.py.
    uvicorn.run("app:api", host="127.0.0.1", port=8000, log_level="info")

# Start the FastAPI server in a background thread (only once)
if "api_started" not in st.session_state:
    threading.Thread(target=run_api, daemon=True).start()
    st.session_state["api_started"] = True
    # Give the server a moment to start up
    time.sleep(1)

# ----------------- Streamlit UI Code -----------------

st.title("📢 News Sentiment Analyzer & Conversational Hindi Speech Generator")

company = st.text_input("Enter Company Name:")

if st.button("Analyze"):
    if company:
        st.info(f"Fetching news articles and analysis for **{company}**...")
        response = requests.post("http://127.0.0.1:8000/analyze/", json={"company": company})
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                st.error(result["error"])
            else:
                st.header(f"News Analysis for {result['Company']}")
                
                # Display Top News Articles
                st.subheader("📰 Top News Articles:")
                for idx, article in enumerate(result["Articles"], start=1):
                    st.markdown(f"### {idx}. {article['Title']}")
                    st.write(f"**Summary:** {article['Summary']}")
                    st.write(f"**Sentiment:** {article['Sentiment']}")
                    st.write(f"**Topics:** {', '.join(article['Topics'])}")
                    st.write(f"**Link:** [Read More]({article['link']})")
                    st.write(f"**Date:** {article['date']}")
                    st.write("---")
                
                # Display Comparative Sentiment Analysis
                st.subheader("📊 Comparative Sentiment Score:")
                comp = result["Comparative Sentiment Score"]
                
                st.markdown("**Sentiment Distribution:**")
                for sentiment, count in comp["Sentiment Distribution"].items():
                    st.write(f"- **{sentiment}:** {count}")
                
                st.markdown("**Coverage Differences:**")
                if comp["Coverage Differences"]:
                    for diff in comp["Coverage Differences"]:
                        st.write(f"- **Comparison:** {diff['Comparison']}")
                        st.write(f"  **Impact:** {diff['Impact']}")
                else:
                    st.write("No coverage differences available.")
                
                st.markdown("**Topic Overlap:**")
                topic_overlap = comp["Topic Overlap"]
                st.write("**Common Topics:**", ", ".join(topic_overlap["Common Topics"]) or "None")
                st.write("**Unique Topics in Article 1:**", ", ".join(topic_overlap["Unique Topics in Article 1"]) or "None")
                st.write("**Unique Topics in Article 2:**", ", ".join(topic_overlap["Unique Topics in Article 2"]) or "None")
                
                # Display Final Sentiment Analysis
                st.subheader("🔍 Final Sentiment Analysis:")
                st.write(result["Final Sentiment Analysis"])
                
                # Display Hindi Speech Summary Audio
                st.subheader("🎙️ Hindi Speech Summary:")
                if result["Audio"] and result["Audio"] != "Audio generation failed":
                    st.audio(result["Audio"])
                else:
                    st.error("Audio generation failed.")
        else:
            st.error("Failed to retrieve data from the API.")
    else:
        st.warning("Please enter a company name.")
