import streamlit as st
import requests

# Update API_URL as needed (for local testing, use localhost)
API_URL = "http://127.0.0.1:8000/analyze/"

st.title("📢 News Sentiment Analyzer & Conversational Hindi Speech Generator")

company = st.text_input("Enter Company Name:")

if st.button("Analyze"):
    if company:
        st.info(f"Fetching news articles and analysis for **{company}**...")
        response = requests.post(API_URL, json={"company": company})
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
                
                # Display Comparative Sentiment Analysis with formatted output
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
