import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from newspaper import Article
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(
    page_title="Learning English from News",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)



# Function to query OpenAI API
def query_openai(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is not set. Please set the 'OPENAI_API_KEY' environment variable or input it in the sidebar.")
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

#  summarize the content
def summarize_learning_content(text):
    try:
        # ส่งคำขอ prompt เข้าไปในระบบ
        prompt = [
            {"role": "system", "content": "You are an assistant that summarizes learning materials into clear, concise subtopics."},
            {"role": "user", "content": f"Summarize the following learning material into key subtopics and provide a concise explanation for each:\n\n{text}"}
        ]
        result = query_openai(prompt)

       
        if result:
            subtopics = result.split("\n")  # แยกตามบรรทัดในแต่ละหัวข้อ
            subtopics = [subtopic.strip() for subtopic in subtopics if subtopic.strip()]  # เอาเฉพาะที่มีเนื้อหา

            return "\n".join(subtopics)
        else:
            return "No valid summary returned."
    except Exception as e:
        return f"Error: {e}"

# หาคีย์เวิร์ดจากทั้งหมด
def extract_keywords(text):
    try:
        
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
        filtered_words = [word.lower() for word in filtered_words]
        if filtered_words:
            return filtered_words[:20]  
        else:
            return "No valid keywords found"
        
    except Exception as e:
        return f"Error during keyword extraction: {e}"

def get_word_info(word, choice):
    prompt = []
    if choice == "Explanation":
        prompt = [
            {"role": "system", "content": "You are an assistant that provides clear, concise definitions for words."},
            {"role": "user", "content": f"Please provide a brief and clear explanation for the word '{word}' in English."}
        ]
    elif choice == "Synonym":
        prompt = [
            {"role": "system", "content": "You are an assistant that provides synonyms for words."},
            {"role": "user", "content": f"Provide synonyms for the word '{word}' without any numbers or extra text. Just the synonyms, separated by commas."}
        ]
    try:
        result = query_openai(prompt)
        return result.lower()  # แปลงผลลัพธ์เป็นตัวพิมพ์เล็กทั้งหมด
    except Exception as e:
        return f"Error: {e}"


# สร้าง DataFrame 
def generate_keywords_df(keywords, choice):
    try:
       
        if not isinstance(keywords, list):
            raise ValueError(f"Expected a list of keywords, but got: {keywords}")
        
        word_info = [get_word_info(word, choice) for word in keywords]
    
        df = pd.DataFrame({
            "Word": keywords,
            choice: word_info
        })
        
        return df
    
    except Exception as e:
        return f"Error: {e}"

def process_text(text, choice):
    keywords = extract_keywords(text) 
    if isinstance(keywords, list):
        df = generate_keywords_df(keywords, choice)  
        return df
    else:
        return keywords  

# app setup 
st.markdown("## 📰 Learning English from News")  # หัวข้อใหญ่
st.write("Paste your URL here to generate a structured summary with subtopics.")

# ส่วน sidebar
st.sidebar.header("How to use")

# อันที่ 1 ให้ใส่ api key
st.sidebar.write("1. Enter your OpenAI API key below 🔑")
api_key = st.sidebar.text_input("OpenAI API Key:", type="password", key="api_key_input")
#ให้เลือก ความหมายหรือ synonym
st.sidebar.write("2. Choose the type of word information you want 📄")
choice = st.sidebar.selectbox("Select type of word information", ("Explanation", "Synonym"))
#อธิบายว่ามันทำะไรได้บ้าง
st.sidebar.header("📖 About")
st.sidebar.write("""Learning English from News provides you with a summary of the news article you enter, 
                 divided into interesting topics. It then selects 20 interesting words for you to learn, 
                 and you can choose whether you want to learn them through synonyms or explanations.
""") 

# หาคำจาก url
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error extracting text from URL: {e}"


news_url = st.text_input("Enter the news article URL:")

if st.button("Generate Summary and Words for Learning"):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        if news_url.strip():
            text = extract_text_from_url(news_url)
            if text:
                
                summary = summarize_learning_content(text)
                st.subheader("News Summary:")
                st.write(summary)

              
                keywords = extract_keywords(summary)

                try:
                    df = generate_keywords_df(keywords, choice)
                    st.subheader(f"Word Table with {choice}:")
                    st.dataframe(df)

                  
                    csv = df.to_csv(index=False)
                    st.subheader("Download the Table as CSV")
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="keywords_with_explanations_or_synonyms.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error generating the table: {e}")
            else:
                st.error("Could not extract text from the URL.")
        else:
            st.error("Please enter a valid news article URL.")
    else:
        st.error("Please enter your OpenAI API key in the sidebar.")
