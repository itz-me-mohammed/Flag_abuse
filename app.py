import requests
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def fetch_comments(video_url, api_key, max_comments=100):
    video_id = video_url.split('v=')[1].split('&')[0]
    comments = []
    next_page_token = None
    
    while len(comments) < max_comments:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults=100"
        
        if next_page_token:
            url += f"&pageToken={next_page_token}"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            for item in data['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
            
            next_page_token = data.get('nextPageToken')
            
            # Break if there are no more pages to fetch
            if not next_page_token:
                break
        else:
            print("Failed to fetch comments.")
            break

    return comments[:max_comments] 

# Load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("autopilot-ai/EthicalEye")
    model = AutoModelForSequenceClassification.from_pretrained("autopilot-ai/EthicalEye")
    return tokenizer, model

# Function to classify comments using the model
def classify_comments(comments, tokenizer, model):
    results = []
    for comment in comments:
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=-1).item()
        results.append((comment, prediction))
    return results

# Streamlit app layout
st.title("YouTube Comment Analyzer")
video_url = st.text_input("Enter YouTube Video URL:")
api_key = st.text_input("Enter Your YouTube API Key:")

# Initialize session state for storing comments and results
if 'comments' not in st.session_state:
    st.session_state.comments = []
if 'classified_comments' not in st.session_state:
    st.session_state.classified_comments = []

if st.button("Fetch and Analyze Comments"):
    # Fetch comments
    new_comments = fetch_comments(video_url, api_key)
    st.session_state.comments.extend(new_comments)  # Add new comments to the session state

    if st.session_state.comments:
        tokenizer, model = load_model()
        classified_comments = classify_comments(st.session_state.comments, tokenizer, model)
        
        # Sort comments: Abusive (1) at the top
        classified_comments.sort(key=lambda x: x[1], reverse=True)
        st.session_state.classified_comments = classified_comments  # Store classified comments in session state

        st.write("### Comments and their Classification:")
        for comment, prediction in st.session_state.classified_comments:
            st.write(f"**Comment:** {comment} | **Classification:** {'Abusive' if prediction == 1 else 'Not Abusive'}")
    else:
        st.write("No comments found.")

# Display all fetched comments so far
if st.session_state.classified_comments:
    st.write("### All Fetched Comments So Far:")
    for comment, prediction in st.session_state.classified_comments:
        st.write(f"**Comment:** {comment} | **Classification:** {'Abusive' if prediction == 1 else 'Not Abusive'}")
