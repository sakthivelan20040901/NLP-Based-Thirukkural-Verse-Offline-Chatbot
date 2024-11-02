from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk

# Download necessary NLTK data if not already downloaded
nltk.download('stopwords', quiet=True)

# Load the dataset
file_path = r"C:\Users\SAKTHIVELAN\Downloads\data - data.csv.csv"

# Read the CSV file into a DataFrame
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading the data: {e}")

# Check for required columns
required_columns = ['ENGLISH_EXPLANATION', 'NUMBER', 'TAMIL_VERSE', 'ENGLISH_VERSE', 'TAMIL_EXPLANATION']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Replace NaN values in the explanation columns with an empty string
df['ENGLISH_EXPLANATION'] = df['ENGLISH_EXPLANATION'].fillna('')
df['TAMIL_EXPLANATION'] = df['TAMIL_EXPLANATION'].fillna('')

# Define Tamil stopwords (you may need to find a suitable list for this)
tamil_stop_words = ["இது", "அது", "ஒரு", "என்", "உங்கள்", "உடன்"]  # Add more stopwords as needed

# Function to preprocess Tamil text
def preprocess_tamil_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    text = [word for word in text if word not in tamil_stop_words]  # Remove Tamil stopwords
    text = ' '.join(text)
    return text

# Function to preprocess English text
def preprocess_english_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    text = [word for word in text if word not in nltk.corpus.stopwords.words('english')]  # Remove stopwords
    text = ' '.join(text)
    return text

# Apply preprocessing to both English and Tamil explanations
df['PROCESSED_EXPLANATION'] = df['ENGLISH_EXPLANATION'].apply(preprocess_english_text) + ' ' + df['TAMIL_VERSE'].apply(preprocess_tamil_text)

# Initialize TF-IDF Vectorizer with bigrams and trigrams, and fit on both Tamil and English text
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['PROCESSED_EXPLANATION'])

# Function to get the most relevant verse based on the query, with Tamil preprocessing
def get_most_relevant_verse(query, language='english'):
    # Preprocess the query based on the language
    if language == 'tamil':
        query = preprocess_tamil_text(query)
    else:
        query = preprocess_english_text(query)
    
    # Transform the query into the same TF-IDF space
    query_tfidf = tfidf_vectorizer.transform([query])
    
    # Compute cosine similarity between the query and each explanation
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    # Get the index of the most similar explanation
    most_similar_idx = cosine_similarities.argmax()
    
    # Retrieve the relevant Thirukkural details
    return df.iloc[most_similar_idx]

# Initialize Flask app
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to process user input and return result
@app.route('/get_verse', methods=['POST'])
def get_verse():
    if request.method == 'POST':
        user_query = request.form['query']
        
        # Check if the input is a number (for retrieving a specific Thirukkural by number)
        if user_query.isdigit():
            kural_number = int(user_query)
            
            # Check if the number is within the valid range of Thirukkurals
            if 1 <= kural_number <= len(df):
                result = df[df['NUMBER'] == kural_number].iloc[0]
            else:
                return render_template('index.html', query=user_query, error="Invalid Thirukkural number. Please enter a number between 1 and 1330.")
        else:
            # Detect language (basic implementation, refine as needed)
            if any("\u0B80" <= char <= "\u0BFF" for char in user_query):  # Tamil Unicode range
                language = 'tamil'
            else:
                language = 'english'
            
            # Get the most relevant verse based on the query and detected language
            result = get_most_relevant_verse(user_query, language)
        
        # Prepare the result for display
        verse_number = result['NUMBER']
        tamil_verse = result['TAMIL_VERSE']
        english_translation = result['ENGLISH_VERSE']
        explanation = result['ENGLISH_EXPLANATION']
        tamil_explanation = result['TAMIL_EXPLANATION']
        
        # Return the response
        return render_template('index.html', query=user_query, verse_number=verse_number, 
                               tamil_verse=tamil_verse, english_translation=english_translation, 
                               explanation=explanation, tamil_explanation=tamil_explanation)

if __name__ == "__main__":
    app.run(debug=True)
