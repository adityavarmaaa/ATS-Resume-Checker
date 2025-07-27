import streamlit as st
import pdfplumber
import docx
import re
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([p.text for p in doc.paragraphs])

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = text.lower()
    return " ".join([word for word in text.split() if word not in stop_words])

def calculate_similarity(resume_text, jd_text):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([resume_text, jd_text])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def keyword_match(resume_text, jd_text):
    jd_doc = nlp(jd_text)
    jd_keywords = set([token.lemma_ for token in jd_doc if token.pos_ in ['NOUN', 'PROPN', 'VERB']])
    resume_doc = nlp(resume_text)
    resume_words = set([token.lemma_ for token in resume_doc])
    matched_keywords = resume_words.intersection(jd_keywords)
    return matched_keywords, len(matched_keywords), len(jd_keywords)

def ats_format_score(text):
    # Very basic checks: no tables, bullet points, excessive columns
    score = 100
    if len(re.findall(r'\|', text)) > 10:
        score -= 20
    if len(re.findall(r'•', text)) > 20:
        score -= 10
    if len(re.findall(r'\t', text)) > 20:
        score -= 10
    return max(score, 0)

# --- Streamlit UI ---
st.set_page_config(page_title="ATS Resume Checker", layout="centered")

st.title("📄 ATS Resume Checker & Reviewer")
st.write("Upload your resume and a job description to see how well you match.")

resume_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])
jd_input = st.text_area("Paste the Job Description here")

if resume_file and jd_input:
    with st.spinner("Analyzing..."):

        resume_text = ""
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(resume_file)

        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(jd_input)

        sim_score = calculate_similarity(resume_clean, jd_clean)
        keywords, matched, total = keyword_match(resume_clean, jd_clean)
        ats_score = ats_format_score(resume_text)

    st.success("✅ Analysis Complete!")

    st.subheader("📊 Results")
    st.metric("🔍 Match Score", f"{sim_score*100:.2f}%")
    st.metric("🧠 Keyword Match", f"{matched}/{total} keywords")
    st.metric("📦 ATS Format Score", f"{ats_score}/100")

    with st.expander("📝 Matched Keywords"):
        st.write(", ".join(keywords) if keywords else "No keywords matched.")

    st.info("✨ Tip: Try to use the exact keywords from the job description in your resume's experience and skills sections.")
