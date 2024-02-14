# CV-Analyzer

# Introduction
The platform employs advanced Natural Language Processing (NLP) algorithms that analyze the candidate's CV and extract relevant information. <br>
CV Analyzer helps job seekers understand their CV's strengths and weaknesses, providing them with valuable insights into their job application process. The platform identifies gaps in employment, lack of relevant skills, and other areas that require improvement. This allows job seekers to make necessary adjustments to their CV and improve their chances of landing their dream job. <br>
For job administrators, CV Analyzer provides a powerful tool for posting job listings and ranking the best CVs based on the analysis provided by the tool. This makes the hiring process more efficient and effective, allowing employers to find the right candidate quickly. <br>
The platform also provides job seekers with personalized suggestions to enhance their CV, such as highlighting the skills and experiences that match the job requirements. This allows job seekers to tailor their resumes to each job they apply for, making them more competitive and increasing their chances of success.

# Objectives
Use CV to predict the most suitable job sectors for the candidate. <br>
Check if the candidate's CV matches the given job requirements. <br>
Rank candidates accordingly based on their CV scores and job requirements. <br>
Automate the resume screening process to reduce errors and save time. <br>
Eliminate human bias in the resume screening process. <br>
Provide recruiters with insights and analytics about the resume pool. <br>
Reviewing resume using Machine learning model <br>

# Installation

## Prerequisites

Ensure you have the following installed before setting up the project:

- Python 3.x
- pip (Python package installer)
- MongoDB installed and running
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bivek-shrestha/CV-Analyzer.git
cd CV-Analyzer
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain the following dependencies:

```plaintext
Flask
pymongo
PyPDF2
numpy
regex
scikit-learn
bson
joblib
pandas
spacy
nltk
uuid
```

### 3. Set Up MongoDB

Make sure MongoDB is installed and running. Update the MongoDB connection string in your Flask app.

### 4. Data Processing and Machine Learning

If you are using machine learning components, follow the respective documentation:

- scikit-learn: [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- spaCy: [spaCy Documentation](https://spacy.io/usage)
- NLTK: [NLTK Documentation](https://www.nltk.org/)

### 5. Flask Frontend

Run the Flask app using the following commands:

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

Visit `http://localhost:5000` in your web browser to access the application.

## Acknowledgments



# Methodology
## Data Collection 
<img width="337" alt="Screenshot 2024-02-14 182220" src="https://github.com/bivek-shrestha/CV-Analyzer/assets/155466197/8e82d1b0-bd2d-4c14-b88f-abfb1d8763fa">

## Preprocessing 
<img width="287" alt="Screenshot 2024-02-14 182322" src="https://github.com/bivek-shrestha/CV-Analyzer/assets/155466197/42b75934-e880-4b9a-a721-4599f168d619">
 <br>

# Model Training
TF-IDF Vectorization,  <br>
Cosine Similarity Algorithm <br>

# Model Architecture
<img width="359" alt="Screenshot 2024-02-14 183059" src="https://github.com/bivek-shrestha/CV-Analyzer/assets/155466197/a9fe8b94-6c69-4f2e-8e7c-d73215bab2bb">
 
# Outputs:
## Login Page
<img width="584" alt="login" src="https://github.com/bivek-shrestha/CV-Analyzer/assets/155466197/7d5d4aaf-d495-4470-80fe-0dea8d055cbc">
## USER
## UserPage
<img width="599" alt="userpage" src="https://github.com/bivek-shrestha/CV-Analyzer/assets/155466197/36285708-108f-4fcc-bb1d-151e94e0ca5c">

## Prediction Page
<img width="590" alt="predictionpage" src="https://github.com/bivek-shrestha/CV-Analyzer/assets/155466197/a5eae74c-0014-4f75-9c9f-5eec556d3a1e">

## Vacancy Page

<img width="597" alt="vacancypage" src="https://github.com/bivek-shrestha/CV-Analyzer/assets/155466197/871841d4-5947-4ec1-bf93-e280915ff134">

# ADMIN

## Admin Job Post Page

<img width="590" alt="adminjobpostpage" src="https://github.com/bivek-shrestha/CV-Analyzer/assets/155466197/41ccb53c-161a-4b4b-a66b-3d3b5451c8c3">

## Admin Vacancy Page
<img width="593" alt="jobvacancy" src="https://github.com/bivek-shrestha/CV-Analyzer/assets/155466197/ab4acfe5-461b-4190-bb57-ce5636b3b126">
