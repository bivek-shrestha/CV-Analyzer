from flask import Flask, render_template, request, session, redirect, url_for, send_file, flash
from pymongo import MongoClient
import os
# import h5py
# import tensorflow as tf
# from tensorflow import keras
import PyPDF2
import numpy as np
import re
# from keras.preprocessing.text import Tokenizer
# from keras.utils import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bson.objectid import ObjectId
import spacy
import uuid
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import load
from sklearn.preprocessing import LabelEncoder
from spacy.matcher import Matcher


le = joblib.load('label_encoder.joblib')
# load the word_vectorizer object
word_vectorizer = load('word_vectorizer.joblib')

model = joblib.load('model1.joblib')

lemmatizer = WordNetLemmatizer()

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)


# load the model from HDF5 format
# model_h5 = tf.keras.models.load_model('my_model.h5')


app = Flask(__name__, static_folder='static')
app.secret_key = 'my_secret_key'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# connect to the MongoDB server
client = MongoClient('mongodb://localhost/')
db = client['mydatabase']
collection = db['users']
collection1 = db['job_vacancies']
collection2 = db['jobadmin']


# hard-coded admin credentials
admin_username = "admin"
admin_password = "password"


def clear_database():
    client = MongoClient('mongodb://localhost/')
    # Replace 'your_database_name' with the actual name of your database
    db = client['mydatabase']

    # Iterate through each collection in the database and drop it
    for collection_name in db.list_collection_names():
        db.drop_collection(collection_name)


# login page
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # check if user is a user or admin
        username = request.form['username']
        password = request.form['password']
        if username == admin_username and password == admin_password:
            session['user_type'] = 'admin'
            return redirect(url_for('admin'))
        else:
            user = collection.find_one(
                {'username': username, 'password': password})
            if user:
                session['user_type'] = 'user'
                return redirect(url_for('user'))
            else:
                return render_template('login.html', error='Invalid username or password')
    else:
        return render_template('login.html')

# user registration page


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # handle form submission
        username = request.form['username']
        password = request.form['password']
        # add the user to the MongoDB collection
        collection.insert_one({'username': username, 'password': password})
        flash('Registration successful, please log in.')
        # redirect to the login page
        return redirect(url_for('login'))
    else:
        return render_template('register.html')

# user page for uploading a CV and getting a prediction


@app.route('/upload_cv', methods=['GET', 'POST'])
def upload_cv():
    # check if user is a user
    if session.get('user_type') == 'user':
        if request.method == 'POST':
            # get the uploaded file
            file = request.files['cv']
            # save the file to disk
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            # extract text from PDF
            original_text = extract_text_from_pdf(filename)
            # preprocess text
            preprocessed_text = preprocess_text(original_text)
            score = score_resume(original_text)
            resume_features = word_vectorizer.transform([preprocessed_text])
            # Get the predicted category using the loaded model
            predicted_category = model.predict(resume_features)[0]
            # Inverse transform the predicted category to get the actual category name
            predicted_category_name = le.inverse_transform(
                [predicted_category])[0]

            return redirect(url_for('prediction', original_text=original_text, filename=filename, score=score, predicted_category_name=predicted_category_name))

        else:
            return render_template('user.html')
    else:
        return redirect(url_for('login'))


job_categories = {
    'Java Developer': ['Java programming', 'Spring Framework', 'RESTful API design', 'Database management'],
    "Data Science": ["Python", "R", "Machine Learning", "Data Analysis", "Data Visualization"],
    "Web Designing": ["HTML", "CSS", "JavaScript", "React", "Node.js"],
    "Digital Marketing": ["SEO", "SEM", "Social Media Marketing", "Content Marketing", "Email Marketing"],
    "Testing": ["Manual Testing", "Automation Testing", "Selenium", "TestNG", "Junit"],
    "DevOps Engineer": ["Jenkins", "Docker", "Kubernetes", "AWS", "Ansible"],
    "Python Developer": ["Python", "Django", "Flask", "RESTful API", "SQLAlchemy"],
    "HR": ["Recruiting", "Employee Relations", "Performance Management", "Compensation and Benefits", "HR Analytics"],
    "Hadoop": ["Hadoop", "MapReduce", "Hive", "Pig", "HBase"],
    "Blockchain": ["Ethereum", "Smart Contracts", "Hyperledger", "Solidity", "Cryptography"],
    "ETL Developer": ["ETL", "Data Warehousing", "Informatica", "Data Integration", "SQL"],
    "Operations Manager": ["Operations Management", "Project Management", "Lean Six Sigma", "Supply Chain Management", "Quality Assurance"],
    "Sales": ["Sales Management", "Business Development", "Account Management", "Lead Generation", "Salesforce"],
    "Mechanical Engineer": ["Mechanical Engineering", "CAD", "SolidWorks", "AutoCAD", "Thermodynamics"],
    "Arts": ["Fine Arts", "Digital Art", "Graphic Design", "Illustration", "Animation"],
    "Database": ["SQL", "MySQL", "Oracle", "NoSQL", "MongoDB"],
    "Electrical Engineering": ["Electrical Engineering", "MATLAB", "Analog Electronics", "Digital Electronics", "Microcontrollers"],
    "Health and fitness": ["Fitness Training", "Nutrition Counseling", "Yoga Instruction", "Personal Training", "Physical Therapy"],
    "PMO": ["Project Management", "Program Management", "Portfolio Management", "PMO Governance", "Agile"],
    "Business Analyst": ["Business Analysis", "Requirements Gathering", "Process Mapping", "Data Modeling", "Business Intelligence"],
    "DotNet Developer": [".NET", "C#", "ASP.NET", "MVC", "Web API"],
    "Automation Testing": ["Selenium", "Test Automation Frameworks", "Java", "TestNG", "Junit"],
    "Network Security Engineer": ["Network Security", "Firewalls", "Intrusion Detection", "Cisco", "Wireshark"],
    "SAP Developer": ["SAP", "ABAP", "SAP HANA", "SAP Fiori", "SAP UI5"],
    "Civil Engineer": ["Civil Engineering", "AutoCAD", "Revit", "Structural Engineering", "Construction Management"],
    "Advocate": ["Legal Research", "Litigation", "Corporate Law", "Intellectual Property", "Contract Law"]
}


# prediction page
@app.route('/prediction')
def prediction():
    filename = request.args.get('filename')
    original_text = request.args.get('original_text')
    url, email, entities, employment_info = entities_from_pdf(original_text)
    score = request.args.get('score')
    predicted_category_name = request.args.get('predicted_category_name')
    skills = job_categories[predicted_category_name]
    pdf_path = os.path.join(os.getcwd(), filename)
    session['pdfpath'] = pdf_path
    return render_template('prediction.html', skills=skills, score=score, predicted_category_name=predicted_category_name, url=url, email=email, entities=entities, employment_info=employment_info)


def score_resume(resume):
    # Parse the resume text using spaCy
    doc = nlp(resume)
    # Compute the number of sentences in the resume
    num_sentences = len(list(doc.sents))
    # Compute the number of words in the resume
    num_words = len(
        [token for token in doc if not token.is_punct and not token.is_space])
    # Compute the average length of a sentence in the resume
    avg_sentence_length = num_words / num_sentences
    # Compute the number of named entities (e.g., organizations, locations) in the resume
    num_named_entities = len(doc.ents)
    # Compute the score as a weighted average of the above features
    score = (0.4 * avg_sentence_length) + \
        (0.3 * num_sentences) + (0.3 * num_named_entities)
    score = min(100, max(1, score))
    return int(score)


@ app.route("/view_pdf")
def view_pdf():
    pdfpath = session.get('pdfpath', None)
    return send_file(f'{pdfpath}', mimetype='application/pdf')


# user page
@app.route('/user', methods=['GET', 'POST'])
def user():
    if session.get('user_type') == 'user':
        if request.method == 'GET':
            return render_template('user.html')
        elif request.method == 'POST':
            if 'cv' in request.files:
                cv = request.files['cv']
                # Save the CV to a file or database
                flash('CV uploaded successfully.')
            else:
                flash('Please upload a CV.')

            return render_template('user.html')

    else:
        return redirect(url_for('login'))


@app.route('/job_vacancies', methods=['GET', 'POST'])
def job_vacancies():
    vacancies = collection1.find({})

    if request.method == 'POST':
        # Get the uploaded file
        cv_file = request.files['cv']
        # Save the file to a local directory
        cv_filename = cv_file.filename
        cv_file.save(os.path.join(app.config['UPLOAD_FOLDER'], cv_filename))
        flash('CV uploaded successfully.')
        return redirect(url_for('job_vacancies', cv_filename=cv_filename))

    return render_template('job_vacancy.html', vacancies=vacancies)


@app.route('/check_eligibility/<vacancy_id>/', methods=['GET', 'POST'])
def check_eligibility(vacancy_id):
    if session.get('user_type') == 'user':
        if request.method == 'POST':
            # extract text from PDF
            cv_file = request.files['cv_file']
            name = request.form['name']
            filename = os.path.join(UPLOAD_FOLDER, cv_file.filename)
            cv_file.save(filename)
            pdf_text = extract_text_from_pdf(filename)
            # preprocess vacancy description and perform eligibility check
            vacancies = list(collection1.find())
            for vacancy in vacancies:
                if str(vacancy['_id']) == str(vacancy_id):
                    vacancy_values = [vacancy['title'],
                                      vacancy['description'], vacancy['location']]
                    vacancy_text = ''
                    for value in vacancy_values:
                        vacancy_text += str(value) + ' '
                    vacancy_text = preprocess_text(vacancy_text)
                    break
            else:
                flash('Vacancy not found.')
                return redirect(url_for('job_vacancies'))
            # preprocess the candidate's text and perform eligibility check
            pdf_text = preprocess_text(pdf_text)
            similarity = check_eligibility_from_text(pdf_text, vacancy_text)

            # Save candidate details and CV to the databases
            collection2.insert_one({'name': name, 'cv_file': filename,
                                   'similarity': similarity, 'vacancy_id': str(vacancy_id)})
            return render_template('eligibility_result.html', similarity=similarity)
        else:
            # handle the GET request for showing a form to upload CV
            vacancies = list(collection1.find())
            for vacancy in vacancies:
                if str(vacancy['_id']) == str(vacancy_id):
                    return render_template('job_vacancy.html', vacancy=vacancy)
            else:
                flash('Vacancy not found.')
                return redirect(url_for('job_vacancies'))
    else:
        flash('You need to be logged in as a user to apply for a job vacancy.')
        return redirect(url_for('login'))


# job vacancies admin page
@app.route('/jobadmin')
def jobadmin():
    vacancies = collection1.find({})
    applied = collection2.find({})
    vacancy_dict = {}
    for v in vacancies:
        vacancy_dict[str(v['_id'])] = {
            'title': v['title'], 'description': v['description'], 'location': v['location'], 'candidates': []}
    for a in applied:
        vacancy_id = str(a['vacancy_id'])
        if vacancy_id in vacancy_dict:
            vacancy_dict[vacancy_id]['candidates'].append(
                {'name': a['name'], 'similarity': a['similarity']})
    vacancy_list = [v for v in vacancy_dict.values()]
    return render_template('jobadmin.html', vacancy_list=vacancy_list)


def check_eligibility_from_text(text1, text2):
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Create TF-IDF vectors for the two texts
    tfidf1 = tfidf_vectorizer.fit_transform([text1])
    tfidf2 = tfidf_vectorizer.transform([text2])

    # Calculate cosine similarity between the two vectors
    cosine_similarity_matrix = cosine_similarity(tfidf1, tfidf2)

    # Round the cosine similarity score to 2 decimal places
    cosine_similarity_score = round(cosine_similarity_matrix[0][0], 2)

    # Return the cosine similarity score
    return (cosine_similarity_score)


# admin page
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if session.get('user_type') == 'admin':
        if request.method == 'GET':
            return render_template('admin.html')
        elif request.method == 'POST':
            title = request.form['title']
            description = request.form['description']
            location = request.form['location']
            # Insert the new job vacancy into the MongoDB collection
            collection1.insert_one({
                'title': title,
                'description': description,
                'location': location,
                '_id': str(uuid.uuid4())  # generate a unique ID as a string
            })
            flash('Job vacancy posted successfully.')

            return redirect(url_for('jobadmin'))

    else:
        return redirect(url_for('login'))


def extract_text_from_pdf(filename):

    with open(filename, 'rb') as f:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(f)

        # Extract text from all the pages
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    # Open the PDF file in read-binary mode


def entities_from_pdf(pdf_text):

    doc = nlp(pdf_text)
    matcher = Matcher(nlp.vocab)

    url_pattern = re.compile(r'(Website:|URL:)(.*)', re.IGNORECASE)
    email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+', re.IGNORECASE)

    url = ''
    email = ''

    # Search for name, URL, and email in the text
    match = url_pattern.search(pdf_text)
    if match:
        url = match.group(2).strip()

    match = email_pattern.search(pdf_text)
    if match:
        email = match.group().strip()

    # Define pattern for matching person entities
    pattern_person = [{'ENT_TYPE': 'PERSON'}]
    matcher.add('PERSON', [pattern_person])

    # Define pattern for matching organization entities
    pattern_org = [{'ENT_TYPE': 'ORG'}]
    matcher.add('ORG', [pattern_org])

    # Find matches for person and organization entities
    matches = matcher(doc)

    # Extract entities and employment information
    entities = []
    employment_info = []

    for match_id, start, end in matches:
        if doc[start:end].label_ == 'PERSON':
            entities.append(('name', doc[start:end].text))
        elif doc[start:end].label_ == 'ORG':
            entities.append(('organization', doc[start:end].text))
            # Look for patterns indicating employment duration
            for sent in doc.sents:
                if doc[start:end].text in sent.text:
                    # Look for phrases like "from [date]" or "for [duration]"
                    if 'from' in sent.text:
                        start_date = sent.text.split('from')[-1].strip()
                        if len(start_date.split()) == 3:
                            try:
                                start_year = int(start_date.split()[-1])
                                employment_info.append(
                                    ('duration', doc[start:end].text, start_year))
                            except ValueError:
                                pass
                    elif 'for' in sent.text:
                        duration = sent.text.split('for')[-1].strip()
                        if 'year' in duration or 'years' in duration:
                            try:
                                years = int(duration.split()[0])
                                employment_info.append(
                                    ('duration', doc[start:end].text, years))
                            except ValueError:
                                pass

    return url, email, entities, employment_info


def preprocess_text(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText


if __name__ == '__main__':
    app.run(debug=True)
