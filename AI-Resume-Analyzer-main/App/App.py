# Developed by dnoobnerd [https://dnoobnerd.netlify.app]    Made with Streamlit


###### Packages Used ######
import spacy
import streamlit as st # core package used in this project
import pandas as pd
import base64, random
import time,datetime
import pymysql
import os
import socket
import platform
import geocoder
import secrets
import io,random
import plotly.express as px # to create visualisations at the admin session
import plotly.graph_objects as go
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from geopy.geocoders import Nominatim
from textblob import TextBlob
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
# libraries used to parse the pdf files
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
# pre stored data for prediction purposes
from Courses import ds_course,web_course,android_course,ios_course,uiux_course,resume_videos,interview_videos
import nltk
nltk.download('stopwords')


###### Preprocessing functions ######


# Generates a link allowing the data in a given panda dataframe to be downloaded in csv format 
def get_csv_download_link(df,filename,text):
    csv = df.to_csv(index=False)
    ## bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()      
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Reads Pdf file and check_extractable
def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    ## close open handles
    converter.close()
    fake_file_handle.close()
    return text


# show uploaded file path to view pdf_display
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# course recommendations which has data already loaded from Courses.py
def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    c = 0
    rec_course = []
    ## slider to choose from range 1-10
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


###### Database Stuffs ######


# sql connector
connection = pymysql.connect(host='localhost',user='root',password='qwerty12345r',db='cv')
cursor = connection.cursor()


# inserting miscellaneous data, fetched results, prediction and recommendation into user_data table
def insert_data(sec_token,ip_add,host_name,dev_user,os_name_ver,latlong,city,state,country,act_name,act_mail,act_mob,name,email,res_score,timestamp,no_of_pages,reco_field,cand_level,skills,recommended_skills,courses,pdf_name):
    DB_table_name = 'user_data'
    insert_sql = "insert into " + DB_table_name + """
    values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (str(sec_token),str(ip_add),host_name,dev_user,os_name_ver,str(latlong),city,state,country,act_name,act_mail,act_mob,name,email,str(res_score),timestamp,str(no_of_pages),reco_field,cand_level,skills,recommended_skills,courses,pdf_name)
    cursor.execute(insert_sql, rec_values)
    connection.commit()


# inserting feedback data into user_feedback table
def insertf_data(feed_name,feed_email,feed_score,comments,Timestamp):
    DBf_table_name = 'user_feedback'
    insertfeed_sql = "insert into " + DBf_table_name + """
    values (0,%s,%s,%s,%s,%s)"""
    rec_values = (feed_name, feed_email, feed_score, comments, Timestamp)
    cursor.execute(insertfeed_sql, rec_values)
    connection.commit()


###### Setting Page Configuration (favicon, Logo, Title) ######


st.set_page_config(
   page_title="CVision",
   page_icon='./Logo/logo.png',
)


###### Main function run() ######


def run():
    
    # (Logo, Heading, Sidebar etc)
    img = Image.open('./Logo/RESUME.png')
    st.image(img)
    st.sidebar.markdown("# Choose Something...")
    activities = ["User", "Feedback", "About", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    link = '' 
    st.sidebar.markdown(link, unsafe_allow_html=True)
    

    ###### Creating Database and Table ######


    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)


    # Create table user_data and user_feedback
    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                    sec_token varchar(20) NOT NULL,
                    ip_add varchar(50) NULL,
                    host_name varchar(50) NULL,
                    dev_user varchar(50) NULL,
                    os_name_ver varchar(50) NULL,
                    latlong varchar(50) NULL,
                    city varchar(50) NULL,
                    state varchar(50) NULL,
                    country varchar(50) NULL,
                    act_name varchar(50) NOT NULL,
                    act_mail varchar(50) NOT NULL,
                    act_mob varchar(20) NOT NULL,
                    Name varchar(500) NOT NULL,
                    Email_ID VARCHAR(500) NOT NULL,
                    resume_score VARCHAR(8) NOT NULL,
                    Timestamp VARCHAR(50) NOT NULL,
                    Page_no VARCHAR(5) NOT NULL,
                    Predicted_Field BLOB NOT NULL,
                    User_level BLOB NOT NULL,
                    Actual_skills BLOB NOT NULL,
                    Recommended_skills BLOB NOT NULL,
                    Recommended_courses BLOB NOT NULL,
                    pdf_name varchar(50) NOT NULL,
                    PRIMARY KEY (ID)
                    );
                """
    cursor.execute(table_sql)


    DBf_table_name = 'user_feedback'
    tablef_sql = "CREATE TABLE IF NOT EXISTS " + DBf_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                        feed_name varchar(50) NOT NULL,
                        feed_email VARCHAR(50) NOT NULL,
                        feed_score VARCHAR(5) NOT NULL,
                        comments VARCHAR(100) NULL,
                        Timestamp VARCHAR(50) NOT NULL,
                        PRIMARY KEY (ID)
                    );
                """
    cursor.execute(tablef_sql)


    ###### CODE FOR CLIENT SIDE (USER) ######

    if choice == 'User':
        
        # Collecting Miscellaneous Information
        act_name = st.text_input('Name*')
        act_mail = st.text_input('Mail*')
        act_mob  = st.text_input('Mobile Number*')
        sec_token = secrets.token_urlsafe(12)
        host_name = socket.gethostname()
        ip_add = socket.gethostbyname(host_name)
        dev_user = os.getlogin()
        os_name_ver = platform.system() + " " + platform.release()
        g = geocoder.ip('me')
        latlong = g.latlng
        geolocator = Nominatim(user_agent="http")
        location = geolocator.reverse(latlong, language='en')
        address = location.raw['address']
        cityy = address.get('city', '')
        statee = address.get('state', '')
        countryy = address.get('country', '')  
        city = cityy
        state = statee
        country = countryy


        # Upload Resume
        st.markdown('''<h5 style='text-align: left; color: white;'> Upload Your Resume, And Get Smart Recommendations</h5>''',unsafe_allow_html=True)
        
        ## file upload in pdf format
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file is not None:
            with st.spinner('Hang On While We Cook Magic For You...'):
                time.sleep(4)
        
            ### saving the uploaded resume to folder
            save_image_path = './Uploaded_Resumes/'+pdf_file.name
            pdf_name = pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)

            ### parsing and extracting whole resume 
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            if resume_data:
                
                ## Get the whole resume data into resume_text
                resume_text = pdf_reader(save_image_path)

                ## Showing Analyzed data from (resume_data)
                st.header("**Resume Analysis 🤘**")
                st.success("Hello "+ resume_data['name'])
                st.subheader("**Your Basic info 👀**")
                try:
                    st.text('Name: '+resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Degree: '+str(resume_data['degree']))                    
                    st.text('Resume pages: '+str(resume_data['no_of_pages']))

                except:
                    pass
                ## Predicting Candidate Experience Level 

                # Load pre-trained job prediction model and vectorizer

                with open("experience_model.pkl", "rb") as exp_model_file:
                    experience_model = pickle.load(exp_model_file)
                with open("experience_vectorizer.pkl", "rb") as vec_model_file:
                    experience_vectorizer = pickle.load(vec_model_file)

                def predict_experience_level(resume_text):
                    text_vectorized = experience_vectorizer.transform([resume_text])
                    return experience_model.predict(text_vectorized)[0]
                
                experience_colors = {"Fresher": "#d73b5c", "Intermediate": "#1ed760", "Experienced": "#fba171"}
                experience_level = predict_experience_level(resume_text)
                st.markdown(f'''<h4 style='text-align: left; color: {experience_colors.get(experience_level, "#000")};'>You are at {experience_level} level!</h4>''', unsafe_allow_html=True)

                ### Trying with different possibilities
                # cand_level = ''
                # if resume_data['no_of_pages'] < 1:                
                #     cand_level = "NA"
                #     st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'>You are at Fresher level!</h4>''',unsafe_allow_html=True)
                
                # #### if internship then intermediate level
                # elif 'INTERNSHIP' in resume_text:
                #     cand_level = "Intermediate"
                #     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                # elif 'INTERNSHIPS' in resume_text:
                #     cand_level = "Intermediate"
                #     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                # elif 'Internship' in resume_text:
                #     cand_level = "Intermediate"
                #     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                # elif 'Internships' in resume_text:
                #     cand_level = "Intermediate"
                #     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                
                # #### if Work Experience/Experience then Experience level
                # elif 'EXPERIENCE' in resume_text:
                #     cand_level = "Experienced"
                #     st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
                # elif 'WORK EXPERIENCE' in resume_text:
                #     cand_level = "Experienced"
                #     st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
                # elif 'Experience' in resume_text:
                #     cand_level = "Experienced"
                #     st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
                # elif 'Work Experience' in resume_text:
                #     cand_level = "Experienced"
                #     st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)
                # else:
                #     cand_level = "Fresher"
                #     st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at Fresher level!!''',unsafe_allow_html=True)


                ## Skills Analyzing and Recommendation

                # ✅ Load trained skills model and vectorizer
                with open("skills_model.pkl", "rb") as model_file:
                    skills_model = pickle.load(model_file)

                with open("skills_vectorizer.pkl", "rb") as vec_file:
                    skills_vectorizer = pickle.load(vec_file)

                with open("job_model.pkl", "rb") as job_model_file:
                    job_model = pickle.load(job_model_file)

                with open("job_vectorizer.pkl", "rb") as job_vec_file:
                    job_vectorizer = pickle.load(job_vec_file)

                # ✅ Extract skills from resume text
                def extract_resume_skills(resume_text):
                    nlp = spacy.load("en_core_web_sm")
                    skill_keywords = set([
                        "python", "machine learning", "deep learning", "tensorflow", "keras", "pytorch",
                        "data science", "sql", "java", "c++", "html", "css", "javascript", "flask", "react",
                        "django", "android", "ios", "swift", "kotlin", "ui/ux", "figma", "adobe photoshop"
                    ])
                    doc = nlp(resume_text.lower())
                    extracted_skills = {token.text for token in doc if token.text in skill_keywords}
                    return list(extracted_skills)

                # ✅ Predict missing skills & courses
                def recommend_skills_and_courses(user_skills, predicted_position):
                    user_vector = skills_vectorizer.transform([", ".join(user_skills)])
                    distances, indices = skills_model.kneighbors(user_vector)

                    recommended_skills = set()
                    for idx in indices[0]:
                        similar_skills = skills_vectorizer.inverse_transform(skills_model._fit_X[idx])[0]
                        recommended_skills.update(similar_skills)

                    recommended_skills = list(recommended_skills - set(user_skills))

                    # Recommend courses based on predicted job position
                    course_mapping = {
                        "Data Science": ds_course,
                        "Web Development": web_course,
                        "Android Development": android_course,
                        "IOS Development": ios_course,
                        "UI/UX": uiux_course
                    }
                    reco_field = predicted_position  # Use predicted job position as reco_field
                    rec_course = course_mapping.get(predicted_position, ["No courses available"])

                    return recommended_skills, reco_field, rec_course

                # ✅ Predict Job Position
                def predict_job_position(resume_text):
                    resume_vector = job_vectorizer.transform([resume_text])
                    predicted_position = job_model.predict(resume_vector)[0]
                    return predicted_position

                # ✅ Display Recommendations in Streamlit
                st.subheader("**Skills, Job Position & Course Recommendations 💡**")

                # Slider to choose number of course recommendations
                num_courses = st.slider("Select number of course recommendations", min_value=1, max_value=10, value=5)

                # Assume `resume_text` is already provided from earlier processing
                extracted_skills = extract_resume_skills(resume_text)
                st_tags(label="### Extracted Skills from Resume", text="These are the skills found in your resume", value=extracted_skills, key="resume_skills")

                predicted_position = predict_job_position(resume_text)
                recommended_skills, reco_field, rec_course = recommend_skills_and_courses(extracted_skills, predicted_position)

                st.subheader("🏆 **Predicted Job Position**")
                st.success(f"Based on your resume, your most suitable job position is: **{predicted_position}**")

                if recommended_skills:
                    st.success(f"**Our analysis suggests you are looking for {reco_field} jobs.**")
                    st_tags(label="### Recommended Skills for You", text="Boost your resume", value=recommended_skills, key="skills_reco")
                    st.markdown(f"""<h5 style='text-align: left; color: #1ed760;'>Adding these skills will improve your chances of getting a {reco_field} job!</h5>""", unsafe_allow_html=True)

                    st.subheader("📚 **Recommended Courses**")
                    if rec_course and rec_course != ["No courses available"]:
                        for course in rec_course[:num_courses]:
                            st.markdown(f"- {course}")
                    else:
                        st.warning("No relevant courses found. Try adding more skills related to your field!")
                else:
                    st.warning("No additional skill recommendations at the moment. Keep updating your skills!")



                # st.subheader("**Skills Recommendation 💡**")
                
                # ### Current Analyzed Skills
                # keywords = st_tags(label='### Your Current Skills',
                # text='See our skills recommendation below',value=resume_data['skills'],key = '1  ')

                # ### Keywords for Recommendations
                # ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep Learning','flask','streamlit']
                # web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress','javascript', 'angular js', 'C#', 'Asp.net', 'flask']
                # android_keyword = ['android','android development','flutter','kotlin','xml','kivy']
                # ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode']
                # uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','solid','grasp','user research','user experience']
                # n_any = ['english','communication','writing', 'microsoft office', 'leadership','customer management', 'social media']
                # ### Skill Recommendations Starts                
                # recommended_skills = []
                # reco_field = ''
                # rec_course = ''

                # ### condition starts to check skills from keywords and predict field
                # for i in resume_data['skills']:
                
                #     #### Data science recommendation
                #     if i.lower() in ds_keyword:
                #         print(i.lower())
                #         reco_field = 'Data Science'
                #         st.success("** Our analysis says you are looking for Data Science Jobs.**")
                #         recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow',"Flask",'Streamlit']
                #         recommended_keywords = st_tags(label='### Recommended skills for you.',
                #         text='Recommended skills generated from System',value=recommended_skills,key = '2')
                #         st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job</h5>''',unsafe_allow_html=True)
                #         # course recommendation
                #         rec_course = course_recommender(ds_course)
                #         break

                #     #### Web development recommendation
                #     elif i.lower() in web_keyword:
                #         print(i.lower())
                #         reco_field = 'Web Development'
                #         st.success("** Our analysis says you are looking for Web Development Jobs **")
                #         recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
                #         recommended_keywords = st_tags(label='### Recommended skills for you.',
                #         text='Recommended skills generated from System',value=recommended_skills,key = '3')
                #         st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                #         # course recommendation
                #         rec_course = course_recommender(web_course)
                #         break

                #     #### Android App Development
                #     elif i.lower() in android_keyword:
                #         print(i.lower())
                #         reco_field = 'Android Development'
                #         st.success("** Our analysis says you are looking for Android App Development Jobs **")
                #         recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                #         recommended_keywords = st_tags(label='### Recommended skills for you.',
                #         text='Recommended skills generated from System',value=recommended_skills,key = '4')
                #         st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                #         # course recommendation
                #         rec_course = course_recommender(android_course)
                #         break

                #     #### IOS App Development
                #     elif i.lower() in ios_keyword:
                #         print(i.lower())
                #         reco_field = 'IOS Development'
                #         st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                #         recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
                #         recommended_keywords = st_tags(label='### Recommended skills for you.',
                #         text='Recommended skills generated from System',value=recommended_skills,key = '5')
                #         st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                #         # course recommendation
                #         rec_course = course_recommender(ios_course)
                #         break

                #     #### Ui-UX Recommendation
                #     elif i.lower() in uiux_keyword:
                #         print(i.lower())
                #         reco_field = 'UI-UX Development'
                #         st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                #         recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
                #         recommended_keywords = st_tags(label='### Recommended skills for you.',
                #         text='Recommended skills generated from System',value=recommended_skills,key = '6')
                #         st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''',unsafe_allow_html=True)
                #         # course recommendation
                #         rec_course = course_recommender(uiux_course)
                #         break

                #     #### For Not Any Recommendations
                #     elif i.lower() in n_any:
                #         print(i.lower())
                #         reco_field = 'NA'
                #         st.warning("** Currently our tool only predicts and recommends for Data Science, Web, Android, IOS and UI/UX Development**")
                #         recommended_skills = ['No Recommendations']
                #         recommended_keywords = st_tags(label='### Recommended skills for you.',
                #         text='Currently No Recommendations',value=recommended_skills,key = '6')
                #         st.markdown('''<h5 style='text-align: left; color: #092851;'>Maybe Available in Future Updates</h5>''',unsafe_allow_html=True)
                #         # course recommendation
                #         rec_course = "Sorry! Not Available for this Field"
                #         break


                ## Resume Scorer & Resume Writing Tips
                # ✅ Load trained resume scoring model and vectorizer

                with open("resume_scorer.pkl", "rb") as model_file:
                    resume_model = pickle.load(model_file)

                with open("resume_vectorizer.pkl", "rb") as vec_file:
                    resume_vectorizer = pickle.load(vec_file)

                # ✅ Predict Resume Score
                def predict_resume_score(resume_text):
                    text_vectorized = resume_vectorizer.transform([resume_text])
                    score = resume_model.predict(text_vectorized)[0]
                    return round(score)

                # ✅ Identify Missing Sections
                def check_resume_sections(resume_text):
                    sections = {
                        "Objective/Summary": ["objective", "summary"],
                        "Education": ["education", "school", "college"],
                        "Experience": ["experience", "work experience"],
                        "Internships": ["internship", "internships"],
                        "Skills": ["skills", "skill"],
                        "Hobbies": ["hobbies", "hobby"],
                        "Interests": ["interests", "interest"],
                        "Achievements": ["achievements", "achievement"],
                        "Certifications": ["certifications", "certification"],
                        "Projects": ["projects", "project"]
                    }
                    missing_sections = []
                    for section, keywords in sections.items():
                        if not any(keyword.lower() in resume_text.lower() for keyword in keywords):
                            missing_sections.append(section)
                    return missing_sections

                # ✅ Display Resume Score and Missing Sections
                st.subheader("**Resume Score & Writing Tips 📝**")
                resume_score = predict_resume_score(resume_text)
                st.markdown(f"""<h4 style='text-align: center; color: #d73b5c;'>Your Resume Score: {resume_score}/100</h4>""", unsafe_allow_html=True)
                st.progress(resume_score / 100)

                missing_sections = check_resume_sections(resume_text)
                if missing_sections:
                    st.warning("Your resume is missing the following key sections:")
                    for section in missing_sections:
                        st.markdown(f"<h5 style='text-align: left; color: white;'>[-] Please add {section}. This will improve your resume quality.</h5>", unsafe_allow_html=True)
                else:
                    st.success("Great job! Your resume covers all essential sections.")


                # st.subheader("**Resume Tips & Ideas 🥂**")
                # resume_score = 0
                
                # ### Predicting Whether these key points are added to the resume
                # if 'Objective' or 'Summary' in resume_text:
                #     resume_score = resume_score+6
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective/Summary</h4>''',unsafe_allow_html=True)                
                # else:
                #     st.markdown('''<h5 style='text-align: left; color: white;'>[-] Please add your career objective, it will give your career intension to the Recruiters.</h4>''',unsafe_allow_html=True)

                # if 'Education' or 'School' or 'College'  in resume_text:
                #     resume_score = resume_score + 12
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Education Details</h4>''',unsafe_allow_html=True)
                # else:
                #     st.markdown('''<h5 style='text-align: left; color: white;'>[-] Please add Education. It will give Your Qualification level to the recruiter</h4>''',unsafe_allow_html=True)

                # if 'EXPERIENCE' in resume_text:
                #     resume_score = resume_score + 16
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Experience</h4>''',unsafe_allow_html=True)
                # elif 'Experience' in resume_text:
                #     resume_score = resume_score + 16
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Experience</h4>''',unsafe_allow_html=True)
                # else:
                #     st.markdown('''<h5 style='text-align: left; color: white;'>[-] Please add Experience. It will help you to stand out from crowd</h4>''',unsafe_allow_html=True)

                # if 'INTERNSHIPS'  in resume_text:
                #     resume_score = resume_score + 6
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                # elif 'INTERNSHIP'  in resume_text:
                #     resume_score = resume_score + 6
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                # elif 'Internships'  in resume_text:
                #     resume_score = resume_score + 6
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                # elif 'Internship'  in resume_text:
                #     resume_score = resume_score + 6
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Internships</h4>''',unsafe_allow_html=True)
                # else:
                #     st.markdown('''<h5 style='text-align: left; color: white;'>[-] Please add Internships. It will help you to stand out from crowd</h4>''',unsafe_allow_html=True)

                # if 'SKILLS'  in resume_text:
                #     resume_score = resume_score + 7
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                # elif 'SKILL'  in resume_text:
                #     resume_score = resume_score + 7
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                # elif 'Skills'  in resume_text:
                #     resume_score = resume_score + 7
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                # elif 'Skill'  in resume_text:
                #     resume_score = resume_score + 7
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                # else:
                #     st.markdown('''<h5 style='text-align: left; color: white;'>[-] Please add Skills. It will help you a lot</h4>''',unsafe_allow_html=True)

                # if 'HOBBIES' in resume_text:
                #     resume_score = resume_score + 4
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''',unsafe_allow_html=True)
                # elif 'Hobbies' in resume_text:
                #     resume_score = resume_score + 4
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''',unsafe_allow_html=True)
                # else:
                #     st.markdown('''<h5 style='text-align: left; color: white;'>[-] Please add Hobbies. It will show your personality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''',unsafe_allow_html=True)

                # if 'INTERESTS'in resume_text:
                #     resume_score = resume_score + 5
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Interest</h4>''',unsafe_allow_html=True)
                # elif 'Interests'in resume_text:
                #     resume_score = resume_score + 5
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Interest</h4>''',unsafe_allow_html=True)
                # else:
                #     st.markdown('''<h5 style='text-align: left; color: white;'>[-] Please add Interest. It will show your interest other that job.</h4>''',unsafe_allow_html=True)

                # if 'ACHIEVEMENTS' in resume_text:
                #     resume_score = resume_score + 13
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements </h4>''',unsafe_allow_html=True)
                # elif 'Achievements' in resume_text:
                #     resume_score = resume_score + 13
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements </h4>''',unsafe_allow_html=True)
                # else:
                #     st.markdown('''<h5 style='text-align: left; color: white;'>[-] Please add Achievements. It will show that you are capable for the required position.</h4>''',unsafe_allow_html=True)

                # if 'CERTIFICATIONS' in resume_text:
                #     resume_score = resume_score + 12
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                # elif 'Certifications' in resume_text:
                #     resume_score = resume_score + 12
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                # elif 'Certification' in resume_text:
                #     resume_score = resume_score + 12
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                # else:
                #     st.markdown('''<h5 style='text-align: left; color: white;'>[-] Please add Certifications. It will show that you have done some specialization for the required position.</h4>''',unsafe_allow_html=True)

                # if 'PROJECTS' in resume_text:
                #     resume_score = resume_score + 19
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                # elif 'PROJECT' in resume_text:
                #     resume_score = resume_score + 19
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                # elif 'Projects' in resume_text:
                #     resume_score = resume_score + 19
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                # elif 'Project' in resume_text:
                #     resume_score = resume_score + 19
                #     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                # else:
                #     st.markdown('''<h5 style='text-align: left; color: white;'>[-] Please add Projects. It will show that you have done work related the required position or not.</h4>''',unsafe_allow_html=True)

                # st.subheader("**Resume Score 📝**")
                
                # st.markdown(
                #     """
                #     <style>
                #         .stProgress > div > div > div > div {
                #             background-color: #d73b5c;
                #         }
                #     </style>""",
                #     unsafe_allow_html=True,
                # )

                ### Score Bar
                score = 0
                for percent_complete in range(resume_score):
                    score +=1

                ### Score
                st.success('** Your Resume Writing Score: ' + str(score)+'**')
                st.warning("** Note: This score is calculated based on the content that you have in your Resume. **")

                # print(str(sec_token), str(ip_add), (host_name), (dev_user), (os_name_ver), (latlong), (city), (state), (country), (act_name), (act_mail), (act_mob), resume_data['name'], resume_data['email'], str(resume_score), timestamp, str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']), str(recommended_skills), str(rec_course), pdf_name)


                ### Getting Current Date and Time
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)


                ## Calling insert_data to add all the data into user_data                
                insert_data(str(sec_token), str(ip_add), (host_name), (dev_user), (os_name_ver), (latlong), (city), (state), (country), (act_name), (act_mail), (act_mob), resume_data['name'], resume_data['email'], str(resume_score), timestamp, str(resume_data['no_of_pages']), reco_field, experience_level, str(resume_data['skills']), str(recommended_skills), str(rec_course), pdf_name)

                ## Recommending Resume Writing Video
                st.header("**Bonus Video for Resume Writing Tips💡**")
                resume_vid = random.choice(resume_videos)
                st.video(resume_vid)

                ## Recommending Interview Preparation Video
                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = random.choice(interview_videos)
                st.video(interview_vid)

                # ✅ Load spaCy model
                nlp = spacy.load("en_core_web_sm")

                # ✅ Function to correct resume text while preserving formatting
                def correct_resume_text(resume_text):
                    doc = nlp(resume_text)
                    corrected_text = []

                    for token in doc:
                        if token.text.isalpha():  # Only correct words (ignore punctuation & numbers)
                            corrected_word = str(TextBlob(token.text).correct())  # Spell check
                            corrected_text.append(corrected_word if corrected_word != token.text else token.text)
                        else:
                            corrected_text.append(token.text)  # Keep punctuation, spaces, and numbers unchanged

                    return " ".join(corrected_text)  # Preserve original spacing

                # ✅ Function to generate a PDF that maintains structure
                def generate_pdf(corrected_text):
                    pdf_buffer = io.BytesIO()
                    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
                    pdf.setFont("Helvetica", 12)

                    y_position = 750  # Start position for text
                    line_height = 20   # Space between lines
                    max_lines_per_page = 35  # Number of lines per page

                    for line in corrected_text.split("\n"):
                        if y_position <= 50:  # Prevents text from cutting off at the bottom
                            pdf.showPage()
                            pdf.setFont("Helvetica", 12)
                            y_position = 750  # Reset position for the new page

                        pdf.drawString(50, y_position, line)
                        y_position -= line_height

                    pdf.save()
                    pdf_buffer.seek(0)  # Move to start of the buffer
                    return pdf_buffer

                # ✅ Streamlit UI for Resume Correction & Download
                st.subheader("📄 **Automatic Resume Correction & PDF Download**")

                corrected_resume = correct_resume_text(resume_text)

                st.text_area("Corrected Resume", corrected_resume, height=300)

                pdf_file = generate_pdf(corrected_resume)

                # Provide a download button for the corrected resume as a PDF
                st.download_button(label="📥 Download Corrected Resume (PDF)",
                                data=pdf_file,
                                file_name="Corrected_Resume.pdf",
                                mime="application/pdf")

                ## On Successful Result 
                #st.balloons()

            else:
                st.error('Something went wrong..')                


    ###### CODE FOR FEEDBACK SIDE ######
    elif choice == 'Feedback':   
        
        # timestamp 
        ts = time.time()
        cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        timestamp = str(cur_date+'_'+cur_time)

        # Feedback Form
        with st.form("my_form"):
            st.write("Feedback form")            
            feed_name = st.text_input('Name')
            feed_email = st.text_input('Email')
            feed_score = st.slider('Rate Us From 1 - 5', 1, 5)
            comments = st.text_input('Comments')
            Timestamp = timestamp        
            submitted = st.form_submit_button("Submit")
            if submitted:
                ## Calling insertf_data to add dat into user feedback
                insertf_data(feed_name,feed_email,feed_score,comments,Timestamp)    
                ## Success Message 
                st.success("Thanks! Your Feedback was recorded.") 
                ## On Successful Submit
                st.balloons()    


        # query to fetch data from user feedback table
        query = 'select * from user_feedback'        
        plotfeed_data = pd.read_sql(query, connection)                        


        # fetching feed_score from the query and getting the unique values and total value count 
        labels = plotfeed_data.feed_score.unique()
        values = plotfeed_data.feed_score.value_counts()


        # plotting pie chart for user ratings
        st.subheader("**Past User Rating's**")
        fig = px.pie(values=values, names=labels, title="Chart of User Rating Score From 1 - 5", color_discrete_sequence=px.colors.sequential.Aggrnyl)
        st.plotly_chart(fig)


        #  Fetching Comment History
        cursor.execute('select feed_name, comments from user_feedback')
        plfeed_cmt_data = cursor.fetchall()

        st.subheader("**User Comment's**")
        dff = pd.DataFrame(plfeed_cmt_data, columns=['User', 'Comment'])
        st.dataframe(dff, width=1000)

    
    ###### CODE FOR ABOUT PAGE ######
    elif choice == 'About':   

        st.subheader("**About The Tool - AI RESUME ANALYZER**")

        st.markdown('''

        <p align='justify'>
            A tool which parses information from a resume using natural language processing and finds the keywords, cluster them onto sectors based on their keywords. And lastly show recommendations, predictions, analytics to the applicant based on keyword matching.
        </p>

        <p align="justify">
            <b>How to use it: -</b> <br/><br/>
            <b>User -</b> <br/>
            In the Side Bar choose yourself as user and fill the required fields and upload your resume in pdf format.<br/>
            Just sit back and relax our tool will do the magic on it's own.<br/><br/>
            <b>Feedback -</b> <br/>
            A place where user can suggest some feedback about the tool.<br/><br/>
            <b>Admin -</b> <br/>
            For login use <b>admin</b> as username and <b>admin@resume-analyzer</b> as password.<br/>
            It will load all the required stuffs and perform analysis.
        </p><br/><br/>

        ''',unsafe_allow_html=True)  


    ###### CODE FOR ADMIN SIDE (ADMIN) ######
    else:
        st.success('Welcome to Admin Side')

        #  Admin Login
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')

        if st.button('Login'):
            
            ## Credentials 
            if ad_user == 'admin' and ad_password == 'admin':
                
                ### Fetch miscellaneous data from user_data(table) and convert it into dataframe
                cursor.execute('''SELECT ID, ip_add, resume_score, convert(Predicted_Field using utf8), convert(User_level using utf8), city, state, country from user_data''')
                datanalys = cursor.fetchall()
                plot_data = pd.DataFrame(datanalys, columns=['Idt', 'IP_add', 'resume_score', 'Predicted_Field', 'User_Level', 'City', 'State', 'Country'])
                
                ### Total Users Count with a Welcome Message
                values = plot_data.Idt.count()
                st.success("Welcome Deepak ! Total %d " % values + " User's Have Used Our Tool : )")                
                
                ### Fetch user data from user_data(table) and convert it into dataframe
                cursor.execute('''SELECT ID, sec_token, ip_add, act_name, act_mail, act_mob, convert(Predicted_Field using utf8), Timestamp, Name, Email_ID, resume_score, Page_no, pdf_name, convert(User_level using utf8), convert(Actual_skills using utf8), convert(Recommended_skills using utf8), convert(Recommended_courses using utf8), city, state, country, latlong, os_name_ver, host_name, dev_user from user_data''')
                data = cursor.fetchall()                

                st.header("**User's Data**")
                df = pd.DataFrame(data, columns=['ID', 'Token', 'IP Address', 'Name', 'Mail', 'Mobile Number', 'Predicted Field', 'Timestamp',
                                                 'Predicted Name', 'Predicted Mail', 'Resume Score', 'Total Page',  'File Name',   
                                                 'User Level', 'Actual Skills', 'Recommended Skills', 'Recommended Course',
                                                 'City', 'State', 'Country', 'Lat Long', 'Server OS', 'Server Name', 'Server User',])
                
                ### Viewing the dataframe
                st.dataframe(df)
                
                ### Downloading Report of user_data in csv file
                st.markdown(get_csv_download_link(df,'User_Data.csv','Download Report'), unsafe_allow_html=True)

                ### Fetch feedback data from user_feedback(table) and convert it into dataframe
                cursor.execute('''SELECT * from user_feedback''')
                data = cursor.fetchall()

                st.header("**User's Feedback Data**")
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Feedback Score', 'Comments', 'Timestamp'])
                st.dataframe(df)

                ### query to fetch data from user_feedback(table)
                query = 'select * from user_feedback'
                plotfeed_data = pd.read_sql(query, connection)                        

                ### Analyzing All the Data's in pie charts

                # fetching feed_score from the query and getting the unique values and total value count 
                labels = plotfeed_data.feed_score.unique()
                values = plotfeed_data.feed_score.value_counts()
                
                # Pie chart for user ratings
                st.subheader("**User Rating's**")
                fig = px.pie(values=values, names=labels, title="Chart of User Rating Score From 1 - 5 🤗", color_discrete_sequence=px.colors.sequential.Aggrnyl)
                st.plotly_chart(fig)

                # fetching Predicted_Field from the query and getting the unique values and total value count                 
                labels = plot_data.Predicted_Field.unique()
                values = plot_data.Predicted_Field.value_counts()

                # Pie chart for predicted field recommendations
                st.subheader("**Pie-Chart for Predicted Field Recommendation**")
                fig = px.pie(df, values=values, names=labels, title='Predicted Field according to the Skills 👽', color_discrete_sequence=px.colors.sequential.Aggrnyl_r)
                st.plotly_chart(fig)

                # fetching User_Level from the query and getting the unique values and total value count                 
                labels = plot_data.User_Level.unique()
                values = plot_data.User_Level.value_counts()

                # Pie chart for User's👨‍💻 Experienced Level
                st.subheader("**Pie-Chart for User's Experienced Level**")
                fig = px.pie(df, values=values, names=labels, title="Pie-Chart 📈 for User's 👨‍💻 Experienced Level", color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig)

                # fetching resume_score from the query and getting the unique values and total value count                 
                labels = plot_data.resume_score.unique()                
                values = plot_data.resume_score.value_counts()

                # Pie chart for Resume Score
                st.subheader("**Pie-Chart for Resume Score**")
                fig = px.pie(df, values=values, names=labels, title='From 1 to 100 💯', color_discrete_sequence=px.colors.sequential.Agsunset)
                st.plotly_chart(fig)

                # fetching IP_add from the query and getting the unique values and total value count 
                labels = plot_data.IP_add.unique()
                values = plot_data.IP_add.value_counts()

                # Pie chart for Users
                st.subheader("**Pie-Chart for Users App Used Count**")
                fig = px.pie(df, values=values, names=labels, title='Usage Based On IP Address 👥', color_discrete_sequence=px.colors.sequential.matter_r)
                st.plotly_chart(fig)

                # fetching City from the query and getting the unique values and total value count 
                labels = plot_data.City.unique()
                values = plot_data.City.value_counts()

                # Pie chart for City
                st.subheader("**Pie-Chart for City**")
                fig = px.pie(df, values=values, names=labels, title='Usage Based On City 🌆', color_discrete_sequence=px.colors.sequential.Jet)
                st.plotly_chart(fig)

                # fetching State from the query and getting the unique values and total value count 
                labels = plot_data.State.unique()
                values = plot_data.State.value_counts()

                # Pie chart for State
                st.subheader("**Pie-Chart for State**")
                fig = px.pie(df, values=values, names=labels, title='Usage Based on State 🚉', color_discrete_sequence=px.colors.sequential.PuBu_r)
                st.plotly_chart(fig)

                # fetching Country from the query and getting the unique values and total value count 
                labels = plot_data.Country.unique()
                values = plot_data.Country.value_counts()

                # Pie chart for Country
                st.subheader("**Pie-Chart for Country**")
                fig = px.pie(df, values=values, names=labels, title='Usage Based on Country 🌏', color_discrete_sequence=px.colors.sequential.Purpor_r)
                st.plotly_chart(fig)

            ## For Wrong Credentials
            else:
                st.error("Wrong ID & Password Provided")

# Calling the main (run()) function to make the whole process run
run()
