import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import csv
import warnings
import streamlit as st


st.set_page_config(page_title="Healthcare Chatbot", page_icon=":hospital:")

warnings.filterwarnings("ignore", category=DeprecationWarning)


def readn(nstr):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)  
        engine.setProperty('rate', 130)  
        engine.say(nstr)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Text-to-speech error: {e}")


@st.cache_data
def load_data():
    training = pd.read_csv('Training.csv')
    testing = pd.read_csv('Testing.csv')
    return training, testing


@st.cache_resource
def prepare_model():
    
    training, testing = load_data()
    
    
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']

    
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    
    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x_train, y_train)
    
    
    model = SVC()
    model.fit(x_train, y_train)
    
    return clf, model, le, cols, x, y


@st.cache_data
def load_dictionaries():
    
    description_list = {}
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]
    
    
    severity_dictionary = {}
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:
                try:
                    severity_dictionary[row[0]] = int(row[1])
                except ValueError:
                    continue
    
    
    precaution_dictionary = {}
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 5:
                precaution_dictionary[row[0]] = [row[1], row[2], row[3], row[4]]
    
    return description_list, severity_dictionary, precaution_dictionary


def main():
    
    clf, model, le, cols, X, y = prepare_model()
    description_list, severity_dictionary, precaution_dictionary = load_dictionaries()
    
    
    symptom_list = list(cols)

    
    st.title("🩺 Healthcare Diagnostic Chatbot")
    
    
    st.sidebar.header("Patient Information")
    patient_name = st.sidebar.text_input("Enter Your Name")
    
    
    st.subheader("Select Your Symptoms")
    selected_symptoms = st.multiselect(
        "Choose symptoms you are experiencing:", 
        symptom_list
    )
    
    
    days_of_symptoms = st.slider("For how many days have you been experiencing these symptoms?", 1, 30, 7)
    
    
    if st.button("Predict Possible Disease"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
            readn("Please select at least one symptom.")
        else:
            
            input_vector = np.zeros(len(symptom_list))
            for symptom in selected_symptoms:
                if symptom in symptom_list:
                    input_vector[symptom_list.index(symptom)] = 1
            
            
            prediction = clf.predict([input_vector])
            disease = le.inverse_transform(prediction)[0]
            
            
            st.success(f"Predicted Disease: {disease}")
            readn(f"Predicted Disease: {disease}")
            
            
            st.subheader("Disease Description")
            description = description_list.get(disease, "Description not available.")
            st.write(description)
            readn(description)
            
            
            st.subheader("Recommended Precautions")
            precautions = precaution_dictionary.get(disease, ["No specific precautions found."])
            for i, precaution in enumerate(precautions, 1):
                st.write(f"{i}. {precaution}")
                readn(f"Precaution {i}: {precaution}")
            
            
            severity_score = sum(severity_dictionary.get(symptom, 0) for symptom in selected_symptoms)
            severity_status = "High" if (severity_score * days_of_symptoms / (len(selected_symptoms) + 1)) > 13 else "Moderate"
            
            st.warning(f"Severity Assessment: {severity_status}")
            readn(f"Severity Assessment: {severity_status}")
            
            if severity_status == "High":
                st.warning("You should consult a doctor immediately.")
                readn("You should consult a doctor immediately.")
            else:
                st.info("Monitor your symptoms and take necessary precautions.")
                readn("Monitor your symptoms and take necessary precautions.")

   
    st.sidebar.markdown("---")
    st.sidebar.info("Disclaimer: This is a diagnostic aid, not a substitute for professional medical advice.")
    st.sidebar.markdown("AIML PROJECT MADE BY: ")
    st.sidebar.markdown("    SHEEN PANDITA 22CSU162")
    st.sidebar.markdown("    SHERIN BAIJU 22CSU163")
    st.sidebar.markdown("    SWATI KAR 22CSU174")


if __name__ == "__main__":
    main()
