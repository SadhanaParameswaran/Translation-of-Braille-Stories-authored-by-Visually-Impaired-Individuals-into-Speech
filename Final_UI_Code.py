import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pytesseract
import cv2
import os
from streamlit.components.v1 import html
from googletrans import Translator
import pyttsx3
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from gtts import gTTS
from io import BytesIO
from IPython.display import Audio, display
import io
import time
import glob

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load the model and tokenizer for text summarization
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Title and Icon
# Emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Character Recognition", page_icon=":speech_balloon:", layout="wide")

# --- HEADER SECTION ---
# Center-align the title using Markdown and CSS
st.markdown(
    """
    <h1 style="text-align: center;">Translation of Children's Braille Stories Authored by Visually Impaired Individuals into Speech in Malayalam and Tamil Languages</h1>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

with st.container():
    st.write("---")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

translator = Translator(service_urls=['translate.google.com'])

folder_path = 'D:/College_Semesters/6th Semester/Projects/NLP_SP/EnglishBooks'

BuntyandBubbly = ""
DontChangeTheWorld = ""
LearnFromMistakes = ""
MeetingAnimals = ""
MyBestFriend = ""
TheAntAndTheDove = ""
TheFoxAndTheStork = ""
TheHareAndTheTortoise = ""
TheMilkmaid = ""
TheThirstyCrow = ""
TheTravellersAndThePlaneTree = ""
YouAreBeautiful = ""

for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            if file_name == "BuntyandBubbly.txt":
                story1 = text
            elif file_name == "DontChangeTheWorld.txt":
                story2 = text
            elif file_name == "LearnFromMistakes.txt":
                story3 = text
            elif file_name == "MeetingAnimals.txt":
                story4 = text
            elif file_name == "MyBestFriend.txt":
                story5 = text
            elif file_name == "TheAntAndTheDove.txt":
                story6 = text
            elif file_name == "TheFoxAndTheStork.txt":
                story7 = text
            elif file_name == "TheHareAndTheTortoise.txt":
                story8 = text
            elif file_name == "TheMilkmaid.txt":
                story9 = text
            elif file_name == "TheThirstyCrow.txt":
                story10 = text
            elif file_name == "TheTravellersAndThePlaneTree.txt":
                story11 = text
            elif file_name == "YouAreBeautiful.txt":
                story12 = text

folder_path = 'D:/College_Semesters/6th Semester/Projects/NLP_SP/BrailleBooks'

BuntyandBubbly_Braille = ""
DontChangeTheWorld_Braille = ""
LearnFromMistakes_Braille = ""
MeetingAnimals_Braille = ""
MyBestFriend_Braille = ""
TheAntAndTheDove_Braille = ""
TheFoxAndTheStork_Braille = ""
TheHareAndTheTortoise_Braille = ""
TheMilkmaid_Braille = ""
TheThirstyCrow_Braille = ""
TheTravellersAndThePlaneTree_Braille = ""
YouAreBeautiful_Braille = ""

for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            if file_name == "BuntyandBubbly_Braille.txt":
                story1_braille = text
            elif file_name == "DontChangeTheWorld_Braille.txt":
                story2_braille = text
            elif file_name == "LearnFromMistakes_Braille.txt":
                story3_braille = text
            elif file_name == "MeetingAnimals_Braille.txt":
                story4_braille = text
            elif file_name == "MyBestFriend_Braille.txt":
                story5_braille = text
            elif file_name == "TheAntAndTheDove_Braille.txt":
                story6_braille = text
            elif file_name == "TheFoxAndTheStork_Braille.txt":
                story7_braille = text
            elif file_name == "TheHareAndTheTortoise_Braille.txt":
                story8_braille = text
            elif file_name == "TheMilkmaid_Braille.txt":
                story9_braille = text
            elif file_name == "TheThirstyCrow_Braille.txt":
                story10_braille = text
            elif file_name == "TheTravellersAndThePlaneTree_Braille.txt":
                story11_braille = text
            elif file_name == "YouAreBeautiful_Braille.txt":
                story12_braille = text

# # Translate the stories to Tamil
story1_tamil = translator.translate(story1, dest='ta').text
story2_tamil = translator.translate(story2, dest='ta').text
story3_tamil = translator.translate(story3, dest='ta').text
story4_tamil = translator.translate(story4, dest='ta').text
story5_tamil = translator.translate(story5, dest='ta').text
story6_tamil = translator.translate(story6, dest='ta').text
story7_tamil = translator.translate(story7, dest='ta').text
story8_tamil = translator.translate(story8, dest='ta').text
story9_tamil = translator.translate(story9, dest='ta').text
story10_tamil = translator.translate(story10, dest='ta').text
story11_tamil = translator.translate(story11, dest='ta').text
story12_tamil = translator.translate(story12, dest='ta').text

# # Translate the stories to Malayalam
story1_malayalam = translator.translate(story1, dest='ml').text
story2_malayalam = translator.translate(story2, dest='ml').text
story3_malayalam = translator.translate(story3, dest='ml').text
story4_malayalam = translator.translate(story4, dest='ml').text
story5_malayalam = translator.translate(story5, dest='ml').text
story6_malayalam = translator.translate(story6, dest='ml').text
story7_malayalam = translator.translate(story7, dest='ml').text
story8_malayalam = translator.translate(story8, dest='ml').text
story9_malayalam = translator.translate(story9, dest='ml').text
story10_malayalam = translator.translate(story10, dest='ml').text
story11_malayalam = translator.translate(story11, dest='ml').text
story12_malayalam = translator.translate(story12, dest='ml').text

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def generate_summary(story, language):
    inputs = tokenizer.encode(story, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs,
                                 max_length=120,  # Adjust the maximum length of the generated summary
                                 min_length=30,   # Adjust the minimum length of the generated summary
                                 num_beams=4,     # Adjust the number of beams for beam search
                                 length_penalty=1.0,  # Adjust the length penalty factor
                                 no_repeat_ngram_size=2,  # Avoid repeating n-grams in the summary
                                 early_stopping=True)   # Enable early stopping to end the generation early
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Generate the summary using T5 model
def generate_summary_T5(story, language):
    inputs = tokenizer.encode(story, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=120, min_length=30, num_beams=4, length_penalty=1.0)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

col1,col2,col3=st.columns(3)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

button_style = """  background-color: #E0BBE4;
                    color: black;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    display: block;
                    margin: 0 auto;
               """
col2.markdown(f'<button style="{button_style}">Choose a Language</button>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

col2.markdown(
    """
    <style>
    .small-file-uploader > label p {
        font-size: 12px;
        padding: 5px 10px;
        background-color: #FFC0CB;
        border-radius: 5px;
        border: 1px solid #ccc;
        line-height: 1;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define options for the dropdown
options = ["English", "Tamil", "Malayalam"]

# Create a dropdown with a default value
selected_option = col2.selectbox("", options)

# Perform backend action based on selected option
if selected_option == "English":
    # Backend action 
    col2.write("Selected English")

elif selected_option == "Tamil":
    # Backend action 
    col2.write("Selected Tamil")

elif selected_option == "Malayalam":
    # Backend action
    col2.write("Selected Malayalam")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

button_style = """  background-color: #E0BBE4;
                    color: black;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    display: block;
                    margin: 0 auto;
               """
col2.markdown(f'<button style="{button_style}">Choose a story</button>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

with st.container():
    st.write("---")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

col1, col2, col3, col4, col5, col6 = st.columns(6)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Open the image file from the folder
image_path1 = 'D:\\College_Semesters\\6th Semester\\Projects\\NLP_SP\\BooksPictures\\BuntyAndBubbly.png'
image1 = Image.open(image_path1)
# Display the image
col1.image(image1, use_column_width=True)
button1 = col1.button("Bunty and Bubbly")

image_path2 = 'D:\\College_Semesters\\6th Semester\\Projects\\NLP_SP\\BooksPictures\\DontChangeTheWorld.png'
image2 = Image.open(image_path2)
col2.image(image2, use_column_width=True)
button2 = col2.button("Dont change the world")

image_path3 = 'D:\\College_Semesters\\6th Semester\\Projects\\NLP_SP\\BooksPictures\\LearnFromMistakes.png'
image3 = Image.open(image_path3)
col3.image(image3, use_column_width=True)
button3 = col3.button("Learn from Mistakes")

image_path4 = 'D:\\College_Semesters\\6th Semester\\Projects\\NLP_SP\\BooksPictures\\MeetingAnimals.png'
image4 = Image.open(image_path4)
col4.image(image4, use_column_width=True)
button4 = col4.button("Meeting Animals")

image_path5 = 'D:\\College_Semesters\\6th Semester\\Projects\\NLP_SP\\BooksPictures\\MyBestFriend.png'
image5 = Image.open(image_path5)
col5.image(image5, use_column_width=True)
button5 = col5.button("My Best Friend")

image_path6 = 'D:\\College_Semesters\\6th Semester\\Projects\\NLP_SP\\BooksPictures\\TheAntAndTheDove.png'
image6 = Image.open(image_path6)
col6.image(image6, use_column_width=True)
button6 = col6.button("The Ant and the Dove")

image_path7 = 'D:\\College_Semesters\\6th Semester\\Projects\\NLP_SP\\BooksPictures\\TheFoxAndTheStork.png'
image7 = Image.open(image_path7)
col1.image(image7, use_column_width=True)
button7 = col1.button("The Fox and the Stork")

image_path8 = 'D:\\College_Semesters\\6th Semester\\Projects\\NLP_SP\\BooksPictures\\TheHareAndTheTortoise.png'
image8 = Image.open(image_path8)
col2.image(image8, use_column_width=True)
button8 = col2.button("The Hare and the Tortoise")

image_path9 = 'D:\\College_Semesters\\6th Semester\\Projects\\NLP_SP\\BooksPictures\\TheMilkmaid.png'
image9 = Image.open(image_path9)
col3.image(image9, use_column_width=True)
button9 = col3.button("The Milkmaid")

# Open the image file from the folder
image_path10 = 'D:\\College_Semesters\\6th Semester\\Projects\\NLP_SP\\BooksPictures\\TheThirstyCrow.png'
image10 = Image.open(image_path10)
col4.image(image10, use_column_width=True)
button10 = col4.button("The Thirsty Crow")

image_path11 = 'D:\\College_Semesters\\6th Semester\\Projects\\NLP_SP\\BooksPictures\\TheTravellersAndThePlaneTree.png'
image11 = Image.open(image_path11)
col5.image(image11, use_column_width=True)
button11 = col5.button("The Travellers and the Plane Tree")

image_path12 = 'D:\\College_Semesters\\6th Semester\\Projects\\NLP_SP\\BooksPictures\\YouAreBeautiful.png'
image12 = Image.open(image_path12)
col6.image(image12, use_column_width=True)
button12 = col6.button("You are beautiful")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

with st.container():
    st.write("---")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if selected_option == "English":
    col1.markdown("<h2 style='text-align: center;'>Story in English</h2>", unsafe_allow_html=True)
    col2.markdown("<h2 style='text-align: center;'>Summary</h2>", unsafe_allow_html=True)
    col3.markdown("<h2 style='text-align: center;'>Story in Braille</h2>", unsafe_allow_html=True)
    
    #PRINTING STORY IN ENGLISH
    if button1:
        col1.write(story1)
    
    if button2:
        col1.write(story2)

    if button3:
        col1.write(story3)

    if button4:
        col1.write(story4)

    if button5:
        col1.write(story5)

    if button6:
        col1.write(story6)

    if button7:
        col1.write(story7)
    
    if button8:
        col1.write(story8)

    if button9:
        col1.write(story9)

    if button10:
        col1.write(story10)

    if button11:
        col1.write(story11)

    if button12:
        col1.write(story12)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    options_models = ["BART Model", "T5 Model"]  # Define options for the dropdown
    selected_option_model = col2.selectbox("", options_models)  # Create a dropdown with a default value
    if selected_option_model == "BART Model": # Perform backend action based on selected option
        col2.write("Selected BART Model")  # Backend action 
    elif selected_option_model == "F5 Model":
        col2.write("Selected F5 Model")  # Backend action 

    #PRINTING SUMMARY
    if selected_option_model == "BART Model":
        if button1:
            col2.write(generate_summary(story1, "English"))

        if button2:
            col2.write(generate_summary(story2, "English"))

        if button3:
            col2.write(generate_summary(story3, "English"))

        if button4:
            col2.write(generate_summary(story4, "English"))

        if button5:
            col2.write(generate_summary(story5, "English"))

        if button6:
            col2.write(generate_summary(story6, "English"))

        if button7:
            col2.write(generate_summary(story7, "English"))

        if button8:
            col2.write(generate_summary(story8, "English"))

        if button9:
            col2.write(generate_summary(story9, "English"))

        if button10:
            col2.write(generate_summary(story10, "English"))

        if button11:
            col2.write(generate_summary(story11, "English"))

        if button12:
            col2.write(generate_summary(story12, "English"))

    elif selected_option_model == "T5 Model":
        if button1:
            col2.write(generate_summary_T5(story1, "English"))

        if button2:
            col2.write(generate_summary_T5(story2, "English"))

        if button3:
            col2.write(generate_summary_T5(story3, "English"))

        if button4:
            col2.write(generate_summary_T5(story4, "English"))

        if button5:
            col2.write(generate_summary_T5(story5, "English"))

        if button6:
            col2.write(generate_summary_T5(story6, "English"))

        if button7:
            col2.write(generate_summary_T5(story7, "English"))

        if button8:
            col2.write(generate_summary_T5(story8, "English"))

        if button9:
            col2.write(generate_summary_T5(story9, "English"))

        if button10:
            col2.write(generate_summary_T5(story10, "English"))

        if button11:
            col2.write(generate_summary_T5(story11, "English"))

        if button12:
            col2.write(generate_summary_T5(story12, "English"))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #PRINTING STORY IN BRAILLE
    if button1:
        col3.write(story1_braille)
    
    if button2:
        col3.write(story2_braille)

    if button3:
        col3.write(story3_braille)
    
    if button4:
        col3.write(story4_braille)

    if button5:
        col3.write(story5_braille)
    
    if button6:
        col3.write(story6_braille)

    if button7:
        col3.write(story7_braille)
    
    if button8:
        col3.write(story8_braille)

    if button9:
        col3.write(story9_braille)
    
    if button10:
        col3.write(story10_braille)

    if button11:
        col3.write(story11_braille)
    
    if button12:
        col3.write(story12_braille)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if selected_option == "Tamil":
    col1.markdown("<h2 style='text-align: center;'>Story in Tamil</h2>", unsafe_allow_html=True)
    col2.markdown("<h2 style='text-align: center;'>Summary</h2>", unsafe_allow_html=True)
    col3.markdown("<h2 style='text-align: center;'>Story in Braille</h2>", unsafe_allow_html=True)
    
    #PRINTING STORY IN TAMIL
    if button1:
        col1.write(story1_tamil)

    if button2:
        col1.write(story2_tamil)

    if button3:
        col1.write(story3_tamil)

    if button4:
        col1.write(story4_tamil)

    if button5:
        col1.write(story5_tamil)

    if button6:
        col1.write(story6_tamil)

    if button7:
        col1.write(story7_tamil)

    if button8:
        col1.write(story8_tamil)

    if button9:
        col1.write(story9_tamil)

    if button10:
        col1.write(story10_tamil)

    if button11:
        col1.write(story11_tamil)

    if button12:
        col1.write(story12_tamil)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    options_models = ["BART Model", "T5 Model"]  # Define options for the dropdown
    selected_option_model = col2.selectbox("", options_models)  # Create a dropdown with a default value
    if selected_option_model == "BART Model": # Perform backend action based on selected option
        col2.write("Selected BART Model")  # Backend action 
    elif selected_option_model == "F5 Model":
        col2.write("Selected F5 Model")  # Backend action 

    #PRINTING SUMMARY
    if selected_option_model == "BART Model":
        if button1:
            col2.write(generate_summary(story1_tamil, "Tamil"))

        if button2:
            col2.write(generate_summary(story2_tamil, "Tamil"))

        if button3:
            col2.write(generate_summary(story3_tamil, "Tamil"))

        if button4:
            col2.write(generate_summary(story4_tamil, "Tamil"))

        if button5:
            col2.write(generate_summary(story5_tamil, "Tamil"))

        if button6:
            col2.write(generate_summary(story6_tamil, "Tamil"))

        if button7:
            col2.write(generate_summary(story7_tamil, "Tamil"))

        if button8:
            col2.write(generate_summary(story8_tamil, "Tamil"))

        if button9:
            col2.write(generate_summary(story9_tamil, "Tamil"))

        if button10:
            col2.write(generate_summary(story10_tamil, "Tamil"))

        if button11:
            col2.write(generate_summary(story11_tamil, "Tamil"))

        if button12:
            col2.write(generate_summary(story12_tamil, "Tamil"))
    
    elif selected_option_model == "T5 Model":
        if button1:
            col2.write(generate_summary_T5(story1_tamil, "Tamil"))

        if button2:
            col2.write(generate_summary_T5(story2_tamil, "Tamil"))

        if button3:
            col2.write(generate_summary_T5(story3_tamil, "Tamil"))

        if button4:
            col2.write(generate_summary_T5(story4_tamil, "Tamil"))

        if button5:
            col2.write(generate_summary_T5(story5_tamil, "Tamil"))

        if button6:
            col2.write(generate_summary_T5(story6_tamil, "Tamil"))

        if button7:
            col2.write(generate_summary_T5(story7_tamil, "Tamil"))

        if button8:
            col2.write(generate_summary_T5(story8_tamil, "Tamil"))

        if button9:
            col2.write(generate_summary_T5(story9_tamil, "Tamil"))

        if button10:
            col2.write(generate_summary_T5(story10_tamil, "Tamil"))

        if button11:
            col2.write(generate_summary_T5(story11_tamil, "Tamil"))

        if button12:
            col2.write(generate_summary_T5(story12_tamil, "Tamil"))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #PRINTING STORY IN BRAILLE
    if button1:
        col3.write(story1_braille)
    
    if button2:
        col3.write(story2_braille)

    if button3:
        col3.write(story3_braille)
    
    if button4:
        col3.write(story4_braille)

    if button5:
        col3.write(story5_braille)
    
    if button6:
        col3.write(story6_braille)

    if button7:
        col3.write(story7_braille)
    
    if button8:
        col3.write(story8_braille)

    if button9:
        col3.write(story9_braille)
    
    if button10:
        col3.write(story10_braille)

    if button11:
        col3.write(story11_braille)
    
    if button12:
        col3.write(story12_braille)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if selected_option == "Malayalam":
    col1.markdown("<h2 style='text-align: center;'>Story in Malayalam</h2>", unsafe_allow_html=True)
    col2.markdown("<h2 style='text-align: center;'>Summary</h2>", unsafe_allow_html=True)
    col3.markdown("<h2 style='text-align: center;'>Story in Braille</h2>", unsafe_allow_html=True)

    #PRINTING STORY IN TAMIL
    if button1:
        col1.write(story1_malayalam)

    if button2:
        col1.write(story2_malayalam)
    
    if button3:
        col1.write(story3_malayalam)

    if button4:
        col1.write(story4_malayalam)

    if button5:
        col1.write(story5_malayalam)

    if button6:
        col1.write(story6_malayalam)

    if button7:
        col1.write(story7_malayalam)

    if button8:
        col1.write(story8_malayalam)
    
    if button9:
        col1.write(story9_malayalam)

    if button10:
        col1.write(story10_malayalam)

    if button11:
        col1.write(story11_malayalam)

    if button12:
        col1.write(story12_malayalam)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    options_models = ["BART Model", "T5 Model"]  # Define options for the dropdown
    selected_option_model = col2.selectbox("", options_models)  # Create a dropdown with a default value
    if selected_option_model == "BART Model": # Perform backend action based on selected option
        col2.write("Selected BART Model")  # Backend action 
    elif selected_option_model == "F5 Model":
        col2.write("Selected F5 Model")  # Backend action 
        
    #PRINTING SUMMARY
    if selected_option_model == "BART Model":
        if button1:
            col2.write(generate_summary(story1_malayalam, "Malayalam"))

        if button2:
            col2.write(generate_summary(story2_malayalam, "Malayalam"))

        if button3:
            col2.write(generate_summary(story3_malayalam, "Malayalam"))

        if button4:
            col2.write(generate_summary(story4_malayalam, "Malayalam"))

        if button5:
            col2.write(generate_summary(story5_malayalam, "Malayalam"))

        if button6:
            col2.write(generate_summary(story6_malayalam, "Malayalam"))

        if button7:
            col2.write(generate_summary(story7_malayalam, "Malayalam"))

        if button8:
            col2.write(generate_summary(story8_malayalam, "Malayalam"))

        if button9:
            col2.write(generate_summary(story9_malayalam, "Malayalam"))

        if button10:
            col2.write(generate_summary(story10_malayalam, "Malayalam"))

        if button11:
            col2.write(generate_summary(story11_malayalam, "Malayalam"))

        if button12:
            col2.write(generate_summary(story12_malayalam, "Malayalam"))

    elif selected_option_model == "T5 Model":
        if button1:
            col2.write(generate_summary_T5(story1_malayalam, "Malayalam"))

        if button2:
            col2.write(generate_summary_T5(story2_malayalam, "Malayalam"))

        if button3:
            col2.write(generate_summary_T5(story3_malayalam, "Malayalam"))

        if button4:
            col2.write(generate_summary_T5(story4_malayalam, "Malayalam"))

        if button5:
            col2.write(generate_summary_T5(story5_malayalam, "Malayalam"))

        if button6:
            col2.write(generate_summary_T5(story6_malayalam, "Malayalam"))

        if button7:
            col2.write(generate_summary_T5(story7_malayalam, "Malayalam"))

        if button8:
            col2.write(generate_summary_T5(story8_malayalam, "Malayalam"))

        if button9:
            col2.write(generate_summary_T5(story9_malayalam, "Malayalam"))

        if button10:
            col2.write(generate_summary_T5(story10_malayalam, "Malayalam"))

        if button11:
            col2.write(generate_summary_T5(story11_malayalam, "Malayalam"))

        if button12:
            col2.write(generate_summary_T5(story12_malayalam, "Malayalam"))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    #PRINTING STORY IN BRAILLE
    if button1:
        col3.write(story1_braille)
    
    if button2:
        col3.write(story2_braille)

    if button3:
        col3.write(story3_braille)
    
    if button4:
        col3.write(story4_braille)

    if button5:
        col3.write(story5_braille)
    
    if button6:
        col3.write(story6_braille)

    if button7:
        col3.write(story7_braille)
    
    if button8:
        col3.write(story8_braille)

    if button9:
        col3.write(story9_braille)
    
    if button10:
        col3.write(story10_braille)

    if button11:
        col3.write(story11_braille)
    
    if button12:
        col3.write(story12_braille)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------