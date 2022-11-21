import gradio as gr
import numpy as np
from PIL import Image
import requests
import re

import hopsworks
import joblib
from consts import *

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("titanic_new", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_new_model.pkl")


def get_deck(cabin):
    if cabin is None:
        return 0
    deck = re.compile("([a-zA-Z]+)").search(cabin)
    if deck is not None:
        deck = deck.group()
    if deck in DECK:
        deck = DECK[deck]
    else:
        deck = 0
    return deck


def get_sex(sex):
    return GENDERS[sex]


def get_embarked(embarked):
    return PORTS[embarked]


def get_title(title):
    if title in TITLES_RARE:
        title = "Rare"
    if title not in TITLES:
        return 0
    return TITLES[title]


def get_age_class(age, p_class):
    return age * p_class


def get_relatives(sib_sp, parch):
    return sib_sp + parch


def get_not_alone(relatives):
    return 1 if relatives > 0 else 0


def get_fare_per_person(fare, relatives):
    return fare / (relatives + 1)


def titanic(p_class, sex, age, sib_sp, parch, fare, cabin, embarked, title):
    # Model input:
    # Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Deck,
    # Title, Age_Class, Relatives, Not_alone, Fare_Per_Person
    p_class = p_class + 1
    deck = get_deck(cabin)
    sex = get_sex(sex)
    embarked = get_embarked(embarked)
    title = get_title(title)
    age_class = get_age_class(age, p_class)
    relatives = get_relatives(sib_sp, parch)
    not_alone = get_not_alone(relatives)
    fare_per_person = get_fare_per_person(fare, relatives)
    input_list = [p_class, sex, age, sib_sp, parch, fare, embarked,
                  deck, title, age_class, relatives, not_alone, fare_per_person]
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
    img_name = "alive.png" if res[0] == 1 else "dead.png"
    img = Image.open(img_name)
    return img


demo = gr.Interface(
    fn=titanic,
    title="Titanic survival predictor",
    description="Experiment with the parameters to predict if the person would or would not survive.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Dropdown(choices=["Class 1", "Class 2", "Class 3"], type="index", label="Class", default="Class 1"),
        gr.inputs.Dropdown(choices=["male", "female"], type="value", label="Gender", default="male"),
        gr.inputs.Number(label="Age"),
        gr.inputs.Number(default=0, label="Number of sibling and spouses on board"),
        gr.inputs.Number(default=0, label="Number of children/parents on board"),
        gr.inputs.Number(label="Fare"),
        gr.inputs.Textbox(label="Cabin"),
        gr.inputs.Dropdown(choices=['S', 'C', 'Q'], label="Port of embarkation", default="S"),
        gr.inputs.Textbox(label="Title")
    ],
    outputs=gr.Image(type="pil"))

demo.launch()

titanic(1, "male", 22, 1, 0, 150, "C85", "S", "Mr")
