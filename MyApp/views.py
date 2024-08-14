from django.shortcuts import render
import json
from nltk.translate import AlignedSent, IBMModel2
import pandas as pd


# Prepare and preprocess the data
def load_and_preprocess_data(filepath):
    data = pd.read_excel(filepath)
    data = data.dropna()
    data['tokenized_filipino'] = data['FILIPINO'].str.lower().apply(lambda x: x.split())
    data['tokenized_teduray'] = data['TEDURAY'].str.lower().apply(lambda x: x.split())
    aligned_texts_filipino_to_teduray = [AlignedSent(fil, ted) for fil, ted in
                                         zip(data['tokenized_filipino'], data['tokenized_teduray'])]
    aligned_texts_teduray_to_filipino = [AlignedSent(ted, fil) for fil, ted in
                                         zip(data['tokenized_filipino'], data['tokenized_teduray'])]
    return data, aligned_texts_filipino_to_teduray, aligned_texts_teduray_to_filipino


training_filepath = 'static/training_datasets.xlsx'
training_data, aligned_texts_filipino_to_teduray, aligned_texts_teduray_to_filipino = load_and_preprocess_data(
    training_filepath)


# Load models and parameters (adjust the file paths as needed)
def load_model_parameters(filename):
    with open(filename, 'r') as f:
        params = json.load(f)
    return params


def reconstruct_model(params, aligned_texts):
    model = IBMModel2(aligned_texts, 5)
    model.translation_table = {
        tuple(k.split()): {tuple(sub_k.split()): v for sub_k, v in sub_v.items()}
        for k, sub_v in params['translation_table'].items()
    }
    return model


params_filipino_to_teduray = load_model_parameters('static/model_params_filipino_to_teduray.json')
params_teduray_to_filipino = load_model_parameters('static/model_params_teduray_to_filipino.json')

model_filipino_to_teduray = reconstruct_model(params_filipino_to_teduray, aligned_texts_filipino_to_teduray)
model_teduray_to_filipino = reconstruct_model(params_teduray_to_filipino, aligned_texts_teduray_to_filipino)


# Function to translate sentences
def translate_sentence(sentence, model):
    tokenized_sentence = sentence.lower().split()
    translation = []
    for word in tokenized_sentence:
        # Check if the word exists in the translation table
        translations = model.translation_table.get((word,), {})
        if translations:
            # Select the word with the highest probability
            translated_word = max(translations, key=translations.get)
            translation.append(' '.join(translated_word))
        else:
            # If the word is not found, append the original word
            translation.append(word)
    return ' '.join(translation)


# Function to detect language
def detect_language(sentence, training_data):
    tokenized_sentence = sentence.lower().split()
    filipino_words = set(training_data['tokenized_filipino'].sum())
    teduray_words = set(training_data['tokenized_teduray'].sum())

    # Count how many words are from each language
    filipino_count = sum(1 for word in tokenized_sentence if word in filipino_words)
    teduray_count = sum(1 for word in tokenized_sentence if word in teduray_words)

    # Determine the language based on word counts
    if filipino_count > teduray_count:
        return 'Filipino'
    elif teduray_count > filipino_count:
        return 'Teduray'
    else:
        return 'Unknown'


def index(request):
    context = {
        'translated_sentence': '',
        'detected_language': ''
    }
    if request.method == 'POST':
        input_sentence = request.POST.get('sentence', '')

        # Detect language
        detected_language = detect_language(input_sentence, training_data)

        # Translate based on detected language
        if detected_language == 'Filipino':
            translated_sentence = translate_sentence(input_sentence, model_filipino_to_teduray)
        elif detected_language == 'Teduray':
            translated_sentence = translate_sentence(input_sentence, model_teduray_to_filipino)
        else:
            translated_sentence = "Could not detect language."

        # Update context
        context.update({
            'input_sentence': input_sentence,
            'translated_sentence': translated_sentence,
            'detected_language': detected_language
        })

    return render(request, 'index.html', context)
