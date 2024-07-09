import json
import spacy
import random
from spacy.util import minibatch, compounding
from spacy.training import Example


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def split_data(data, split_ratio=0.8):
    random.seed(1)
    random.shuffle(data)
    split_index = int(split_ratio * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data


def prepare_data(data):
    texts = [item["text"] for item in data]
    cats = [item["cats"] for item in data]
    spacy_data = list(zip(texts, [{'cats': cats} for cats in cats]))
    return spacy_data


def create_model():
    nlp = spacy.blank('en')
    textcat = nlp.add_pipe('textcat', last=True)
    textcat.add_label('normal')
    textcat.add_label('manipulative')
    return nlp


def train_model(nlp, train_data_spacy, n_iter=10):
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for i in range(n_iter):
            losses = {}
            random.shuffle(train_data_spacy)
            batches = minibatch(train_data_spacy, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = [Example.from_dict(nlp.make_doc(text), annotation) for text, annotation in zip(texts, annotations)]
                nlp.update(examples, sgd=optimizer, losses=losses)
            print(f'Iteration {i+1}: Losses - {losses}')


def save_model(nlp, output_dir='./model'):
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")
