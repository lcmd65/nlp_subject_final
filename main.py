from utils.function import *
from utils.evaluation import *

def build():
    file_path = 'dataset/dataset.json'
    data = load_data(file_path)
    train_data, test_data = split_data(data)
    train_data_spacy = prepare_data(train_data)
    test_data_spacy = prepare_test_data(test_data)

    nlp = create_model()
    train_model(nlp, train_data_spacy)
    save_model(nlp)

    accuracy = evaluate(nlp, test_data_spacy)
    print(f"Accuracy on test set: {accuracy}")
    
if __name__ == "__main__":
    build()