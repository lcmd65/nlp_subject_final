def evaluate(model, test_data):
    correct = 0
    total = len(test_data)
    
    for text, annotation in test_data:
        doc = model(text)
        predicted_label = max(doc.cats, key=doc.cats.get)
        true_label = max(annotation['cats'], key=annotation['cats'].get)
        if predicted_label == true_label:
            correct += 1
    
    accuracy = correct / total
    return accuracy


def prepare_test_data(test_data):
    texts = [item["text"] for item in test_data]
    annotations = [{'cats': item["cats"]} for item in test_data]
    return list(zip(texts, annotations))