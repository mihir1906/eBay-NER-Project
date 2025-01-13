from simpletransformers.ner import NERModel, NERArgs
import torch

def main():
    model_args = NERArgs()

    # Device Configuration
    use_cuda = torch.cuda.is_available()
    model_args.device = "cuda" if use_cuda else "cpu"

    # Define the label list, including 'O'
    aspect_names = [
        'B-Modell', 'B-Marke', 'B-Produktlinie', 'B-Schuhschaft-Typ',
        'B-Farbe', 'B-US-Schuhgröße', 'B-EU-Schuhgröße', 'B-No Tag',
        'I-No Tag', 'B-Akzente', 'B-Abteilung', 'B-Produktart',
        'B-Muster', 'B-Stil', 'B-Zwischensohlen-Typ', 'B-Obermaterial',
        'B-Herstellernummer', 'B-Gewebeart', 'B-Verschluss', 'B-Erscheinungsjahr',
        'B-Anlass', 'I-Produktlinie', 'B-Aktivität', 'I-Abteilung',
        'B-Besonderheiten', 'B-Maßeinheit', 'B-Thema', 'B-UK-Schuhgröße',
        'B-Dämpfungsgrad', 'B-Herstellungsland und -region', 'I-Produktart',
        'B-Schuhweite', 'B-Jahreszeit', 'B-Laufsohlenmaterial',
        'B-Innensohlenmaterial', 'B-Obscure', 'I-Farbe', 'I-Modell',
        'B-Futtermaterial', 'B-Charakter', 'B-Stollentyp',
        'B-Charakter Familie', 'I-Marke', 'I-Zwischensohlen-Typ',
        'I-Aktivität', 'I-Anlass', 'I-Obscure', 'I-EU-Schuhgröße',
        'I-UK-Schuhgröße', 'I-Besonderheiten', 'I-Obermaterial'
    ]
    label_list = ['O'] + aspect_names

    #Load the saved best model
    loaded_model = NERModel(
        "xlmroberta",
        "outputs/best_model/",
        labels=label_list,
        use_cuda=use_cuda
    )

    # Make predictions on new data
    sample_sentence = input("Enter a tagline to perform NER on: ")

    predictions, raw_outputs = loaded_model.predict([sample_sentence])

    for sentence, prediction in zip(sample_sentence, predictions):
        print(f"Sentence: {sentence}")
        print(f"Predictions: {prediction}")
        print("\n") 

if __name__ == "__main__":
    main()