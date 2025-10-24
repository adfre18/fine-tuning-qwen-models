import json
import re,math, os
from collections import defaultdict
import difflib
import unicodedata
from Levenshtein import distance as levenshtein_distance
from difflib import SequenceMatcher

from tqdm import tqdm


def convert_format(input_text):
    # Find all annotated entities
    pattern = r'<(\w+)\s*>(.+?)</\1\s*>'
    entities = []

    # Remove annotations and collect entity information
    def replace_func(match):
        entity_type = match.group(1)
        entity_text = match.group(2)
        start = match.start() - sum(
            len(m.group()) - len(m.group(2)) for m in re.finditer(pattern, input_text[:match.start()]))
        end = start + len(entity_text)
        entities.append((start, end, entity_type, entity_text))
        return entity_text

    clean_text = re.sub(pattern, replace_func, input_text)

    # Create the output format
    return [clean_text, {'entities': entities}]


def calculate_metrics(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1



def normalize_text(text):
    """
    Normalize text by removing diacritics and converting to lowercase.
    """
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\.', ' ', text)  # Replace . by space
    text = re.sub(r'\,', ' ', text)  # Replace , by space
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower().strip()


def find_closest_match(ocr_text, target_text):
    """
    Find the substring in ocr_text that is closest to target_text using Levenshtein distance.

    Parameters:
        ocr_text (str): The text obtained from OCR.
        target_text (str): The text to search for.

    Returns:
        tuple: Closest substring and its Levenshtein distance to target_text.
    """

    min_distance = float('inf')
    closest_match = None

    # Iterate through all possible substrings
    for length in range(1, len(target_text) + 1):  # Allow shorter substrings
        for i in range(len(ocr_text) - length + 1):
            substring = ocr_text[i:i + length]
            distance = levenshtein_distance(substring, target_text)

            if distance < min_distance:
                min_distance = distance
                closest_match = substring

    return closest_match, min_distance


def compare_annotations(manual_entities, generated_entities, Const_distance_metrix):
    results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # -----------------------------ZZ verze ktera jeste nevic kontoluje, zda je text entit stejny
    for generated_entity in generated_entities:
        nasel = False
        for manual_entity in manual_entities:
            if generated_entity["entity"] == manual_entity["entity"]:
                match, distance = find_closest_match(normalize_text(manual_entity["text"]), normalize_text(generated_entity["text"]))  # vysledky: nasla_e=852, nenasla_e=387
                if distance <= math.ceil(len(manual_entity["text"]) * Const_distance_metrix):
                    results[generated_entity["entity"]]["tp"] += 1
                    manual_entities.remove(manual_entity)  #uz sem nasel a nechci ji znovu zapocitavat
                    nasel = True
                    break
        if not nasel:
            results[generated_entity["entity"]]["fp"] += 1
    for manual_entity in manual_entities:
        results[manual_entity["entity"]]["fn"] +=  1
    # -----------------------------ZZ verze ktera jeste nevic kontoluje, zda je text entit stejny (blizky podle Levenshtein < 0.25 deky originalu)

    return results


def evaluate_ner_per_entity(annotations, total_annotations, ModelNERtext, ignoreEntities, Const_distance_metrix):

    total_results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    startIndex = 0
    for index_json in tqdm(range(startIndex, total_annotations), desc="Processing annotations"):
        if (('attributes' in annotations['annotations'][index_json]) and ('TestTrainVal' in annotations['annotations'][index_json]['attributes'])
                and ('test' in annotations['annotations'][index_json]['attributes']['TestTrainVal'])):

            manual_entities = annotations['annotations'][index_json]['attributes']['ner_entities']
            generated_entities = annotations['annotations'][index_json]['attributes'][ModelNERtext]

            results = compare_annotations(manual_entities, generated_entities, Const_distance_metrix)

            for entity_type, counts in results.items():
                total_results[entity_type]["tp"] += counts["tp"]
                total_results[entity_type]["fp"] += counts["fp"]
                total_results[entity_type]["fn"] += counts["fn"]

    final_results = {}
    for entity_type, counts in total_results.items():
        if entity_type in ignoreEntities:
            continue
        precision, recall, f1 = calculate_metrics(counts["tp"], counts["fp"], counts["fn"])
        final_results[entity_type] = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        }

    # Calculate overall metrics
    overall_tp = sum(counts["tp"] for counts in total_results.values())
    overall_fp = sum(counts["fp"] for counts in total_results.values())
    overall_fn = sum(counts["fn"] for counts in total_results.values())
    overall_precision, overall_recall, overall_f1 = calculate_metrics(overall_tp, overall_fp, overall_fn)
    final_results["Overall"] = {
        "Precision": overall_precision,
        "Recall": overall_recall,
        "F1-score": overall_f1
    }

    return final_results


def main(json_annot, ModelNERtext, ignoreEntities, Const_distance_metrix):

    # otevri anotace v json formatu
    with open(json_annot, 'r', encoding='utf-8') as jsonl_file:
        data_annot = json.load(jsonl_file)
        total_annotations = len(data_annot['annotations'])
    print(f'Anotace ( pocet: {total_annotations} ) ze souboru {json_annot} načteny')

    results = evaluate_ner_per_entity(data_annot, total_annotations, ModelNERtext, ignoreEntities, Const_distance_metrix)


    for entity_type, metrics in results.items():
        print(f"{entity_type}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()

    return results


if __name__ == "__main__":

    ModelNERtext = "qwen7bftner-from-txt_and_image_NER_entities"
    jsonl_annot_path = 'Annotations//Qwen_data//NER_annotations_with_texts_2_TestTrainVal_Qwen7BftNER-from-TXT_and_image.json'
    #ModelNERtext = "qwen7bftner_NER_entities"
    #jsonl_annot_path = 'Annotations//Qwen_data//NER_annotations_with_texts_2_TestTrainVal_Qwen7BftNER.json'


    ignoreEntities = ["DATE", "BLBOST"] #jake entity chci ignorovat pri vyhodnocovani
    Const_distance_metrix = 0.25

    result = main(jsonl_annot_path, ModelNERtext, ignoreEntities, Const_distance_metrix)

    #uloz vysledky
    dir_name, file_name = os.path.split(jsonl_annot_path)
    new_file_name = 'Results_' + ModelNERtext + '_' + file_name
    result_file = os.path.join(dir_name, new_file_name)
    with open(result_file, 'w', encoding='utf-8') as result_file:
        json.dump(result, result_file, indent=4)
