import json
import re,math
from collections import defaultdict
import difflib
import unicodedata
from Levenshtein import distance as levenshtein_distance
from difflib import SequenceMatcher


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

def compare_annotations(manual_annotations, generated_annotations):
    results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    manual_entities = set(tuple(entity) for entity in manual_annotations['entities'])
    generated_entities = set(tuple(entity) for entity in generated_annotations['entities'])

    ## MHl - verze ktera nepocita s odlisnosti text v entitach
    #for entity in manual_entities.union(generated_entities):
    #    entity_type = entity[2]
    #    if entity in manual_entities and entity in generated_entities:
    #        results[entity_type]["tp"] += 1 # MHl - verze ktera nepocita s odlisnosti text v entitach
    #    elif entity in manual_entities:
    #        results[entity_type]["fn"] += 1
    #    else:
    #        results[entity_type]["fp"] += 1

    # -----------------------------ZZ verze ktera jeste nevic kontoluje, zda je text entit stejny
    for generated_entity in generated_entities:
        nasel = False
        for manual_entity in manual_entities:
            if generated_entity[2] == manual_entity[2]:
                #similar, message = compare_sentences(normalize_text(manual_entity[3]), normalize_text(generated_entity[3]))
                match, distance = find_closest_match(normalize_text(manual_entity[3]), normalize_text(generated_entity[3]))  # vysledky: nasla_e=852, nenasla_e=387
                if distance <= math.ceil(len(manual_entity[3]) * 0.25):
                    results[generated_entity[2]]["tp"] += 1
                    manual_entities.remove(manual_entity)
                    nasel = True
                    break
        if not nasel:
            results[generated_entity[2]]["fp"] += 1
    for manual_entity in manual_entities:
        results[manual_entity[2]]["fn"] +=  1
    # -----------------------------ZZ verze ktera jeste nevic kontoluje, zda je text entit stejny

    return results








def compare_sentences(sentence1, sentence2, similarity_threshold=0.9, max_end_diff_length=5):
    # Create a SequenceMatcher object
    matcher = difflib.SequenceMatcher(None, sentence1, sentence2)

    # Get the similarity ratio
    ratio = matcher.ratio()

    if ratio >= similarity_threshold:
        # If similarity is high, check the differences
        differences = list(matcher.get_opcodes())

        # Check if the only difference is at the end
        if len(differences) > 1 and differences[-1][0] in ['delete', 'replace', 'insert']:
            last_diff = differences[-1]
            if last_diff[0] == 'delete':
                end_diff_length = last_diff[4] - last_diff[3]
                missing_part = sentence1[last_diff[3]:last_diff[4]]
            elif last_diff[0] == 'insert':
                end_diff_length = last_diff[2] - last_diff[1]
                missing_part = sentence2[last_diff[1]:last_diff[2]]
            else:  # 'replace'
                end_diff_length = max(last_diff[4] - last_diff[3], last_diff[2] - last_diff[1])
                missing_part = f"'{sentence1[last_diff[3]:last_diff[4]]}' replaced with '{sentence2[last_diff[1]:last_diff[2]]}'"

            if end_diff_length <= max_end_diff_length:
                return True, f"Similar, difference at end: {missing_part}"
        elif len(differences) == 2 and differences[1][0] == 'equal':
            # This handles cases where the difference is at the start
            diff_part = sentence1[differences[0][3]:differences[0][4]]
            return True, f"Similar, difference at start: '{diff_part}'"

    # If we haven't returned by now, sentences are considered different
    return False, "Sentences are significantly different"


def evaluate_ner_per_entity(manual_data, generated_data, ignoreEntities):
    total_results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for manual, generated in zip(manual_data, generated_data):
        if manual[0] != generated[0]:
            # Check if the sentences are similar
            similar, message = compare_sentences(manual[0], generated[0])
            if not similar:
                print("Texts don't match. Evaluation requires the same base text.")
                print(message)
                #continue # MHl - preskakoval , pokud to nebylo dostatecne stejne
            else:
                print(f"Texts are similar: {message}")

        results = compare_annotations(manual[1], generated[1])
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


def main(ignoreEntities, input_type, json_annot, jsonl_file_path=None, txt_file_path=None):

    if input_type == 'json':

        file_name = jsonl_file_path.split('/')[-1]

        json_list = []
        question_list = []

        with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                json_list.append(json.loads(line)['response_content'])
                question_list.append(json.loads(line)['input_message']['messages'][1]['content'])

        test_data_eval = json_list

        with open(json_annot, 'r', encoding='utf-8') as jsonl_file:
            json_list_a = [json.loads(line)['messages'][2]['content'] for line in jsonl_file]

        test_data_annot = json_list_a

    elif input_type == 'txt':

        file_name = txt_file_path.split('/')[-1]


        with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
            txt_list = [line.strip() for line in txt_file]

        test_data_eval = txt_list

        with open(json_annot, 'r', encoding='utf-8') as jsonl_file:
            json_list_a = [json.loads(line)['messages'][2]['content'] for line in jsonl_file]

        test_data_annot = json_list_a

    converted_data_eval = []
    converted_data_annot = []

    for data in test_data_eval:
        converted_data_eval.append(convert_format(data))

    for data in test_data_annot:
        converted_data_annot.append(convert_format(data))

    results = evaluate_ner_per_entity(converted_data_annot, converted_data_eval, ignoreEntities)
    print(f"{file_name}")
    for entity_type, metrics in results.items():
        print(f"{entity_type}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()

    return results


if __name__ == "__main__":


    # pro 3tridy OCR QWEN ft7BftOCR - NER model chatgpt4mini NOft
    jsonl_file_path = 'gpt-4o-mini-t_api_responses_testBlindOCRqwen7BftOCR.jsonl'
    jsonl_annot_path = 'ner_training_data_testOCRqwen7BftOCR.jsonl'


    ignoreEntities = ["DATE", "Person", "I"]  # jake entity chci ignorovat pri vyhodnocovani

    input_type = 'json'
    resultCesta=jsonl_file_path.split('/')
    if len(resultCesta) > 1:
        result_file = resultCesta[0] + '/Results_' + resultCesta[1].replace('.jsonl', '.json')
    else:
        result_file = 'Results_' + jsonl_file_path.replace('.jsonl', '.json')

    result = main(ignoreEntities, input_type, jsonl_annot_path, jsonl_file_path=jsonl_file_path)
    with open(result_file, 'w', encoding='utf-8') as result_file:
        json.dump(result, result_file, indent=4)