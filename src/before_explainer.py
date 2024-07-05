import numpy as np
import codecs
import operator
import json
from trainer import train_loader
from tester import test_loader, distance, load_ids_from_file, check_id_exists
import time
from datetime import datetime
import argparse

def load_entity_embeddings(check_file_path):
    entity_embeddings = {}
    entity_id_list = []
    with codecs.open(check_file_path, 'r', 'utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            entity_id = parts[0]
            entity_id = int(entity_id)
            entity_id_list.append(entity_id)
            embedding = list(map(float, parts[1].strip('[]').split(',')))
            entity_embeddings[entity_id] = embedding
    return entity_id_list, entity_embeddings

def check_id_exists(check_id, entity_id_list):
    if check_id in entity_id_list:
        return 1
    else:
        return 0

def preprocess_test_set(data_name, alpha, dim):
    check_file_path = f"../res/entity_{dim}dim_{data_name}_batch200"
    start_time = time.time()  # Start time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    _, _, train_triple = train_loader(f"../dataset/{data_name}/")  # Read train_triple for filtering training method

    entity_dict, relation_dict, test_triple = \
        test_loader(f"../res/entity_{dim}dim_{data_name}_batch200", f"../res/relation_{dim}dim_{data_name}_batch200", f"../dataset/{data_name}/test.txt")

    entity_id_list, _ = load_entity_embeddings(check_file_path)

    testing_set = {}
    test = 0

    for triple in test_triple:
        h = triple[0]  # int
        r = triple[1]
        t = triple[2]
        triple_id = (h, r, t)

        test_t_exists = check_id_exists(check_id=h, entity_id_list=entity_id_list)
        test_h_exists = check_id_exists(check_id=t, entity_id_list=entity_id_list)

        if test_t_exists and test_h_exists:
            # Only when both head and tail entities have embeddings, they will be added to the testing_set
            h_emb = entity_dict[h]
            r_emb = entity_dict[r]
            t_emb = entity_dict[t]
            triple_emb = [h_emb.tolist(), r_emb.tolist(), t_emb.tolist()]
            # Add key-value pair
            triple_id_str = f"{h},{r},{t}"  # Convert tuple to string
            testing_set[triple_id_str] = triple_emb
            test += 1
        else:
            continue

    print("the size of the testing set is ", test)
    print("finished testing set generating.")

    # Store testing_set into JSON file
    with open(f'rectified_test_{data_name}_set.json', 'w') as json_file:
        json.dump(testing_set, json_file, indent=4)

    print(f"testing set saved to rectified_test_{data_name}_set.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ExplainWithPrototype Parameters")
    parser.add_argument('--data_name', type=str, default='yago3-10', required=False, help='Name of the dataset')
    parser.add_argument('--alpha', type=float, default=0.5, required=False, help='Alpha value for embedding shifting')
    parser.add_argument('--dim', type=int, default=50, required=False, help='The number of embedding dim.')

    args = parser.parse_args()
    data_name = args.data_name
    alpha = args.alpha
    dim = args.dim

    preprocess_test_set(data_name, alpha, dim)
