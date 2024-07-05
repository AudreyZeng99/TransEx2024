import numpy as np
import codecs
import operator
import json
from trainer import train_loader
from tester import test_loader, distance, load_ids_from_file, check_id_exists
# from transE_speed import data_loader, entity2id, relation2id
import time
# from collections import defaultdict
from datetime import datetime
# from itertools import islice
import argparse
import json

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def read_and_process_json(file_path):
    # Read the JSON file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    test_head_emb = {}
    test_relation_emb = {}
    test_tail_emb = {}
    test_triple = {}
    test_entity_emb = {}

    # Process each unit
    for key, value in data.items():
        hid, rid, tid = map(int, key.split(','))
        triple = (hid, rid, tid)
        list1, list2, list3 = value

        test_head_emb[hid] = list1
        test_relation_emb[rid] = list2
        test_tail_emb[tid] = list3
        test_triple[triple] = [list1, list2, list3]

    for key, value in test_head_emb.items():
        test_entity_emb[key] = value
    for key, value in test_tail_emb.items():
        if key not in test_entity_emb:
            test_entity_emb[key] = value

    return test_triple, test_entity_emb, test_head_emb, test_relation_emb, test_tail_emb


def save_to_txt(file_path, data_dict):
    with open(file_path, 'w') as f:
        for key, value in data_dict.items():
            f.write(f"{key}\t{value}\n")


class ExplainWithPrototype:
    def __init__(self, entity_dict, relation_dict, train_triple, test_triple, test_entity_emb, check_file_path, dim,
                 alpha, mode, isFit=True):
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.test_triple = test_triple
        self.test_entity_dict = test_entity_emb
        self.train_triple = train_triple
        print(len(self.test_entity_dict), len(self.relation_dict), len(self.test_triple), len(self.train_triple))
        self.check_file_path = check_file_path
        self.check_entity_emb_dict = {}
        self.explain_triple = test_triple
        self.isFit = isFit
        self.dim = dim
        self.mode = mode

        self.head_emb_shifted = entity_dict
        self.relation_emb_shifted = relation_dict
        self.tail_emb_shifted = entity_dict

        self.alpha = alpha
        self.prototype = ()

        self._hits1 = 0
        self._MRR = 0
        self.hits1 = 0
        self.MRR = 0

        self.skip = 0
        self.skip_t = 0
        self.skip_h = 0
        self.skip_r = 0

    def prototype_generator(self):
        print("*************************************")
        print("Prototype generating...")
        counter = 0
        h_p = np.zeros(self.dim)  # Initialize a zero vector with dimension dim
        r_p = np.zeros(self.dim)  # Initialize a zero vector with dimension dim
        t_p = np.zeros(self.dim)  # Initialize a zero vector with dimension dim

        for train_h_id, train_r_id, train_t_id in self.train_triple:
            h_emb = self.entity_dict[train_h_id]
            r_emb = self.relation_dict[train_r_id]
            t_emb = self.entity_dict[train_t_id]

            h_p += h_emb
            r_p += r_emb
            t_p += t_emb

            counter += 1
            # print("counter = ",counter)
        h_p /= counter
        r_p /= counter
        t_p /= counter

        self.prototype = (h_p, r_p, t_p)
        print("Successfully get prototype!")

    def plot_prototypes(self):
        h_p, r_p, t_p = self.prototype
        vectors = np.array([h_p, r_p, t_p])

        # Use PCA to reduce to 2D
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(vectors)

        # Plot in 2D
        plt.figure()
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], color=['red', 'green', 'blue'])
        for i, txt in enumerate(['h_p', 'r_p', 't_p']):
            plt.annotate(txt, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
        plt.title('Prototypes in Semantic Space (2D)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True)
        plt.show()

    def embedding_shifting(self):
        step = 1
        start = time.time()
        print("Phase1: Embedding Shifting")
        # Phase 1: Apply shifting
        counter = 0
        hits = 0
        reciprocal_rank_sum = 0

        # Generate the prototype
        self.prototype_generator()

        h_p = self.prototype[0]
        # print("prototype head:", h_p)
        r_p = self.prototype[1]
        t_p = self.prototype[2]

        print("***")

        for target in self.test_triple.keys():
            # print("target = ", target)
            h_target, r_target, t_target = target
            # print(h_target,r_target,t_target)
            # Initialize ranking dictionary
            rank_tail_dict = {}
            for t_id in self.test_entity_dict.keys():
                h_emb = np.array(self.test_entity_dict[h_target])
                r_emb = np.array(self.relation_dict[r_target])
                t_emb = np.array(self.test_entity_dict[t_id])
                # Calculate the distance for tail prediction
                rank_tail_dict[t_id] = distance(h_emb, r_emb, t_emb)

            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))
            # Check top ranking and apply shifting
            for i in range(len(rank_tail_sorted)):
                if t_target == rank_tail_sorted[i][0]:
                    if i == 0:
                        hits += 1
                    elif i > 0:
                        # Apply shifting to the current triple embeddings
                        self.explain_triple[target][0] += self.alpha * (h_p - self.explain_triple[target][0])
                        self.explain_triple[target][1] += self.alpha * (r_p - self.explain_triple[target][1])
                        self.explain_triple[target][2] += self.alpha * (t_p - self.explain_triple[target][2])
                        print("successfully shifting triple", target, "!")
                    reciprocal_rank_sum += 1 / (i + 1)
                    break
        # Output final evaluation results
        self.hits1 = hits / len(self.explain_triple)
        self.MRR = reciprocal_rank_sum / len(self.explain_triple)
        print(f"Test Hits@1: {self.hits1}, Test MRR: {self.MRR}")
        # print(f"Total execution time: {time.time() - start:.2f} seconds")
        # print(f"Skip number: {self.skip}, Skipped tail number: {self.skip_t}, Skipped Head Number: {self.skip_h}, Skipped relation Number: {self.skip_r}")

        print('There we have', counter, ' that are not hit@1 triple finished the embedding shifting.')
        print("Phase2: Explanation Evaluation")
        # Phase 2: Recalculate distances and evaluate
        hits = 0
        reciprocal_rank_sum = 0

        # Directly test on the shifting set which has already done embedding shifting
        for target in self.explain_triple.keys():
            h_target, r_target, t_target = target
            # Initialize ranking dictionary
            rank_tail_dict = {}
            for t_id in self.test_entity_dict.keys():
                if t_id != t_target:
                    # When it is not the target, t_emb is not shifted
                    h_emb = np.array(self.test_entity_dict[h_target])
                    r_emb = np.array(self.relation_dict[r_target])
                    t_emb = np.array(self.test_entity_dict[t_id])
                else:
                    # When it is the target, t_emb is shifted
                    if mode == 't':
                        h_emb = np.array(self.test_entity_dict[h_target])
                        r_emb = np.array(self.relation_dict[r_target])
                        t_emb = np.array(self.explain_triple[target][2])
                    elif mode == 'h':
                        h_emb = np.array(self.explain_triple[target][0])
                        r_emb = np.array(self.relation_dict[r_target])
                        t_emb = np.array(self.test_entity_dict[t_id])
                    elif mode == 'r':
                        h_emb = np.array(self.explain_triple[target][0])
                        r_emb = np.array(self.explain_triple[target][1])
                        t_emb = np.array(self.test_entity_dict[t_id])
                    elif mode == 'all':
                        h_emb = np.array(self.explain_triple[target][0])
                        r_emb = np.array(self.explain_triple[target][1])
                        t_emb = np.array(self.explain_triple[target][2])
                    else:
                        print("You forgot to set the mode!")
                # Calculate the distance for tail prediction
                # Check if the ranking has improved after shifting
                rank_tail_dict[t_id] = distance(h_emb, r_emb, t_emb)

            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))
            # Check top ranking and apply shifting
            for i in range(len(rank_tail_sorted)):
                if t_target == rank_tail_sorted[i][0]:
                    if i == 0:
                        hits += 1
                    reciprocal_rank_sum += 1 / (i + 1)
                    break

        # Output final evaluation results
        self._hits1 = hits / len(self.explain_triple)
        self._MRR = reciprocal_rank_sum / len(self.explain_triple)
        print(f"Final Hits@1: {self._hits1}, Final MRR: {self._MRR}")
        print(f"Total execution time: {time.time() - start:.2f} seconds")
        print(
            f"Skip number: {self.skip}, Skipped tail number: {self.skip_t}, Skipped Head Number: {self.skip_h}, Skipped relation Number: {self.skip_r}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ExplainWithPrototype Parameters")
    parser.add_argument('--data_name', type=str, default='wn18rr', required=False, help='Name of the dataset')
    parser.add_argument('--alpha', type=float, default=0.8, required=False, help='Alpha value for embedding shifting')
    parser.add_argument('--dim', type=int, default=50, required=False, help='The number of embedding dim.')
    parser.add_argument('--mode', type=str, default='t', required=False)

    args = parser.parse_args()
    data_name = args.data_name
    alpha = args.alpha
    dim = args.dim
    mode = args.mode

    print("dataset:", data_name, " alpha = ", alpha, " embedding dim = ", dim)
    check_file_path = f"../res/entity_{dim}dim_{data_name}_batch200"

    start_time = time.time()  # Start time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    _, _, train_triple = train_loader(f"../dataset/{data_name}/")  # Read train_triple for filtering training method

    entity_dict, relation_dict, _ = \
        test_loader(f"../res/entity_{dim}dim_{data_name}_batch200", f"../res/relation_{dim}dim_{data_name}_batch200",
                    f"../dataset/{data_name}/test.txt")

    json_file_path = 'rectified_test_wn18rr_set.json'

    # Read and process the JSON file
    # The test_head_emb, test_relation_emb, test_tail_emb here are the embeddings of all actual triples used
    # test_triple is the actual test dataset
    test_triple, test_entity_emb, test_head_emb, test_relation_emb, test_tail_emb = read_and_process_json(
        json_file_path)

    # Save to TXT file
    save_to_txt('test_head_emb.txt', test_head_emb)
    save_to_txt('test_relation_emb.txt', test_relation_emb)
    save_to_txt('test_tail_emb.txt', test_tail_emb)

    explain = ExplainWithPrototype(entity_dict, relation_dict, train_triple, test_triple, test_entity_emb,
                                   check_file_path, dim, alpha, mode, isFit=False)
    # explain.prototype_generator()
    # print("Embedding Shifting...")
    explain.embedding_shifting()
    # explain.prototype_generator()
    # explain.plot_prototypes()

    f = open(f"../outputs/get_{mode}-shifted-explain_on_{data_name}_{current_time}_a{alpha}.txt", 'w')
    f.write("Test hits@1: " + str(explain.hits1) + '\n')
    f.write("Test MRR: " + str(explain.MRR) + '\n')
    f.write("Final hits@1: " + str(explain._hits1) + '\n')
    f.write("Final MRR: " + str(explain._MRR) + '\n')
    f.close()

    end_time = time.time()  # End time
    total_time = end_time - start_time  # Calculate total execution time
    print(f"Total execution time: {total_time:.2f} seconds")  # Print total execution time
