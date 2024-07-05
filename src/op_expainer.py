import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from trainer import train_loader
from tester import test_loader, distance, load_ids_from_file, check_id_exists
import time
from datetime import datetime
import argparse
import json
import codecs
import operator


class PrototypeModel(nn.Module):
    def __init__(self, dim):
        super(PrototypeModel, self).__init__()
        self.h_p = nn.Parameter(torch.randn(dim))
        self.r_p = nn.Parameter(torch.randn(dim))
        self.t_p = nn.Parameter(torch.randn(dim))

    def forward(self):
        return self.h_p, self.r_p, self.t_p


def compute_loss(h_p, r_p, t_p, lambda1=1.0, lambda2=1.0, margin=0.2):
    # Margin loss: head + relation - tail
    margin_loss = torch.norm(h_p + r_p - t_p, p=2)

    # Cosine similarity for orthogonality
    cos_sim_hr = torch.nn.functional.cosine_similarity(h_p, r_p, dim=0)
    cos_sim_ht = torch.nn.functional.cosine_similarity(h_p, t_p, dim=0)
    cos_sim_rt = torch.nn.functional.cosine_similarity(r_p, t_p, dim=0)

    # Orthogonality loss
    orthogonality_loss = -(cos_sim_hr + cos_sim_ht + cos_sim_rt)

    # Total loss
    total_loss = lambda1 * margin_loss + lambda2 * orthogonality_loss
    return total_loss


def train_prototype_model(train_triple, entity_dict, relation_dict, dim, p_nepochs=200, lr=0.001, lambda1=1.0, lambda2=1.0):
    model = PrototypeModel(dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(p_nepochs):
        h_p, r_p, t_p = model()

        loss = compute_loss(h_p, r_p, t_p, lambda1, lambda2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return h_p.detach().numpy(), r_p.detach().numpy(), t_p.detach().numpy()

def read_and_process_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    test_head_emb = {}
    test_relation_emb = {}
    test_tail_emb = {}
    test_triple = {}
    test_entity_emb = {}
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
                 mode, isFit=True):
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
        self.epsilon_h = []
        self.epsilon_r = []
        self.epsilon_t = []
        self.mode = mode

    def prototype_generator(self, h_p, r_p, t_p):
        self.prototype = (h_p, r_p, t_p)
        print("Successfully get prototype!")

    def plot_prototypes(self):
        h_p, r_p, t_p = self.prototype
        vectors = np.array([h_p, r_p, t_p])
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(vectors)
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
        counter = 0
        hits = 0
        reciprocal_rank_sum = 0
        dop = 0

        h_p, r_p, t_p = self.prototype
        for target in self.test_triple.keys():
            h_target, r_target, t_target = target
            rank_tail_dict = {}
            for t_id in self.test_entity_dict.keys():
                h_emb = np.array(self.test_entity_dict[h_target])
                r_emb = np.array(self.relation_dict[r_target])
                t_emb = np.array(self.test_entity_dict[t_id])
                rank_tail_dict[t_id] = distance(h_emb, r_emb, t_emb)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))
            for i in range(len(rank_tail_sorted)):
                if t_target == rank_tail_sorted[i][0]:
                    if i == 0:
                        hits += 1
                    elif i > 0:
                        # self.explain_triple[target][0] += self.alpha * (h_p - self.explain_triple[target][0])
                        # self.explain_triple[target][1] += self.alpha * (r_p - self.explain_triple[target][1])
                        # self.explain_triple[target][2] += self.alpha * (t_p - self.explain_triple[target][2])

                        self.epsilon_h = h_p - self.explain_triple[target][0]
                        self.epsilon_r = r_p - self.explain_triple[target][1]
                        self.epsilon_t = t_p - self.explain_triple[target][2]
                        print("successfully get epsilon from ", target, "!")
                    reciprocal_rank_sum += 1 / (i + 1)
                    break
        self.hits1 = hits / len(self.explain_triple)
        self.MRR = reciprocal_rank_sum / len(self.explain_triple)
        print(f"Test Hits@1: {self.hits1}, Test MRR: {self.MRR}")
        print('There we have', counter, ' that are not hit@1 triple finished the embedding shifting.')
        print("Phase2: Explanation Evaluation")
        hits = 0
        reciprocal_rank_sum = 0
        for target in self.explain_triple.keys():
            h_target, r_target, t_target = target
            rank_tail_dict = {}
            for t_id in self.test_entity_dict.keys():
                if t_id != t_target:
                    h_emb = np.array(self.test_entity_dict[h_target])
                    r_emb = np.array(self.relation_dict[r_target])
                    t_emb = np.array(self.test_entity_dict[t_id])
                # else:
                #     # h_emb = np.array(self.test_entity_dict[h_target])
                #     # r_emb = np.array(self.relation_dict[r_target])
                #     h_emb = np.array(self.explain_triple[target][0])
                #     r_emb = np.array(self.explain_triple[target][1])
                #     t_emb = np.array(self.explain_triple[target][2])
                else:
                    if self.mode == 't':
                        h_emb = np.array(self.test_entity_dict[h_target])
                        r_emb = np.array(self.relation_dict[r_target])
                        t_emb = np.array(self.prototype[2])
                    elif self.mode == 'r':
                        h_emb = np.array(self.prototype[0])
                        r_emb = np.array(self.prototype[1])
                        t_emb = np.array(self.test_entity_dict[t_id])
                    elif self.mode == 'all':
                        h_emb = np.array(self.prototype[0])
                        r_emb = np.array(self.prototype[1])
                        t_emb = np.array(self.prototype[2])
                    else:
                        print("You didn't set a correct mode")
                rank_tail_dict[t_id] = distance(h_emb, r_emb, t_emb)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))
            for i in range(len(rank_tail_sorted)):
                if t_target == rank_tail_sorted[i][0]:
                    if i == 0:
                        hits += 1
                    reciprocal_rank_sum += 1 / (i + 1)
                    break
        self._hits1 = hits / len(self.explain_triple)
        self._MRR = reciprocal_rank_sum / len(self.explain_triple)
        print(f"Final Hits@1: {self._hits1}, Final MRR: {self._MRR}")
        print(f"Total execution time: {time.time() - start:.2f} seconds")
        print(
            f"Skip number: {self.skip}, Skipped tail number: {self.skip_t}, Skipped Head Number: {self.skip_h}, Skipped relation Number: {self.skip_r}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ExplainWithPrototype Parameters")
    parser.add_argument('--data_name', type=str, default='wn18rr', required=False, help='Name of the dataset')
    parser.add_argument('--alpha', type=float, default=0.5, required=False, help='Alpha value for embedding shifting')
    parser.add_argument('--dim', type=int, default=50, required=False, help='The number of embedding dim.')
    parser.add_argument('--p_nepoch', type=int, default=100, required=False)
    parser.add_argument('--mode', type=str, default='t', required=True)


    args = parser.parse_args()
    data_name = args.data_name
    alpha = args.alpha
    dim = args.dim
    p_nepoch = args.p_nepoch
    mode = args.mode


    # print("dataset:", data_name, " alpha = ", alpha, " embedding dim = ", dim)
    print("dataset:", data_name, " mode = ", mode, " embedding dim = ", dim)
    check_file_path = f"../res/entity_{dim}dim_{data_name}_batch200"

    start_time = time.time()  # Start time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    _, _, train_triple = train_loader(f"../dataset/{data_name}/")  # Read train_triple for filtering training method

    entity_dict, relation_dict, _ = \
        test_loader(f"../res/entity_{dim}dim_{data_name}_batch200", f"../res/relation_{dim}dim_{data_name}_batch200",
                    f"../dataset/{data_name}/test.txt")

    json_file_path = 'rectified_test_wn18rr_set.json'
    test_triple, test_entity_emb, test_head_emb, test_relation_emb, test_tail_emb = read_and_process_json(
        json_file_path)
    save_to_txt('test_head_emb.txt', test_head_emb)
    save_to_txt('test_relation_emb.txt', test_relation_emb)
    save_to_txt('test_tail_emb.txt', test_tail_emb)

    h_p, r_p, t_p = train_prototype_model(train_triple, entity_dict, relation_dict, dim, p_nepochs=p_nepoch, lr=0.01)
    # explain = ExplainWithPrototype(entity_dict, relation_dict, train_triple, test_triple, test_entity_emb,
    #                                check_file_path, dim, alpha, isFit=False)

    explain = ExplainWithPrototype(entity_dict, relation_dict, train_triple, test_triple, test_entity_emb,
                                   check_file_path, dim, mode, isFit=False)
    explain.prototype_generator(h_p, r_p, t_p)
    explain.embedding_shifting()

    # f = open(f"../res/get_op_explain_on_{data_name}_{current_time}_a{alpha}.txt", 'w')
    f = open(f"../outputs/get_op_explain_on_{data_name}_{current_time}_mode{mode}.txt", 'w')
    f.write("Test hits@1: " + str(explain.hits1) + '\n')
    f.write("Test MRR: " + str(explain.MRR) + '\n')
    f.write("Final hits@1: " + str(explain._hits1) + '\n')
    f.write("Final MRR: " + str(explain._MRR) + '\n')
    f.close()

    end_time = time.time()  # End time
    total_time = end_time - start_time  # Calculate total execution time
    print(f"Total execution time: {total_time:.2f} seconds")  # Print total execution time
