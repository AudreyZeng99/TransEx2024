import numpy as np
import codecs
import operator
import json
from trainer import train_loader, entity2id, relation2id
# from transE_speed import data_loader, entity2id, relation2id
import time


def test_loader(entity_file, relation_file, test_file):
    # entity_file: entity \t embedding
    entity_dict = {}
    relation_dict = {}
    test_triple = []

    with codecs.open(entity_file) as e_f:
        lines = e_f.readlines()
        for line in lines:
            entity, embedding = line.strip().split('\t')  # Get entity and its vector
            embedding = np.array(json.loads(embedding))
            entity_dict[int(entity)] = embedding  # Map entity and vector to a dictionary

    with codecs.open(relation_file) as r_f:
        lines = r_f.readlines()
        for line in lines:
            relation, embedding = line.strip().split('\t')  # Get relation and its vector
            embedding = np.array(json.loads(embedding))
            relation_dict[int(relation)] = embedding  # Map relation and vector to a dictionary

    with codecs.open(test_file) as t_f:
        lines = t_f.readlines()
        for line in lines:
            triple = line.strip().split('\t')  # Get test set data
            if len(triple) != 3:  # Find head-relation-tail triples as the test set
                continue
            h_ = entity2id[triple[0]]
            r_ = relation2id[triple[1]]
            t_ = entity2id[triple[2]]

            test_triple.append(tuple((h_, r_, t_)))

    return entity_dict, relation_dict, test_triple  # Get entities, relations, test set


def distance(h, r, t):
    return np.linalg.norm(h + r - t)


def load_ids_from_file(filename):
    id_set = set()
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            _, entity_id = line.strip().split('\t')
            id_set.add(int(entity_id))  # Convert ID to integer and add to the set
    return id_set


def check_id_exists(entity_id, id_set):
    return entity_id in id_set


class Test:
    def __init__(self, entity_dict, relation_dict, test_triple, train_triple, isFit=True):
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.test_triple = test_triple
        self.train_triple = train_triple
        print(len(self.entity_dict), len(self.relation_dict), len(self.test_triple), len(self.train_triple))
        self.isFit = isFit

        self.hits1 = 0
        self.MRR = 0

    def rank(self):
        hits = 0
        reciprocal_rank_sum = 0
        step = 1
        start = time.time()
        for triple in self.test_triple:
            # Skip entities without embeddings
            test_h_id, _, test_t_id = triple
            if test_h_id not in self.entity_dict or test_t_id not in self.entity_dict:
                continue  # If head or tail entity is not in entity_dict, skip this triple
            if test_h_id not in self.entity_dict:
                print("Head entity ID not found:", test_h_id)
            if test_t_id not in self.entity_dict:
                print("Tail entity ID not found:", test_t_id)

            rank_head_dict = {}
            rank_tail_dict = {}

            for entity in self.entity_dict.keys():  # triple is the test triple, entity is the loaded entity, keys() are labels like 0,1,2,3,...
                if self.isFit:  # When testing, check if the new triple appears in the training set, if so, delete it (filtered setting)
                    if [entity, triple[1], triple[2]] not in self.train_triple:  # Test set's tail entity, relation infers head entity
                        h_emb = self.entity_dict[entity]
                        r_emb = self.relation_dict[triple[1]]
                        t_emb = self.entity_dict[triple[2]]
                        rank_head_dict[entity] = distance(h_emb, r_emb, t_emb)  # distance head + relation - tail
                else:  # When testing, do not check if the new triple appears in the training set (raw setting)
                    h_emb = self.entity_dict[entity]
                    r_emb = self.relation_dict[triple[1]]
                    t_emb = self.entity_dict[triple[2]]
                    rank_head_dict[entity] = distance(h_emb, r_emb, t_emb)

                if self.isFit:
                    if [triple[0], triple[2], entity] not in self.train_triple:  # Test set's head entity, relation infers tail entity
                        h_emb = self.entity_dict[triple[0]]
                        r_emb = self.relation_dict[triple[1]]
                        t_emb = self.entity_dict[entity]
                        rank_tail_dict[entity] = distance(h_emb, r_emb, t_emb)
                else:
                    h_emb = self.entity_dict[triple[0]]
                    r_emb = self.relation_dict[triple[1]]
                    t_emb = self.entity_dict[entity]
                    rank_tail_dict[entity] = distance(h_emb, r_emb, t_emb)
            # sorted(iterable, key=None, reverse=False), key -- the element to compare
            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1))
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))

            # Calculate tail entity rank
            for i in range(len(rank_tail_sorted)):
                if triple[2] == rank_tail_sorted[i][0]:
                    if i == 0:
                        hits += 1
                    reciprocal_rank_sum += 1 / (i + 1)
                    break

            step += 1
            if step % 200 == 0:
                end = time.time()
                print("step: ", step, " ,hit_top1_rate: ", hits / (2 * step), " ,MRR ", reciprocal_rank_sum / (2 * step),
                      'time of testing one triple: %s' % (round((end - start), 3)))
                start = end
        self.hits1 = hits / (2 * len(self.test_triple))
        self.MRR = reciprocal_rank_sum / (2 * len(self.test_triple))

    def relation_rank(self):  # Most papers do not introduce relation ranking; this is written in this code. Entity ranking is below; replacing entity might reduce the score.
        hits = 0
        rank_sum = 0
        step = 1

        start = time.time()
        for triple in self.test_triple:
            rank_dict = {}
            for r in self.relation_dict.keys():
                if self.isFit and (triple[0], triple[1], r) in self.train_triple:
                    continue
                h_emb = self.entity_dict[triple[0]]
                r_emb = self.relation_dict[r]
                t_emb = self.entity_dict[triple[1]]
                rank_dict[r] = distance(h_emb, r_emb, t_emb)

            rank_sorted = sorted(rank_dict.items(), key=operator.itemgetter(1))  # The itemgetter function in the operator module is mainly used to get data of a specific dimension of an object
            # Rank based on rank_dict dictionary score
            rank = 1
            for i in rank_sorted:
                if triple[2] == i[0]:  # If the prediction is correct, break the loop
                    break
                rank += 1
            if rank < 10:
                hits += 1
            rank_sum = rank_sum + rank + 1  # Not sure why add 1

            step += 1  # Increment step for each test triple
            if step % 200 == 0:
                end = time.time()
                print("step: ", step, " ,hit_top10_rate: ", hits / step, " ,MRR ", rank_sum / step,
                      'used time: %s' % (round((end - start), 3)))
                start = end

        self.relation_hits10 = hits / len(self.test_triple)
        self.relation_mean_rank = rank_sum / len(self.test_triple)


if __name__ == '__main__':
    data_name = 'yago3-10'
    _, _, train_triple = train_loader(f"../dataset/{data_name}/")  # Read train_triple for filtering training method

    entity_dict, relation_dict, test_triple = \
        test_loader(f"../res/entity_50dim_{data_name}_batch200", f"../res/relation_50dim_{data_name}_batch200",
                   f"../dataset/{data_name}/test.txt")
    # dataloader("..\\res\\entity_temp_260epoch","..\\res\\relation_temp_260epoch",
    #            "..\\FB15k\\test.txt")

    test = Test(entity_dict, relation_dict, test_triple, train_triple, isFit=False)

    # test.relation_rank()
    # print("relation hits@10: ", test.relation_hits10)
    # print("relation meanrank: ", test.relation_mean_rank)

    # print("Replacing head and tail triples takes more time...")
    test.rank()
    # print("entity hits@10: ", test.hits10)
    # print("entity meanrank: ", test.mean_rank)
    print("Entity Hits@1: ", test.hits1)
    print("Entity MRR: ", test.MRR)

    f = open(f"../res/result_on_{data_name}.txt", 'w')
    f.write("entity hits@10: " + str(test.hits1) + '\n')
    f.write("entity hits@10: " + str(test.MRR) + '\n')
    # f.write("entity meanrank: " + str(test.mean_rank) + '\n')
    # f.write("relation hits@10: " + str(test.relation_hits10) + '\n')
    # f.write("relation meanrank: " + str(test.relation_mean_rank) + '\n')
    f.close()
