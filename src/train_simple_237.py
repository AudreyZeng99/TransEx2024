import codecs
import random
import math
import numpy as np
import copy
import time

entity2id = {}
relation2id = {}


def train_loader(file):
    file1 = file + "train.txt"
    file2 = file + "entity2id.txt"
    file3 = file + "relation2id.txt"

    with codecs.open(file2, 'r') as f1, codecs.open(file3, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity2id[line[0]] = int(line[1])

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation2id[line[0]] = int(line[1])

    entity_set = set()
    relation_set = set()
    triple_list = []

    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = entity2id[triple[0]]
            r_ = relation2id[triple[1]]
            t_ = entity2id[triple[2]]

            triple_list.append([h_, r_, t_])

            entity_set.add(h_)
            entity_set.add(t_)
            relation_set.add(r_)

    return entity_set, relation_set, triple_list


def distanceL2(h, r, t):
    return np.sum(np.square(h + r - t))


def distanceL1(h, r, t):
    return np.sum(np.fabs(h + r - t))


class TransE:
    def __init__(self, entity_set, relation_set, triple_list,
                 embedding_dim=100, learning_rate=0.01, margin=1, L1=True):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.triple_list = triple_list
        self.L1 = L1

        self.loss = 0

    def emb_initialize(self):
        relation_dict = {}
        entity_dict = {}

        for relation in self.relation:
            r_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp, ord=2)

        for entity in self.entity:
            e_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)

        self.relation = relation_dict
        self.entity = entity_dict

    def train(self, epochs):
        nbatches = 200
        batch_size = len(self.triple_list) // nbatches
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0
            num = 0

            for k in range(nbatches):
                Sbatch = random.sample(self.triple_list, batch_size) # obtain a batch of triple samples
                Tbatch = []# corrupted sample
                num = num + 1

                # for each triple, select 3 corrupted samples
                for triple in Sbatch:
                    corrupted_triple = self.Corrupt(triple)
                    Tbatch.append((triple, corrupted_triple))

                self.update_embeddings(Tbatch)
                self.loss = self.loss / num

            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("loss: ", self.loss)

            if epoch % 10 == 0:
                with codecs.open("../res/entity_temp", "w") as f_e:
                    for e in self.entity.keys():
                        f_e.write(str(e) + "\t")
                        f_e.write(str(list(self.entity[e])))
                        f_e.write("\n")
                with codecs.open("../res/relation_temp", "w") as f_r:
                    for r in self.relation.keys():
                        f_r.write(str(r) + "\t")
                        f_r.write(str(list(self.relation[r])))
                        f_r.write("\n")
                with codecs.open("../res/result_temp", "a") as f_s:
                    f_s.write("epoch: %d\tloss: %s\n" % (epoch, self.loss))

        print("写入文件...")
        with codecs.open(f"../res/entity_50dim_batch{nbatches}", "w") as f1:
            for e in self.entity.keys():
                f1.write(str(e) + "\t")
                f1.write(str(list(self.entity[e])))
                f1.write("\n")

        with codecs.open(f"../res/relation_50dim_batch{nbatches}", "w") as f2:
            for r in self.relation.keys():
                f2.write(str(r) + "\t")
                f2.write(str(list(self.relation[r])))
                f2.write("\n")
        print("写入完成")

    # def Corrupt(self, triple):
    #     corrupted_triple = copy.deepcopy(triple)
    #     seed = random.random()
    #     if seed > 0.5:
    #         head = triple[0]
    #         rand_head = head
    #         while rand_head == head:
    #             rand_head = random.randint(0, len(self.entity) - 1)
    #         corrupted_triple[0] = rand_head
    #     else:
    #         tail = triple[1]
    #         rand_tail = tail
    #         while rand_tail == tail:
    #             rand_tail = random.randint(0, len(self.entity) - 1)
    #         corrupted_triple[1] = rand_tail
    #     return corrupted_triple
    def Corrupt(self, triple):
        corrupted_triple = copy.deepcopy(triple)
        seed = random.random()
        entities = list(self.entity)
        if seed > 0.5:
            # randomly replace the head entities
            head = triple[0]
            rand_head = head
            while rand_head == head:
                rand_head = random.choice(entities)
            corrupted_triple[0] = rand_head
        else:
            # randomly replace the tail entities
            tail = triple[1]
            rand_tail = tail
            while rand_tail == tail:
                rand_tail = random.choice(entities)
            corrupted_triple[1] = rand_tail
        return corrupted_triple


    def update_embeddings(self, Tbatch):
        entity_updated = {}
        relation_updated = {}
        for triple, corrupted_triple in Tbatch:
            try:
                h_correct = self.entity[triple[0]]
                t_correct = self.entity[triple[1]]
                relation = self.relation[triple[2]]

                h_corrupt = self.entity[corrupted_triple[0]]
                t_corrupt = self.entity[corrupted_triple[1]]
            except KeyError as e:
                print(f"KeyError: {e} not found in entity or relation dictionary.")
                print(f"Triple: {triple}")
                print(f"Corrupted Triple: {corrupted_triple}")
                continue

            if triple[0] not in entity_updated:
                entity_updated[triple[0]] = copy.copy(self.entity[triple[0]])
            if triple[1] not in entity_updated:
                entity_updated[triple[1]] = copy.copy(self.entity[triple[1]])
            if triple[2] not in relation_updated:
                relation_updated[triple[2]] = copy.copy(self.relation[triple[2]])
            if corrupted_triple[0] not in entity_updated:
                entity_updated[corrupted_triple[0]] = copy.copy(self.entity[corrupted_triple[0]])
            if corrupted_triple[1] not in entity_updated:
                entity_updated[corrupted_triple[1]] = copy.copy(self.entity[corrupted_triple[1]])

            if self.L1:
                dist_correct = distanceL1(h_correct, relation, t_correct)
                dist_corrupt = distanceL1(h_corrupt, relation, t_corrupt)
            else:
                dist_correct = distanceL2(h_correct, relation, t_correct)
                dist_corrupt = distanceL2(h_corrupt, relation, t_corrupt)

            err = self.hinge_loss(dist_correct, dist_corrupt)

            if err > 0:
                self.loss += err
                grad_pos = 2 * (h_correct + relation - t_correct)
                grad_neg = 2 * (h_corrupt + relation - t_corrupt)

                if self.L1:
                    grad_pos = np.sign(grad_pos)
                    grad_neg = np.sign(grad_neg)

                entity_updated[triple[0]] -= self.learning_rate * grad_pos
                entity_updated[triple[1]] -= (-1) * self.learning_rate * grad_pos

                entity_updated[corrupted_triple[0]] -= (-1) * self.learning_rate * grad_neg
                entity_updated[corrupted_triple[1]] -= self.learning_rate * grad_neg

                relation_updated[triple[2]] -= self.learning_rate * grad_pos
                relation_updated[triple[2]] -= (-1) * self.learning_rate * grad_neg

        for i in entity_updated.keys():
            entity_updated[i] /= np.linalg.norm(entity_updated[i])
            self.entity[i] = entity_updated[i]
        for i in relation_updated.keys():
            relation_updated[i] /= np.linalg.norm(relation_updated[i])
            self.relation[i] = relation_updated[i]

    def hinge_loss(self, dist_correct, dist_corrupt):
        return max(0, dist_correct - dist_corrupt + self.margin)


if __name__ == '__main__':
    file1 = "../dataset/fb15k-237/"
    entity_set, relation_set, triple_list = train_loader(file1)
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))
    transE = TransE(entity_set, relation_set, triple_list, embedding_dim=50, learning_rate=0.02, margin=1, L1=True)
    transE.emb_initialize()
    transE.train(epochs=100)
