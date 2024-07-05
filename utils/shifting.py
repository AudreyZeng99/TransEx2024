import numpy as np
from scipy.spatial.distance import euclidean
import json

class Shifting:
    def __init__(self, dataset_path, result_file_path, prototypes_file_path, outputs_file_path, alpha=0.1):
        self.dataset_path = dataset_path
        self.result_file_path = result_file_path
        self.prototypes_file_path = prototypes_file_path
        self.outputs_file_path = outputs_file_path
        self.alpha = alpha
        self.dataset = 'fb15k-237'
        self.h_prototype, self.t_prototype, self.r_prototype = self.read_prototypes()
        self.entity2id = self.load_entity2id()
        self.relation2id = self.load_relation2id()
        self.entity_embeddings = self.load_embeddings('entity_50dim_batch200')
        self.relation_embeddings = self.load_embeddings('relation_50dim_batch200')
        self.test_data = self.load_test_data()

    def read_prototypes(self):
        with open(self.prototypes_file_path, 'r') as f:
            lines = f.readlines()
            h_prototype = np.array([float(x) for x in lines[0].split(': ')[1].strip().split()])
            t_prototype = np.array([float(x) for x in lines[1].split(': ')[1].strip().split()])
            r_prototype = np.array([float(x) for x in lines[2].split(': ')[1].strip().split()])
        return h_prototype, t_prototype, r_prototype

    def load_entity2id(self):
        entity2id = {}
        with open(self.dataset_path + self.dataset + '/entity2id.txt', 'r') as f:
            for line in f:
                entity, eid = line.strip().split()
                entity2id[entity] = int(eid)
        return entity2id

    def load_relation2id(self):
        relation2id = {}
        with open(self.dataset_path + self.dataset + '/relation2id.txt', 'r') as f:
            for line in f:
                relation, rid = line.strip().split()
                relation2id[relation] = int(rid)
        return relation2id

    def load_embeddings(self, file_name):
        embeddings = {}
        with open(self.result_file_path + file_name, 'r') as f:
            for line in f:
                parts = line.strip().replace('[', '').replace(']', '').replace(',', '').split()
                eid = int(parts[0])
                embedding = np.array([float(x) for x in parts[1:]])
                embeddings[eid] = embedding
        return embeddings

    def load_test_data(self):
        test_data = []
        with open(self.dataset_path + self.dataset + '/test_swapped.txt', 'r') as f:
            for line in f:
                h, t, r = line.strip().split()
                test_data.append((h, t, r))
        return test_data

    def find_missing_embeddings(self):
        missing_entities = set()
        missing_relations = set()
        for h, t, r in self.test_data:
            if h not in self.entity2id:
                missing_entities.add(h)
            if t not in self.entity2id:
                missing_entities.add(t)
            if r not in self.relation2id:
                missing_relations.add(r)
        return missing_entities, missing_relations

    def calculate_distances(self):
        distances = []
        for h, t, r in self.test_data:
            if h in self.entity2id and t in self.entity2id and r in self.relation2id:
                h_id = self.entity2id[h]
                t_id = self.entity2id[t]
                r_id = self.relation2id[r]
                if h_id in self.entity_embeddings and t_id in self.entity_embeddings and r_id in self.relation_embeddings:
                    h_emb = self.entity_embeddings[h_id]
                    t_emb = self.entity_embeddings[t_id]
                    r_emb = self.relation_embeddings[r_id]
                    d_h = euclidean(h_emb, self.h_prototype)
                    d_t = euclidean(t_emb, self.t_prototype)
                    d_r = euclidean(r_emb, self.r_prototype)
                    total_distance = d_h + d_t + d_r
                    distances.append((total_distance, h_id, t_id, r_id, h_emb, t_emb, r_emb))
        return distances

    def shift_embeddings(self, distances):
        distances.sort(key=lambda x: x[0])
        selected_triples = distances[:100]
        shifted_data = []
        for total_distance, h_id, t_id, r_id, h_emb, t_emb, r_emb in selected_triples:
            h_emb_sf = h_emb + self.alpha * (self.h_prototype - h_emb)
            t_emb_sf = t_emb + self.alpha * (self.t_prototype - t_emb)
            r_emb_sf = r_emb + self.alpha * (self.r_prototype - r_emb)
            shifted_data.append({
                'triple_id': (h_id, t_id, r_id),
                'h_emb_sf': h_emb_sf.tolist(),
                't_emb_sf': t_emb_sf.tolist(),
                'r_emb_sf': r_emb_sf.tolist()
            })
        return shifted_data

    def save_results(self, shifted_data):
        output_file = self.outputs_file_path + f"shifted_selected_embedding_a{self.alpha}.json"
        with open(output_file, 'w') as f:
            json.dump(shifted_data, f, indent=2)
        return output_file

    def run(self):
        missing_entities, missing_relations = self.find_missing_embeddings()
        if missing_entities:
            print(f"Missing entity embeddings for: {missing_entities}")
        if missing_relations:
            print(f"Missing relation embeddings for: {missing_relations}")
        distances = self.calculate_distances()
        shifted_data = self.shift_embeddings(distances)
        output_file = self.save_results(shifted_data)
        print(f"Results saved to {output_file}")
        return output_file


# example：
shifting = Shifting(
    dataset_path='/content/drive/MyDrive/KGEE/Tutorial/TransE/dataset/',
    result_file_path='/content/drive/MyDrive/KGEE/Tutorial/TransE/res/',
    prototypes_file_path='/content/drive/MyDrive/KGEE/Tutorial/TransE/prototype/output_prototypes.txt',
    outputs_file_path='/content/drive/MyDrive/KGEE/Tutorial/TransE/outputs/',
    alpha=0.1
)
output_file_path = shifting.run()
print(f"Output file path: {output_file_path}")