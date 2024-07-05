import numpy as np
import pandas as pd

# 声明文件路径变量
# entity2id_path = '/content/drive/MyDrive/KGEE/Tutorial/TransE/dataset/fb15k-237/entity2id_.txt'
# relation2id_path = '/content/drive/MyDrive/KGEE/Tutorial/TransE/dataset/fb15k-237/relation2id_.txt'
# train_swapped_path = '/content/drive/MyDrive/KGEE/Tutorial/TransE/dataset/fb15k-237/train_swapped.txt'
# entity_embeddings_path = '/content/drive/MyDrive/KGEE/Tutorial/TransE/res/entity_50dim_batch200'
# relation_embeddings_path = '/content/drive/MyDrive/KGEE/Tutorial/TransE/res/relation_50dim_batch200'
# output_path = '/content/drive/MyDrive/KGEE/Tutorial/TransE/prototype/output_prototypes.txt'
entity2id_path = '/content/drive/MyDrive/KGEE/Tutorial/TransE/dataset/fb15k-237/entity2id_.txt'
relation2id_path = '/content/drive/MyDrive/KGEE/Tutorial/TransE/dataset/fb15k-237/relation2id_.txt'
train_swapped_path = '/content/drive/MyDrive/KGEE/Tutorial/TransE/dataset/fb15k-237/train_swapped.txt'
entity_embeddings_path = '/content/drive/MyDrive/KGEE/Tutorial/TransE/res/entity_50dim_batch200'
relation_embeddings_path = '/content/drive/MyDrive/KGEE/Tutorial/TransE/res/relation_50dim_batch200'
output_path = '/content/drive/MyDrive/KGEE/Tutorial/TransE/prototype/output_prototypes.txt'

# 读取文件并解析数据
entity2id = pd.read_csv(entity2id_path, sep='\t', header=None, names=['entity', 'id'])
relation2id = pd.read_csv(relation2id_path, sep='\t', header=None, names=['relation', 'id'])
train_swapped = pd.read_csv(train_swapped_path, sep='\t', header=None, names=['head', 'tail', 'relation'])

entity_embeddings = pd.read_csv(entity_embeddings_path, sep='\t', header=None, names=['id', 'embedding'])
relation_embeddings = pd.read_csv(relation_embeddings_path, sep='\t', header=None, names=['id', 'embedding'])

# 转换为字典形式
entity2id_dict = dict(zip(entity2id['entity'], entity2id['id']))
relation2id_dict = dict(zip(relation2id['relation'], relation2id['id']))

# 使用json.loads将嵌入字符串解析为数组
entity_embeddings_dict = {row['id']: np.array(json.loads(row['embedding'])) for _, row in entity_embeddings.iterrows()}
relation_embeddings_dict = {row['id']: np.array(json.loads(row['embedding'])) for _, row in relation_embeddings.iterrows()}

# 计算头实体、尾实体和关系的出现次数
head_counts = train_swapped['head'].value_counts().to_dict()
tail_counts = train_swapped['tail'].value_counts().to_dict()
relation_counts = train_swapped['relation'].value_counts().to_dict()

# 加权求和计算prototype向量
def weighted_average(embeddings_dict, counts_dict, id_dict):
    total_weight = 0
    weighted_sum = np.zeros(50)  # 假设embedding维度为50
    for entity, count in counts_dict.items():
        entity_id = id_dict.get(entity)
        if entity_id is not None:
            embedding = embeddings_dict.get(entity_id)
            if embedding is not None:
                weighted_sum += count * embedding
                total_weight += count
    return weighted_sum / total_weight if total_weight > 0 else np.zeros(50)

h_prototype = weighted_average(entity_embeddings_dict, head_counts, entity2id_dict)
t_prototype = weighted_average(entity_embeddings_dict, tail_counts, entity2id_dict)

def weighted_average_relation(embeddings_dict, counts_dict, id_dict):
    total_weight = 0
    weighted_sum = np.zeros(50)  # 假设embedding维度为50
    for relation, count in counts_dict.items():
        relation_id = id_dict.get(relation)
        if relation_id is not None:
            embedding = embeddings_dict.get(relation_id)
            if embedding is not None:
                weighted_sum += count * embedding
                total_weight += count
    return weighted_sum / total_weight if total_weight > 0 else np.zeros(50)

r_prototype = weighted_average_relation(relation_embeddings_dict, relation_counts, relation2id_dict)

# 保存结果到文件
with open(output_path, 'w') as f:
    f.write('h_prototype: ' + ' '.join(map(str, h_prototype)) + '\n')
    f.write('t_prototype: ' + ' '.join(map(str, t_prototype)) + '\n')
    f.write('r_prototype: ' + ' '.join(map(str, r_prototype)) + '\n')

print("Prototype vectors have been saved to:", output_path)
