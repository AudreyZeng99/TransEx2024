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


def read_and_process_json(file_path):
    # 读取 JSON 文件
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    test_head_emb = {}
    test_relation_emb = {}
    test_tail_emb = {}
    test_triple = {}
    test_entity_emb = {}

    # 处理每个单元
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
    def __init__(self, entity_dict, relation_dict, train_triple, test_triple, test_entity_emb, check_file_path, dim, alpha, isFit=True):
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


    def prototype_generator(self):
        print("*************************************")
        print("Prototype generating...")
        counter = 0
        h_p = np.zeros(self.dim)  # 初始化全零向量，维度为dim
        r_p = np.zeros(self.dim)  # 初始化全零向量，维度为dim
        t_p = np.zeros(self.dim)  # 初始化全零向量，维度为dim

        for train_h_id, train_r_id, train_t_id in self.train_triple:
            h_emb = self.entity_dict[train_h_id]
            r_emb = self.relation_dict[train_r_id]
            t_emb = self.entity_dict[train_t_id]

            h_p += h_emb
            r_p += r_emb
            t_p += t_emb

            counter +=1
            # print("counter = ",counter)
        h_p /= counter
        r_p /= counter
        t_p /= counter

        self.prototype = (h_p, r_p, t_p)
        print("Successfully get prototype!")

    def embedding_shifting(self):
        step = 1
        start = time.time()
        print("Phrase1: Embedding Shifting")
        # 第一阶段：应用偏移
        counter = 0
        hits = 0
        reciprocal_rank_sum = 0

        # generate the prototype
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
            # 初始化排名字典
            rank_tail_dict = {}
            for t_id in self.test_entity_dict.keys():
                h_emb = np.array(self.test_entity_dict[h_target])
                r_emb = np.array(self.relation_dict[r_target])
                t_emb = np.array(self.test_entity_dict[t_id])
                # 计算对应tail prediction的距离
                rank_tail_dict[t_id] = distance(h_emb, r_emb, t_emb)

            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))
            # 检查首位排名并应用偏移
            for i in range(len(rank_tail_sorted)):
                if t_target == rank_tail_sorted[i][0]:
                    if i == 0:
                        hits +=1
                    elif i > 0:
                        # 当前的triple对应embedding需要进行shifting
                        # 暂存当前的
                        # h_ = self.check_entity_emb_dict[h_target]
                        # r_ = self.relation_dict[r_target]
                        # t_ = self.check_entity_emb_dict[t_target]   # 此时t_target == t_id

                        # 修改testing_set的备份shifting_set中当前triple对应的embedding
                        self.explain_triple[target][0] += self.alpha * (h_p - self.explain_triple[target][0])
                        self.explain_triple[target][1] += self.alpha * (r_p - self.explain_triple[target][1])
                        self.explain_triple[target][2] += self.alpha * (t_p - self.explain_triple[target][2])
                        print("successfully shifting triple", target,"!")
                    reciprocal_rank_sum += 1 / (i + 1)
                    break
        # 输出最终评估结果
        self.hits1 = hits / len(self.explain_triple)
        self.MRR = reciprocal_rank_sum / len(self.explain_triple)
        print(f"Test Hits@1: {self.hits1}, Test MRR: {self.MRR}")
        # print(f"Total execution time: {time.time() - start:.2f} seconds")
        # print(f"Skip number: {self.skip}, Skipped tail number: {self.skip_t}, Skipped Head Number: {self.skip_h}, Skipped relation Number: {self.skip_r}")


        print('There we have', counter, ' that are not hit@1 triple finished the embedding shifting.')
        print("Phrase2: Explanation Evaluation")
        # 第二阶段：重新计算距离和评估
        hits = 0
        reciprocal_rank_sum = 0

        # 直接读取已经做完embedding shifting的shifting_set进行测试即可
        for target in self.explain_triple.keys():
            h_target, r_target, t_target = target
            # 初始化排名字典
            rank_tail_dict = {}
            for t_id in self.test_entity_dict.keys():
                if t_id != t_target:
                    # 当不是target的时候，t_emb不取shifting
                    h_emb = np.array(self.test_entity_dict[h_target])
                    r_emb = np.array(self.relation_dict[r_target])
                    t_emb = np.array(self.test_entity_dict[t_id])
                else:
                    # 当是target的时候，t_emb取shifting
                    h_emb = np.array(self.explain_triple[target][0])
                    r_emb = np.array(self.explain_triple[target][1])
                    t_emb = np.array(self.explain_triple[target][2])
                # 计算对应tail prediction的距离
                # 其实就是看shifting之后的排名是否能有进步
                rank_tail_dict[t_id] = distance(h_emb, r_emb, t_emb)

            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))
            # 检查首位排名并应用偏移
            for i in range(len(rank_tail_sorted)):
                if t_target == rank_tail_sorted[i][0]:
                    if i == 0:
                        hits +=1
                    reciprocal_rank_sum += 1 / (i + 1)
                    break

        # 输出最终评估结果
        self._hits1 = hits / len(self.explain_triple)
        self._MRR = reciprocal_rank_sum / len(self.explain_triple)
        print(f"Final Hits@1: {self._hits1}, Final MRR: {self._MRR}")
        print(f"Total execution time: {time.time() - start:.2f} seconds")
        print(f"Skip number: {self.skip}, Skipped tail number: {self.skip_t}, Skipped Head Number: {self.skip_h}, Skipped relation Number: {self.skip_r}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ExplainWithPrototype Parameters")
    parser.add_argument('--data_name', type=str, default='fb15k-237', required=False, help='Name of the dataset')
    parser.add_argument('--alpha', type=float, default=0.2, required=False, help='Alpha value for embedding shifting')
    parser.add_argument('--dim', type=int, default=50, required=False, help='The number of embedding dim.')

    args = parser.parse_args()
    data_name = args.data_name
    alpha = args.alpha
    dim = args.dim

    print("dataset:", data_name, " alpha = ", alpha," embedding dim = ", dim)
    check_file_path = f"../res/entity_{dim}dim_{data_name}_batch200"

    start_time = time.time()  # 开始时间
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    _, _, train_triple = train_loader(f"../dataset/{data_name}/")  # 读取train_triple是为了filter训练方法

    entity_dict, relation_dict, _ = \
        test_loader(f"../res/entity_{dim}dim_{data_name}_batch200", f"../res/relation_{dim}dim_{data_name}_batch200", f"../dataset/{data_name}/test.txt")

    json_file_path = 'rectified_test_wn18rr_set.json'

    # 读取并处理 JSON 文件
    # 这里的test_head_emb, test_relation_emb, test_tail_emb就是实际会用到的所有triple的embedding
    # test_triple是真正的测试数据集
    test_triple, test_entity_emb, test_head_emb, test_relation_emb, test_tail_emb = read_and_process_json(json_file_path)

    # 保存到 TXT 文件中
    save_to_txt('test_head_emb.txt', test_head_emb)
    save_to_txt('test_relation_emb.txt', test_relation_emb)
    save_to_txt('test_tail_emb.txt', test_tail_emb)


    explain = ExplainWithPrototype(entity_dict, relation_dict,train_triple, test_triple,test_entity_emb, check_file_path,dim, alpha, isFit=False)
    # explain.prototype_generator()
    # print("Embedding Shifting...")
    explain.embedding_shifting()
    # 进行embedding shifting 之后

    f = open(f"../res/get_explain_on_{data_name}_{current_time}_a{alpha}.txt", 'w')
    f.write("Test hits@1: " + str(explain.hits1) + '\n')
    f.write("Test MRR: " + str(explain.MRR) + '\n')
    f.write("Final hits@1: " + str(explain._hits1) + '\n')
    f.write("Final MRR: " + str(explain._MRR) + '\n')
    f.close()

    end_time = time.time()  # 结束时间
    total_time = end_time - start_time  # 计算总体执行时间
    print(f"Total execution time: {total_time:.2f} seconds")  # 打印总体执行时间