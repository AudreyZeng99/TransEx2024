def name_2_id(input_file_path, dict_e, dict_r):
    with open(str(input_file_path), 'r') as file:
        lines = file.readlines()

    for line in lines:
        elements = line.strip().split('\t')
        if elements:
            h = elements[0]
            t = elements[1]
            r = elements[2]

            if h not in dict_e:
                dict_e[h] = len(dict_e)  # 为新实体分配唯一编号

            if t not in dict_e:
                dict_e[t] = len(dict_e)  # 为新实体分配唯一编号

            if r not in dict_r:
                dict_r[r] = len(dict_r)  # 为新关系分配唯一编号
    # return dict_e, dict_r


def write_to_file(filename, dict_obj):
    with open(filename, 'w') as file:
        for key, value in dict_obj.items():
            file.write(f'{key}\t{value}\n')


def main():
    dict_e = {}
    dict_r = {}
    file_list = ['train.txt', 'test.txt']
    for input_file in file_list:
        name_2_id(input_file, dict_e, dict_r)

    write_to_file('entity2id.txt', dict_e)
    write_to_file('relation2id.txt', dict_r)

if __name__ == "__main__":
    main()
