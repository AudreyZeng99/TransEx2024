def name_2_id(input_file_path, entity2id, relation2id):

    set_e = set()
    set_r = set()

    with open(str(input_file_path),'r') as file:
        lines = file.readlines()

    with open(entity2id, 'w') as output_file:
        for line in lines:
            elements = line.strip().split('\t')
            if elements:
                h = elements[0]
                t = elements[1]

                if h not in set_e:
                    output_file.write(h+'\n')
                    set_e.add(h) # note that h is written into output_file

                if t not in set_e:
                    output_file.write(t + '\n')
                    set_e.add(t)  # note that h is written into output_file

    with open(relation2id, 'w') as output_file:
        for line in lines:
            elements = line.strip().split('\t')
            if elements:
                r = elements[2]

                if r not in set_r:
                    output_file.write(r + '\n')
                    set_r.add(r)  # note that h is written into output_file


def main():
    file_list = ['train.txt', 'test.txt']
    for input_file in file_list:
        name_2_id(input_file, entity2id='entity2id.txt', relation2id='relation2id.txt')


if __name__ == "__main__":
    main()