import codecs

def extract_entities_relations(files):
    entities = set()
    relations = set()

    for file in files:
        with codecs.open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                head, relation, tail = parts
                entities.add(head)
                entities.add(tail)
                relations.add(relation)

    return list(entities), list(relations)

def write_to_file(items, file_path):
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(items):
            f.write(f"{item}\t{str(idx)}\n")

import codecs

def swap_columns(input_file, output_file):
    with codecs.open(input_file, 'r', encoding='utf-8') as infile, \
         codecs.open(output_file, 'w', encoding='utf-8') as outfile:

        lines = infile.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            # Swap the second and third columns
            swapped_line = f"{parts[0]}\t{parts[2]}\t{parts[1]}\n"
            outfile.write(swapped_line)


def main():
    # All entities and relations to be used later should have an ID
    # files = ["train.txt", "test.txt", "valid.txt", "adversarial_test.txt"]
    files = ["../dataset/fb15k-237/train.txt", "../dataset/fb15k-237/test.txt"]

    entities, relations = extract_entities_relations(files)

    write_to_file(entities, "../dataset/fb15k-237/entity2id.txt")
    write_to_file(relations, "../dataset/fb15k-237/relation2id.txt")

    # Swapping the 2nd and 3rd columns of the txt files.
    for file in files:
        input_path = f"../fb15k-237/{file}"
        output_path = f"../fb15k-237/{file.split('.')[0]}.txt"
        swap_columns(input_path, output_path)
        print(f"Processed {file}: saved as {file.split('.')[0]}.txt")


if __name__ == "__main__":
    main()
