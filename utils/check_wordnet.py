import nltk
from nltk.corpus import wordnet as wn


nltk.download('wordnet')
nltk.download('omw-1.4')  # Optional, for additional wordnet languages



# Function to look up synset by ID
def lookup_synset(synset_offset):
    return wn.synset_from_pos_and_offset('n', int(synset_offset))

# Example synset IDs from your triples
synset_ids = ['0645599', '06349220', '04564698', '04341686', '03754979', '00033020', '01740393']

# Lookup and print details
for synset_id in synset_ids:
    synset = lookup_synset(synset_id)
    print(f'Synset ID: {synset_id}')
    print(f'  Lemma Names: {synset.lemma_names()}')
    print(f'  Definition: {synset.definition()}')
    print(f'  Examples: {synset.examples()}')
