<<<<<<< HEAD
# TransEx
=======
# TransEx2024
>>>>>>> TransEx2024/main
A Prototype-enhanced Explanation for Embedding-based Link Prediction
## How to replica the experiment
### Data preparing
run file name2id_with_id.py to get the file "entity2id.txt" and "relation2id.txt" for each dataset.
### TransE training
after loading the dataset to /datasets, run file /src/trainer.py
### Test phrase
run /src/tester.py
### Explain phrase
run /src/before_explainer.py
run /src/explainer.py
