# ChristianAI

## Installing Requirements

`pip3 install -r requirements.txt`

## Training

### Training Prep
**For Macbook with M-Series Chip**

`acceleerate config`

**Make sure to select these options**

```aiignore
In which compute environment are you running?
This machine                                                                                                                                                                                               
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                                                                                                                       
No distributed training                                                                                                                                                                                    
Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:NO                                                                                 
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO                                                                                                                                          
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use mixed precision?                                                                                                                                                                        
bf16       
```

### Create Embeddings:

`python3 create_embeddings.py`

### Train.py

```aiignore

░▒▓ ~/Documents/GIT-REPOS/ChristianAI  on main +9 !1 ?1 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ≡  at 03:32:31 AM ▓▒░─╮
❯ python3 train.py                                                                                                                                                                                      ─╯
Using device: mps
trainable params: 851,968 || all params: 1,236,666,368 || trainable%: 0.0689
{'loss': 1.0498, 'grad_norm': 4.2168803215026855, 'learning_rate': 0.000294200532691812, 'epoch': 0.13}                                                                                                    
{'loss': 0.9794, 'grad_norm': 93252.8203125, 'learning_rate': 0.00028775668012715865, 'epoch': 0.26}                                                                                                                                                                                                                                                                                                                                                                                                             
{'loss': 0.7728, 'grad_norm': 5.031601905822754, 'learning_rate': 4.2271672824125775e-06, 'epoch': 5.92}                                                                                                   
{'train_runtime': 65593.5223, 'train_samples_per_second': 2.845, 'train_steps_per_second': 0.711, 'train_loss': 0.8709704970790199, 'epoch': 6.0}                                                          
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 46656/46656 [18:13:13<00:00,  1.41s/it]

```

### Generate Embeddings

`python3 generate_embeddings.py`

## Run via Streamlit

`streamlit run streamlit_app.py`

## Run via RestFUL API

`python3 app.py`