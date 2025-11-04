# Pattern-Recognition 

## Group exercises 
### Multilayer Perceptron (MLP)
#### Hyperparameter grid (train/val)
1) `{'hidden_size': 64,  'lr': 1e-3,  'dropout': 0.10, 'weight_decay': 0.0}`  
   ![Acc](results-MLP/mlp_1_acc.png)  
   ![Loss](results-MLP/mlp_1_loss.png)

2) `{'hidden_size': 128, 'lr': 1e-3,  'dropout': 0.10, 'weight_decay': 0.0}`  
   ![Acc](results-MLP/mlp_2_acc.png)  
   ![Loss](results-MLP/mlp_2_loss.png)

3) `{'hidden_size': 256, 'lr': 1e-3,  'dropout': 0.20, 'weight_decay': 0.0}`  
   ![Acc](results-MLP/mlp_3_acc.png)  
   ![Loss](results-MLP/mlp_3_loss.png)

4) `{'hidden_size': 128, 'lr': 5e-4,  'dropout': 0.10, 'weight_decay': 0.0}`  
   ![Acc](results-MLP/mlp_4_acc.png)  
   ![Loss](results-MLP/mlp_4_loss.png)

5) `{'hidden_size': 128, 'lr': 1e-4,  'dropout': 0.20, 'weight_decay': 1e-4}`  
   ![Acc](results-MLP/mlp_5_acc.png)  
   ![Loss](results-MLP/mlp_5_loss.png)

**Grid summary:** see `results/grid_results.csv` (one row per combo; sort by `val_acc`).
| Col A | Col B | Col C |
|-------|------:|:-----:|
| a1    |   12  |  ok   |
| a2    |  345  |  no   |

**Best hyperparameters:** stored in `results/best_hparams.json` (weights at best val epoch: `results/best_on_val_mlp.pth`).

### Final training (train + val with best hyperparameters)
Training history: `results/final_train_history.csv`  
![Final acc](results-MLP/final_train_acc.png)  
![Final loss](results-MLP/final_train_loss.png)

### Test performance
**TEST accuracy: 0.9770.**

### Observations
- `lr=1e-3` with `hidden_size≈128` and `dropout=0.1` gives the best stability/accuracy (~97% val).  
- Larger hidden size (256) overfits earlier (validation loss rises after ~5–7 epochs).  
- Lower `lr` (5e-4) is smoother but slower; `1e-4 + weight_decay` underfits.



For MLP how de read the results : 
1) Hyperparameter search (train/val)  

grid_results.csv 
One row per hyper-parameter combo (hidden size, lr, etc.) with the validation accuracy (val_acc). Sort by val_acc to see the winner. 

mlp_i_loss.png and mlp_i_acc.png  
Training vs validation curves for combo i.  
i is the index of the combo in the grid list inside the code (1, 2, 3, …).  
The train–validation curves across 15 epochs for each grid combination i, in this order :  

{'hidden_size': 64, 'lr': 1e-3, 'dropout': 0.10, 'weight_decay': 0.0}
{'hidden_size': 128, 'lr': 1e-3, 'dropout': 0.10, 'weight_decay': 0.0}
{'hidden_size': 256, 'lr': 1e-3, 'dropout': 0.20, 'weight_decay': 0.0}
{'hidden_size': 128, 'lr': 5e-4, 'dropout': 0.10, 'weight_decay': 0.0}
{'hidden_size': 128, 'lr': 1e-4, 'dropout': 0.20, 'weight_decay': 1e-4} 

best_hparams.json --> The winning hyperparameters (the combo with the best val_acc). 

best_on_val_mlp.pth --> Model weights saved at the best validation epoch for the winning combo. 

2) Final training (train + val)  

final_train_history.csv --> Training history (loss/accuracy) when retraining on train+val using the winning hyperparameters. 

final_train_loss.png, final_train_acc.png  
Plots for the final training (no validation curves here—everything is used for training). 

final_mlp_trainval.pth  
Final model weights (this is the model used to evaluate the test set). 

3) Test metric The test accuracy is printed in the console at the very end as: 
    TEST accuracy: 0.9770 

 
