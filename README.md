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

**Grid summary:** see `results-MLP/grid_results.csv` (one row per combo; sort by `val_acc`).
| # | hidden_size | lr     | dropout | weight_decay | val_acc |
|---|-------------|--------|---------|--------------|--------:|
| 1 | 64          | 1e-3   | 0.10    | 0.0          | 0.9683  |
| 2 | 128         | 1e-3   | 0.10    | 0.0          | 0.9725  |
| 3 | 256         | 1e-3   | 0.20    | 0.0          | 0.9767  |
| 4 | 128         | 5e-4   | 0.10    | 0.0          | 0.9715  |
| 5 | 128         | 1e-4   | 0.20    | 1e-4         | 0.9547  |


**Best hyperparameters:** stored in `results-MLP/best_hparams.json` (weights at best val epoch: `results-MLP/best_on_val_mlp.pth`).
 `{'hidden_size': 256, 'lr': 1e-3,  'dropout': 0.20, 'weight_decay': 0.0}`

### Final training (train + val with best hyperparameters)
Training history: `results-MLP/final_train_history.csv`
No validation curves here, everything is used for training.
![Final acc](results-MLP/final_train_acc.png)  
![Final loss](results-MLP/final_train_loss.png)

### Test performance
The final model weights (this is the model used to evaluate the test set) `results-MLP/final_mlp_trainval.pth`
Performance on the testing set:
**Accuracy: 0.9770.**

 
