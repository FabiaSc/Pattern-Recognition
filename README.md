# Pattern-Recognition
### Group exercises
For MLP how de read the results :
  1) Hyperparameter search (train/val)
  grid_results.csv
  One row per hyper-parameter combo (hidden size, lr, etc.) with the validation accuracy (val_acc). Sort by val_acc to see the winner.
  
  mlp_<i>_loss.png and mlp_<i>_acc.png
  Training vs validation curves for combo i.
  i is the index of the combo in the grid list inside the code (1, 2, 3, …). 
      The train–validation curves across 15 epochs for each grid combination i, in this order :
        {'hidden_size': 64, 'lr': 1e-3, 'dropout': 0.10, 'weight_decay': 0.0}
        {'hidden_size': 128, 'lr': 1e-3, 'dropout': 0.10, 'weight_decay': 0.0}
        {'hidden_size': 256, 'lr': 1e-3, 'dropout': 0.20, 'weight_decay': 0.0}
        {'hidden_size': 128, 'lr': 5e-4, 'dropout': 0.10, 'weight_decay': 0.0}
        {'hidden_size': 128, 'lr': 1e-4, 'dropout': 0.20, 'weight_decay': 1e-4}
  
  best_hparams.json
  The winning hyperparameters (the combo with the best val_acc).
  
  best_on_val_mlp.pth
  Model weights saved at the best validation epoch for the winning combo.
  
  2) Final training (train + val)
  final_train_history.csv
  Training history (loss/accuracy) when retraining on train+val using the winning hyperparameters.
  
  final_train_loss.png, final_train_acc.png
  Plots for the final training (no validation curves here—everything is used for training).
  
  final_mlp_trainval.pth
  Final model weights (this is the model used to evaluate the test set).
  
  3) Test metric
  The test accuracy is printed in the console at the very end as:
  >>> TEST accuracy: 0.9770
