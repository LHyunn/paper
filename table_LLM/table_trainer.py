from transformers import Trainer, TrainerCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import numpy as np
import torch
class TableLLMTrainer(Trainer):
    def __init__(self, column_names, result_column, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.column_names = column_names
        self.result_column = result_column
        self.count = 0

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # get the labels
    #     labels = inputs.get("labels")
    #     # get the outputs
    #     outputs = model(**inputs)
    #     loss = outputs["loss"]
    #     logits = outputs["logits"]
    #     pred = torch.argmax(logits, dim=-1)
        
    #     # decode the labels and predictions
    #     labels_decoded = self.tokenizer.batch_decode(labels[:, -4:], skip_special_tokens=True)
    #     pred_decoded = self.tokenizer.batch_decode(pred[:, -5:], skip_special_tokens=True)
        

    #     try:
    #         # convert decoded strings to float values
    #         labels_float = torch.tensor([float(label.split()[-1]) for label in labels_decoded])
    #         pred_float = torch.tensor([float(p.split()[-1]) for p in pred_decoded])
            
    #         # calculate MSE 
    #         mse_loss = torch.nn.MSELoss()(pred_float, labels_float) * 0.1
    #     except:
    #         mse_loss = torch.tensor(3.6)
        
    #     # add MAE to the original loss
    #     total_loss = loss + mse_loss
        
    #     # update the loss in outputs
    #     outputs["loss"] = total_loss
        
    #     return (total_loss, outputs) if return_outputs else total_loss

        # loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        # mae_loss = torch.nn.L1Loss()(pred[:, -3].float(), labels[:, -2].float())
        # loss += mae_loss
        # print(f"loss: {loss}")
        # return (loss, outputs) if return_outputs else loss
    
class CustomCallbackForTableLLM(TrainerCallback):
    def __init__(self, tablellm, eval_dataframe, save_dir, max_retry=10, metric_to_check="R2"):
        self.tablellm = tablellm
        self.eval_dataframe = eval_dataframe
        self.origin_result = self.eval_dataframe['result'].tolist().copy()
        self.eval_dataframe['result'] = np.nan
        self.max_retry = max_retry
        self.metric_to_check = metric_to_check
        self.metric_value = -np.inf
        self.save_dir = save_dir

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        with torch.no_grad():
            count = 0
            while count < self.max_retry:
                try:
                    impute_result = list(map(float, self.tablellm.impute(self.eval_dataframe)['result'].tolist()))
                    mae = mean_absolute_error(self.origin_result, impute_result)
                    mse = mean_squared_error(self.origin_result, impute_result)
                    rmse = np.sqrt(mse)  # numpy를 사용하여 RMSE 계산
                    r_2 = r2_score(self.origin_result, impute_result)
                    metric = {
                        "MAE": mae,
                        "MSE": mse,
                        "RMSE": rmse,
                        "R2": r_2
                    }
                    print(f"\nImpute MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r_2}")
                    break
                except Exception as e:
                    count += 1
                    print(f"\n{e}, Retry {count} times")
                
                if count == self.max_retry:
                    print("Impute failed")
                    return

            if self.metric_value < metric[self.metric_to_check]:
                self.metric_value = metric[self.metric_to_check]
                self.tablellm.save(f"{self.save_dir}/best_model")
                print(f"\nBest model found with {self.metric_to_check}: {self.metric_value}")
            else:
                print(f"\nCurrent model is not the best model with {self.metric_to_check}: {metric[self.metric_to_check]}")
            

            
            

        


