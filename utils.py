import torch;
import numpy as np;
import pandas as pd;

def dataFrameStatus(df: pd.DataFrame) -> None:
    print(f"LOOKING AT DF size({len(df)})\n+====\n");

    for col in df.columns:
        na_vals = df[col].isna().sum();
        null_vals = df[col].isnull().sum();
        none_vals = df[col].eq(None).sum();

        total_missing = na_vals + null_vals + none_vals
        
        if total_missing == 0:
            print(f"| FEATURE {col} CONTAINS NO EMPTY VALUES ++\n---");
            continue;
        
        print(f"| FEATURE {col} :");
        print(f"| NULL COUNT -> {null_vals} | NaN COUNT -> {na_vals} | NONE COUNT -> {none_vals}");
        print(f"| TOTAL MISSING VALUES {total_missing} -> {(total_missing/len(df)) * 100:.2f}%\n---");
    
    print("\n+====\n");
    return;


def batchAccuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Returns the average accuracy of each predicted sample compared to the labels,
    based on how close the predicted price is to the actual price.

    The accuracy is calculated as the percentage difference between the predicted 
    and actual values, where a smaller difference results in higher accuracy.

    Args:
    preds (torch.Tensor): Predicted prices
    labels (torch.Tensor): Actual prices

    Returns:
    float: Average accuracy as a percentage of how close the predicted price is to the actual price
    """

    # Ensure preds and labels are 1D tensors and move to CPU if necessary
    preds = preds.squeeze().detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # Calculate the absolute percentage error between predicted and actual values
    errors = np.abs(preds - labels) / np.abs(labels) * 100

    # Compute the accuracy as 100% minus the average percentage error
    accuracy_mean = 100 - np.mean(errors)
    return accuracy_mean