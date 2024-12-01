import torch;
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
    Returns the average accuracy of each predicted sample compared to the labels.

    e.g:
    predicted 5 -> actual 10,
    returns 0.5

    Args:
    preds (torch.Tensor): Predicted values
    labels (torch.Tensor): Actual labels

    Returns:
    float: Average accuracy
    """

    # Ensure preds and labels are 1D tensors
    preds = preds.squeeze().detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    accuracy_mean: float = sum(preds == labels) / len(preds)
    return accuracy_mean
