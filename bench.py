import pandas as pd
import pathlib

model_paths = [
    "/home/ljw/sdc1/CRISPR_results/CRIformer/CRIformer/SX_spcas9/default",
    "/home/ljw/sdc1/CRISPR_results/CRIfuser/CRIfuser/SX_spcas9/default",
    "/home/ljw/sdc1/CRISPR_results/DeepHF/DeepHF/SX_spcas9/default",
    "/home/ljw/sdc1/CRISPR_results/DeepHF/MLP/SX_spcas9/default",
    "/home/ljw/sdc1/CRISPR_results/DeepHF/CNN/SX_spcas9/default",
    "/home/ljw/sdc1/CRISPR_results/DeepHF/XGBoost/SX_spcas9/default",
    # "/home/ljw/sdc1/CRISPR_results/DeepHF/Ridge/SX_spcas9/default",
    "/home/ljw/sdc1/CRISPR_results/FOREcasT/FOREcasT/SX_spcas9/default",
    "/home/ljw/sdc1/CRISPR_results/inDelphi/inDelphi/SX_spcas9/default",
    "/home/ljw/sdc1/CRISPR_results/Lindel/Lindel/SX_spcas9/default",
]

metrics = []
for model_path in model_paths:
    model_path = pathlib.Path(model_path)
    df = pd.DataFrame(data=model_path / "test_result.csv")
    metrics.append(
        df["NonWildTypeCrossEntropy_loss"].sum()
        / df["NonWildTypeCrossEntropy_loss_num"].sum()
    )

pd.DataFrame({"model_path": model_paths, "metric": metrics})
