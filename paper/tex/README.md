# Advantage

- Previous models like inDelphi, FOREcasT and Lindel cannot predict templated insertion.
- Previous models cannot handle the case with simultaneous insertion and deletion.
- Previous models restricts insertion up to 2bp. (inDelphi and FOREcasT 1bp, Lindel 2bp). Our model support insertion up to 6bp.
- We try multiple architectures and find the diffusion architecture outperforms transformer, LSTM, CNN, MLP, ridge regression, XGBoost.
- We retrain inDelphi, FOREcasT and Lindel on the NAA pam data. Our diffusion model is better than them.

# TODO

- select loss function before selecting architecture and doing benchmark
- diffusion dynamic figures
- probability heatmap
