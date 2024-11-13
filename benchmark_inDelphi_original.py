#!AI_models/inDelphi/reference/.conda/bin/python

with open("AI_models/inDelphi/reference/temp.csv", "w") as pd:
        pass
with open("AI_models/inDelphi/reference/temp.csv", "a") as pd, open("AI_models/inDelphi/reference/temp.log", "w") as ld:
    import sys
    sys.stdout = ld
    sys.stderr = ld

    from AI_models.inDelphi.reference.inDelphi import init_model, predict
    init_model(celltype=sys.argv[1])

    for line in sys.stdin:
        ref, cut = line.strip().split()
        cut = int(cut)
        pred_df, _ = predict(ref, cut)
        pred_df.to_csv(pd, header=False, index=False)
    