#!/usr/bin/env python

from huggingface_hub import create_collection, add_collection_item

owner = "ljw20180420"

collection = create_collection(
    title="CRISPR ML",
    namespace=owner,
    description="Mechine learning models to predict the editing products of CRISPR spcas9/spymac/ispymac.",
    exists_ok=True,
)

for data_name in ["SX_spcas9", "SX_spymac", "SX_ispymac"]:
    for preprocess, model_cls in [
        ("CRIformer", "CRIformer"),
        ("inDelphi", "inDelphi"),
        ("Lindel", "Lindel"),
        ("DeepHF", "DeepHF"),
        ("DeepHF", "CNN"),
        ("DeepHF", "MLP"),
        ("DeepHF", "XGBoost"),
        ("DeepHF", "SGDClassifier"),
        ("CRIfuser", "CRIfuser"),
        ("FOREcasT", "FOREcasT"),
    ]:
        item_id = f"{owner}/{preprocess}_{model_cls}_{data_name}"
        add_collection_item(
            collection.slug,
            item_id=item_id,
            item_type="model",
            note=f"Preprocess method: {preprocess}. Model class: {model_cls}. Dataset name: {data_name}.",
            exists_ok=True,
        )
