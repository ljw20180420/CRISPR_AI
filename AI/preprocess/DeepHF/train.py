import pathlib
import logging
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from .model import DeepHFConfig, DeepHFModel
from .load_data import data_collator, get_energy, SeqTokenizer


def train_DeepHF(
    data_name: str,
    test_ratio: float,
    validation_ratio: float,
    ref1len: int,
    ref2len: int,
    random_insert_uplimit: int,
    insert_uplimit: int,
    owner: str,
    seq_length: int,
    em_drop: float,
    fc_drop: float,
    initializer: str,
    em_dim: int,
    rnn_units: int,
    fc_num_hidden_layers: int,
    fc_num_units: int,
    fc_activation: str,
    ext1_up: int,
    ext1_down: int,
    ext2_up: int,
    ext2_down: int,
    batch_size: int,
    optimizer: str,
    learning_rate: float,
    scheduler: str,
    num_epochs: float,
    warmup_ratio: float,
    output_dir: pathlib.Path,
    device: str,
    seed: int,
    logger: logging.Logger,
) -> None:
    logger.info("loading data")
    ds = load_dataset(
        path=f"{owner}/CRISPR_data",
        name=data_name,
        trust_remote_code=True,
        test_ratio=test_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
        ref1len=ref1len,
        ref2len=ref2len,
        random_insert_uplimit=random_insert_uplimit,
        insert_uplimit=insert_uplimit,
    )

    logger.info("initialize model")
    DeepHFConfig.register_for_auto_class()
    DeepHFModel.register_for_auto_class()
    DeepHF_model = DeepHFModel(
        DeepHFConfig(
            seq_length=seq_length,
            em_drop=em_drop,
            fc_drop=fc_drop,
            initializer=initializer,
            em_dim=em_dim,
            rnn_units=rnn_units,
            fc_num_hidden_layers=fc_num_hidden_layers,
            fc_num_units=fc_num_units,
            fc_activation=fc_activation,
            ext1_up=ext1_up,
            ext1_down=ext1_down,
            ext2_up=ext2_up,
            ext2_down=ext2_down,
            seed=seed,
        )
    )

    logger.info("train model")
    training_args = TrainingArguments(
        output_dir=output_dir / "DeepHF/DeepHF" / data_name,
        seed=seed,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        use_cpu=True if device == "cpu" else False,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        label_names=DeepHFConfig.label_names,
    )
    training_args.set_dataloader(
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )
    training_args.set_optimizer(
        name=optimizer,
        learning_rate=learning_rate,
    )
    training_args.set_lr_scheduler(
        name=scheduler,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
    )
    trainer = Trainer(
        model=DeepHF_model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=lambda examples, ext1_up=ext1_up, ext1_down=ext1_down, ext2_up=ext2_up, ext2_down=ext2_down, energy_records=get_energy(
            [
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NAA_scaffold_nbt_A1.csv.rnafold.sgRNA",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NAA_scaffold_nbt_A2.csv.rnafold.sgRNA",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NAA_scaffold_nbt_A3.csv.rnafold.sgRNA",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NGG_scaffold_nor_G1.csv.rnafold.sgRNA",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NGG_scaffold_nor_G2.csv.rnafold.sgRNA",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NGG_scaffold_nor_G3.csv.rnafold.sgRNA",
            ],
            [
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NAA_scaffold_nbt_A1.csv.rnafold.sgRNA+scaffold",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NAA_scaffold_nbt_A2.csv.rnafold.sgRNA+scaffold",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NAA_scaffold_nbt_A3.csv.rnafold.sgRNA+scaffold",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NGG_scaffold_nor_G1.csv.rnafold.sgRNA+scaffold",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NGG_scaffold_nor_G2.csv.rnafold.sgRNA+scaffold",
                "AI_models/DeepHF/rnafold/final_hgsgrna_libb_all_0811_NGG_scaffold_nor_G3.csv.rnafold.sgRNA+scaffold",
            ],
        ), seq_tokenizer=SeqTokenizer(
            "PSACGT"
        ): data_collator(
            examples,
            ext1_up,
            ext1_down,
            ext2_up,
            ext2_down,
            energy_records,
            seq_tokenizer,
            True,
        ),
    )
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        trainer.train()

    logger.info("save model")
    trainer.save_model()
    trainer.create_model_card()
