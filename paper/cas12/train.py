import json
from numbers import Number

import torch
from common_ai.generator import MyGenerator
from common_ai.initializer import MyInitializer
from common_ai.lr_scheduler import MyLrScheduler
from common_ai.optimizer import MyOptimizer
from tqdm import tqdm

from AI.dataset import MyDataset
from AI.metric import MSE, GreatestCommonCrossEntropy
from AI.preprocess.CRIfuser.model import CRIfuser


def train_model(cfg: dict, model_name: str, data_name: str) -> None:
    checkpoint = None
    last_epoch = -1
    for epoch in range(cfg["train"]["num_epochs"] - 1, -1, -1):
        pt_path = (
            cfg["output_dir"]
            / "trains"
            / model_name
            / f"checkpoint-{epoch}"
            / "checkpoint.pt"
        )
        if pt_path.exists():
            checkpoint = torch.load(
                pt_path,
                weights_only=False,
            )
            last_epoch = epoch
            break

    my_generator = MyGenerator(**cfg["train"]["generator"])
    if checkpoint is not None:
        my_generator.load_state_dict(checkpoint["generator"])

    model = eval(model_name)(**cfg[model_name])
    # Move model to the device before setup optimizer because some optimizers like Adagrad use device information.
    setattr(model, "device", cfg["train"]["device"])
    model.to(model.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    else:
        my_initializer = MyInitializer(**cfg["train"]["initializer"])
        if hasattr(model, "my_initialize_model"):
            model.my_initialize_model(my_initializer, my_generator)
        else:
            my_initializer(model, my_generator)

    if isinstance(model, torch.nn.Module):
        my_optimizer = MyOptimizer(**cfg["train"]["optimizer"])
        my_optimizer(model)
        my_lr_scheduler = MyLrScheduler(**cfg["train"]["lr_scheduler"])
        my_lr_scheduler(my_optimizer)
        if checkpoint is not None:
            my_optimizer.load_state_dict(checkpoint["optimizer"])
            my_lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    metrics = {
        name: eval(name)(**param) for name, param in cfg["train"]["metrics"].items()
    }

    dataset = MyDataset(**cfg["train"]["datasets"]["data_name"])()
    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset["train"],
        batch_size=cfg["train"]["batch_size"],
        collate_fn=lambda examples: examples,
        shuffle=True,
        generator=my_generator.torch_c_rng,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        dataset=dataset["validation"],
        batch_size=cfg["train"]["batch_size"],
        collate_fn=lambda examples: examples,
    )

    for epoch in tqdm(range(last_epoch + 1, cfg["train"]["num_epochs"])):
        # train
        model.train()
        model.zero_grad()  # optimizer.zero_grad() is different when multiple models share a common optimizer
        train_loss, train_loss_num, grad_norm = 0.0, 0.0, 0.0
        for step, examples in tqdm(enumerate(train_dataloader)):
            batch = model.data_collator(
                examples, output_label=True, my_generator=my_generator
            )

            result = model(
                input=batch["input"],
                label=batch["label"],
                my_generator=my_generator,
            )

            result["loss"].backward()
            if (step + 1) % cfg["train"]["accumulate_steps"] == 0 or step == len(
                train_dataloader
            ):
                grad_norm += torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=cfg["train"]["clip_value"]
                ).item()
                my_optimizer.step()
                model.zero_grad()

            train_loss += (
                result["loss"].item()
                if not isinstance(result["loss"], Number)
                else result["loss"]
            )
            train_loss_num += (
                result["loss_num"].item()
                if not isinstance(result["loss_num"], Number)
                else result["loss_num"]
            )

        if model_name == "inDelphi":
            model.train_knn(train_dataloader, eval_dataloader, my_generator)

        # eval
        with torch.no_grad():
            if isinstance(model, torch.nn.Module):
                model.eval()
            eval_loss, eval_loss_num = 0.0, 0.0
            for examples in tqdm(eval_dataloader):
                batch = model.data_collator(
                    examples, output_label=True, my_generator=my_generator
                )
                result = model(
                    input=batch["input"],
                    label=batch["label"],
                    my_generator=my_generator,
                )

                eval_loss += (
                    result["loss"].item()
                    if not isinstance(result["loss"], Number)
                    else result["loss"]
                )
                eval_loss_num += (
                    result["loss_num"].item()
                    if not isinstance(result["loss_num"], Number)
                    else result["loss_num"]
                )
                df = model.eval_output(examples, batch, my_generator)
                for metric_name, metric_fun in metrics.items():
                    metric_fun.step(
                        df=df,
                        examples=examples,
                        batch=batch,
                    )

            metric_loss_dict = {
                metric_name: metric_fun.epoch()
                for metric_name, metric_fun in metrics.items()
            }

        # log
        loginfo = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_num": train_loss_num,
            "grad_norm": grad_norm,
            "eval_loss": eval_loss,
            "eval_loss_num": eval_loss_num,
            **metric_loss_dict,
        }

        if isinstance(model, torch.nn.Module):
            loginfo["learning_rate"] = my_lr_scheduler.get_last_lr()[0]
            my_lr_scheduler.step(eval_loss / eval_loss_num)

        (cfg["output_dir"] / "trains" / model_name).mkdir(exist_ok=True, parents=True)
        with open(cfg["output_dir"] / "trains" / model_name / "info.log", "a") as fd:
            json.dump(loginfo, fd)
            fd.flush()

        # save
        (cfg["output_dir"] / "trains" / model_name / f"checkpoint-{epoch}").mkdir(
            exist_ok=True, parents=True
        )

        checkpoint = {
            "model": model.state_dict(),
            "generator": my_generator.state_dict(),
        }
        if isinstance(model, torch.nn.Module):
            checkpoint.update({
                "optimizer": my_optimizer.state_dict(),
                "lr_scheduler": my_lr_scheduler.state_dict(),
            })

        torch.save(
            obj=checkpoint,
            f=cfg["output_dir"]
            / "trains"
            / model_name
            / f"checkpoint-{epoch}"
            / "checkpoint.pt",
        )
