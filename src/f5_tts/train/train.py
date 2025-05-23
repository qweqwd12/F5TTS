# training script.

import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer, DiT
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


import torch

os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name="F5TTS_Base")
def main(cfg):
    model_cls = hydra.utils.get_class(f"f5_tts.model.{cfg.model.backbone}")
    model_arc = cfg.model.arch
    tokenizer = cfg.model.tokenizer
    mel_spec_type = cfg.model.mel_spec.mel_spec_type

    exp_name = f"{cfg.model.name}_{mel_spec_type}_{cfg.model.tokenizer}_{cfg.datasets.name}"
    # wandb_resume_id = None

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = cfg.datasets.name
    else:
        tokenizer_path = cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # set model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    # # Load pre-trained weights if checkpoint_path is specified
    # if cfg.model.checkpoint_path:
    #     print(f"Loading pre-trained weights from {cfg.model.checkpoint_path}")
    #     checkpoint = torch.load(cfg.model.checkpoint_path)
    #     model.load_state_dict(checkpoint)
        
    # init trainer
        # 冻结与音色和音调无关的参数
    if isinstance(model.transformer, DiT):
        for param in model.transformer.time_embed.parameters():
            param.requires_grad = False
        for param in model.transformer.text_embed.parameters():
            param.requires_grad = False


    trainer = Trainer(
        model,
        epochs=cfg.optim.epochs,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path="/root/autodl-tmp/F5-TTS/ckpts/F5TTS_Base_vocos_pinyin_losl",
        batch_size_per_gpu=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        # wandb_project="CFM-TTS",
        # wandb_run_name=exp_name,
        # wandb_resume_id=wandb_resume_id,
        run_name=exp_name,
        last_per_updates=cfg.ckpts.last_per_updates,
        log_samples=cfg.ckpts.log_samples,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=cfg.model.vocoder.is_local,
        local_vocoder_path="/root/autodl-tmp/F5-TTS/ckpts/vocos",
        model_cfg_dict=OmegaConf.to_container(cfg, resolve=True),
        resume=False,
    )



    train_dataset = load_dataset(cfg.datasets.name, tokenizer, mel_spec_kwargs=cfg.model.mel_spec)
    trainer.train(
        train_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
