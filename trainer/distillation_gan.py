import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset, MixedDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from utils.lora import PeftModel

from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from torch.nn import ModuleList
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD, GAN
import torch
import wandb
import time
import os

import imageio
from tqdm import tqdm
import torchvision
import numpy as np

def save_video(video_tensor, save_path, fps=30, quality=9, ffmpeg_params=None):
    """
    保存一个形状为 [C, T, H, W] 的视频张量到文件
    video_tensor: torch.Tensor, float32, 值域 [-1, 1] 或 [0, 1]
    """
    assert video_tensor.dim() == 4, "video_tensor 必须是 4 维 [C, T, H, W]"
    
    # 转到 CPU，避免 GPU 张量直接参与 numpy 转换
    video_tensor = video_tensor.detach().cpu()

    # [-1, 1] → [0, 255]
    if video_tensor.min() < 0:
        video_tensor = (video_tensor + 1) / 2  # [-1,1] → [0,1]
    video_tensor = (video_tensor * 255).clamp(0, 255).byte()

    # [C, T, H, W] → [T, H, W, C]
    video_tensor = video_tensor.permute(1, 2, 3, 0).numpy()

    # 保存视频
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(video_tensor, desc="Saving video"):
        writer.append_data(frame)
    writer.close()


def get_lora_config():
    return LoraConfig(
        r=16,  # 低秩矩阵维度
        lora_alpha=32,
        target_modules=["q","k","v","o","ffn.0","ffn.2"],  # 目标模块（根据模型调整）
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            # wandb.login()
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        elif config.distribution_loss == "gan":
            self.model = GAN(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        # # print(config)
        # if getattr(config, "generator_ckpt", False):
        #     print(f"Loading pretrained generator from {config.generator_ckpt}")
        #     state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        #     if "generator" in state_dict:
        #         state_dict = state_dict["generator"]
        #     elif "model" in state_dict:
        #         state_dict = state_dict["model"]
        #     self.model.generator.load_state_dict(
        #         state_dict, strict=True
        #     )



        if config.resume_path is None:
            lora_config = get_lora_config()
            self.model.generator = PeftModel(
                    self.model.generator, 
                    lora_config,
                    adapter_name="default",
                    autocast_adapter_dtype=True,
                    low_cpu_mem_usage=False
                )
        else:
            resume_path = os.path.join(config.resume_path, "generator_model")
            print(f"Load from {resume_path}")
            self.model.generator = PeftModel.from_pretrained(self.model.generator, resume_path, is_trainable=True)
        self.model.generator.print_trainable_parameters() 

        if config.resume_path is None:
            lora_config = get_lora_config()
            self.model.fake_score = PeftModel(
                    self.model.fake_score, 
                    lora_config,
                    adapter_name="default",
                    autocast_adapter_dtype=True,
                    low_cpu_mem_usage=False
                )
        else:
            resume_path = os.path.join(config.resume_path, "fake_score_model")
            print(f"Load from {resume_path}")
            self.model.fake_score = PeftModel.from_pretrained(self.model.fake_score, resume_path, is_trainable=True)
        self.model.fake_score.print_trainable_parameters() 

        
        
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        # self.model.real_score = fsdp_wrap(
        #     self.model.real_score,
        #     sharding_strategy=config.sharding_strategy,
        #     mixed_precision=config.mixed_precision,
        #     wrap_strategy=config.real_score_fsdp_wrap_strategy
        # )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        # self.model.text_encoder = fsdp_wrap(
        #     self.model.text_encoder,
        #     sharding_strategy=config.sharding_strategy,
        #     mixed_precision=config.mixed_precision,
        #     wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
        #     cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        # )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        meta_path = self.config.meta_path
        root_dir = self.config.root_dir
        dataset = MixedDataset(meta_path, root_dir)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)
        self.dataset = dataset

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts


        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

        embed_dict_path = "ref_lib"
        self.load_embed_dict(embed_dict_path)

    def save(self):
        print("Start gathering distributed model states...")
        save_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        save_path_score = os.path.join(save_path, "fake_score_model")
        save_path_generator = os.path.join(save_path, "generator_model")
        os.makedirs(save_path_score, exist_ok=True)
        os.makedirs(save_path_generator, exist_ok=True)
        self.model.fake_score.save_pretrained(save_path_score)
        self.model.generator.save_pretrained(save_path_generator)

    def load_embed_dict(self, embed_dict_root):
        file_changed = False
        self.model.text_encoder = None
        global_dict_path = os.path.join(embed_dict_root, "global_embed_dict.pt")

        if not os.path.exists(global_dict_path):
            self.global_embed_dict = {}
            if self.model.text_encoder is None:
                self.model.load_text_model()
                self.model.text_encoder = fsdp_wrap(
                    self.model.text_encoder,
                    sharding_strategy=self.config.sharding_strategy,
                    mixed_precision=self.config.mixed_precision,
                    wrap_strategy=self.config.text_encoder_fsdp_wrap_strategy,
                    cpu_offload=getattr(self.config, "text_encoder_cpu_offload", False)
                )
            unconditional_dict = self.model.text_encoder(
                text_prompts=[self.config.negative_prompt])
            unconditional_dict = {k: v.detach().to("cpu", dtype=self.dtype)
                                    for k, v in unconditional_dict.items()}
            self.global_embed_dict = unconditional_dict
            os.makedirs(os.path.dirname(global_dict_path), exist_ok=True)
            torch.save(self.global_embed_dict, global_dict_path)
            print("Global dict saved to", global_dict_path)
        else:
            self.global_embed_dict = torch.load(global_dict_path, map_location="cpu")


        if self.model.text_encoder is not None:
            self.model.delete_text_model()

    def load_batch(self, batch):
        for key in batch.keys():
            if key != "base_name" and key != "clean_token":
                path = batch[key][0]
                tensor = torch.load(path, map_location="cpu").to(self.dtype).to(self.device)
                batch[key] = tensor
            if key == "clean_token":
                batch[key] = torch.zeros([1,48,21,44,78]).to(self.dtype).to(self.device)
        return batch

    def fwdbwd_one_step(self, batch, train_generator):
        batch = self.load_batch(batch)
        frame_token = batch["frame_token"]
        text_token = batch["text_token"]
        memory_token = batch["memory_token"]
        clean_token = batch["clean_token"]

        self.model.eval()  

        if self.step % 20 == 0:
            torch.cuda.empty_cache()


        image_or_video_shape = list(self.config.image_or_video_shape)

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = {'prompt_embeds': text_token}

            embed = self.global_embed_dict["prompt_embeds"].to(device=self.device, dtype=self.dtype)
            unconditional_dict = {'prompt_embeds': embed}

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                frame_token=frame_token,
                memory_token=memory_token,
                clean_token=clean_token,
            )

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict = {}
            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            real_image_or_video=clean_token,
            frame_token=frame_token,
            memory_token=memory_token
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    # def generate_video(self, pipeline, prompts, image=None):
    #     batch_size = len(prompts)
    #     if image is not None:
    #         image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

    #         # Encode the input image as the first latent
    #         initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
    #         initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
    #         sampled_noise = torch.randn(
    #             [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
    #             device="cuda",
    #             dtype=self.dtype
    #         )
    #     else:
    #         initial_latent = None
    #         sampled_noise = torch.randn(
    #             [batch_size, self.model.num_training_frames, 16, 60, 104],
    #             device="cuda",
    #             dtype=self.dtype
    #         )

    #     video, _ = pipeline.inference(
    #         noise=sampled_noise,
    #         text_prompts=prompts,
    #         return_latents=True,
    #         initial_latent=initial_latent
    #     )
    #     current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
    #     return current_video
    
    def generate_video(self, step):
        
        count = 0
        rank = dist.get_rank()
        MAX_COUNT = 1
        os.makedirs("tmp", exist_ok=True)
        txt_path = os.path.join("tmp", f"video_info_rank-{rank}.txt")
        with open(txt_path, "w") as f:
            while True:
                batch = self.dataset.get_examples()
                batch = self.load_batch(batch)
                frame_token = batch["frame_token"]
                text_token = batch["text_token"]
                memory_token = batch["memory_token"]
                clean_token = batch["clean_token"]

                conditional_dict = {'prompt_embeds': text_token}
                embed = self.global_embed_dict["prompt_embeds"].to(device=self.device, dtype=self.dtype)
                unconditional_dict = {'prompt_embeds': embed}

                base_name = batch["base_name"][0]
                print("base_name", base_name)

                image_or_video_shape = list(self.config.image_or_video_shape)


                video = self.model.generator_inference(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    frame_token=frame_token,
                    memory_token=memory_token,
                    clean_token=clean_token
                )
                print("decoder video shape",video.shape)

                output_path = os.path.join("tmp", f"teacher_{self.step:06d}_{base_name}.mp4")
                f.write(f"{base_name},{output_path}\n")

                # print(video.shape) 
                # print(video.dtype)  
                # print(video.min().item(), video.max().item()) 

                save_video(video, output_path, fps=15, quality=5)
            
                count += 1
                if count >= MAX_COUNT:
                    break

        dist.barrier()

        if wandb.run is not None:
            print("in main process")
            all_video_infos = []
            world_size = dist.get_world_size()
            for r in range(world_size):
                rank_txt = os.path.join("tmp", f"video_info_rank-{r}.txt")
                print(rank_txt)
                if os.path.exists(rank_txt):
                    print("exist")
                    with open(rank_txt, "r") as f:
                        for line in f:
                            base_name, output_path = line.strip().split(",", 1)
                            all_video_infos.append((base_name, output_path))

            for video_name, output_path in all_video_infos:
                print("log", video_name)
                wandb.log({f"gen/video_{video_name}": wandb.Video(output_path, fps=16, format="mp4")},step=step)
                # wandb.log({f"src/video_{video_name}": wandb.Video(input_path, fps=15, format="mp4")},step=steps)



    def train(self):
        start_step = self.step

        while True:
            
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            if TRAIN_GENERATOR:
                print("(Gen) Training step %d" % self.step)
            else:
                print("(Des) Training step %d" % self.step)


            EVALUATION = False
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
            
            if EVALUATION:
                self.generate_video(self.step)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
