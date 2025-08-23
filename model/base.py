from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from pipeline import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.wan_wrapper_small import WanDiffusionWrapper_small


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.real_model_name = "Wan2.2-TI2V-5B"
        self.fake_model_name = "Wan2.1-T2V-1.3B"


        self.fake_score = WanDiffusionWrapper_small(model_name=self.fake_model_name, is_causal=False)
        self.generator.model.requires_grad_(True)

        # self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
        # self.real_score.model.requires_grad_(False)

        # self.fake_score = WanDiffusionWrapper_small(model_name=self.fake_model_name, is_causal=False)
        # self.fake_score.model.requires_grad_(True)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def load_text_model(self):
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

    def delete_text_model(self):
        if hasattr(self, "text_encoder"):
            del self.text_encoder
            self.text_encoder = None

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        frame_token: torch.tensor = None,
        memory_token: torch.tensor = None,
        initial_noise: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        # if initial_latent is not None:
        #     conditional_dict["initial_latent"] = initial_latent
        # if memory_token is not None:
        #     conditional_dict["memory_token"] = memory_token

        noise_shape = image_or_video_shape.copy()

        if initial_noise is None:
            noise = torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype)
        else:
            noise = initial_noise.to(self.device, dtype=self.dtype)


        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=noise,
            conditional_dict = conditional_dict,
            unconditional_dict = unconditional_dict,
            frame_token = frame_token,
            memory_token = memory_token,
        )
        # Slice last 21 frames
        if pred_image_or_video.shape[2] > 21:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :, :-20, ...]
                # Deccode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, :, -1:, ...].to(self.dtype)
                # frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, :, -20:, ...]], dim=2)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        # if num_generated_frames != min_num_frames:
        #     # Currently, we do not use gradient for the first chunk, since it contains image latents
        #     gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
        #     if self.args.independent_first_frame:
        #         gradient_mask[:, :1] = False
        #     else:
        #         gradient_mask[:, :self.num_frame_per_block] = False
        # else:
        #     gradient_mask = None

        gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        frame_token: torch.tensor = None,
        memory_token: torch.tensor = None,
        inference = False,
        causal_block = False,
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        if inference:
            return self.inference_pipeline.inference_with_causal_block(
                noise = noise,
                conditional_dict = conditional_dict,
                unconditional_dict = unconditional_dict,
                frame_token = frame_token,
                memory_token = memory_token,
            )
        
        elif causal_block:
            return self.inference_pipeline.inference_with_causal_block(
                noise = noise,
                conditional_dict = conditional_dict,
                unconditional_dict = unconditional_dict,
                frame_token = frame_token,
                memory_token = memory_token,
            )
        
        else:
            return self.inference_pipeline.inference_with_causal_block(
                noise = noise,
                conditional_dict =conditional_dict,
                unconditional_dict = unconditional_dict,
                frame_token = frame_token,
                memory_token = memory_token,
            )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            vae=self.vae,
            # num_frame_per_block=self.num_frame_per_block,
            # independent_first_frame=self.args.independent_first_frame,
            # same_step_across_blocks=self.args.same_step_across_blocks,
            # last_step_only=self.args.last_step_only,
            # num_max_frames=self.num_training_frames,
            # context_noise=self.args.context_noise
        )
