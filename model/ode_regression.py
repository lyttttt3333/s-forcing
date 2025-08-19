import torch.nn.functional as F
from typing import Tuple
import torch

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class ODERegression(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details
        """
        super().__init__(args, device)

        # Step 1: Initialize all models

        # self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        # self.generator.model.requires_grad_(True)
        # if getattr(args, "generator_ckpt", False):
        #     print(f"Loading pretrained generator from {args.generator_ckpt}")
        #     state_dict = torch.load(args.generator_ckpt, map_location="cpu")[
        #         'generator']
        #     self.generator.load_state_dict(
        #         state_dict, strict=True
        #     )

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

        self.scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=1000,
                        shift=1,
                        use_dynamic_shifting=False)
        self.scheduler.set_timesteps(50, device=self.device, shift=5)
        full_timestep = self.scheduler.timesteps
        sample_step = [0,36,44,49]
        # self.denoising_step_list = []
        # for step in sample_step:
        #     self.denoising_step_list.append(full_timestep[step].to(torch.int64).unsqueeze(0))
        denoising_step_list = [1000, 750, 500, 250]
        self.denoising_step_list = torch.tensor(denoising_step_list).to(self.device).to(torch.int64)

        print(f"########### denoising list {self.denoising_step_list} ############")


    # def _initialize_models(self, args, device):
    #     self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
    #     self.generator.model.requires_grad_(True)

    #     # self.text_encoder = WanTextEncoder()
    #     # self.text_encoder.requires_grad_(False)

    #     self.vae = WanVAEWrapper()
    #     self.vae.requires_grad_(False)

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor, eval = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        ode_latent = ode_latent.transpose(3,2)
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape

        # Step 1: Randomly choose a timestep for each frame
        index = self._get_timestep(
            0,
            self.denoising_step_list.shape[0],
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=False
        )

        if eval:
            index = 0 * index

        index[:, 0] = self.denoising_step_list.shape[0] - 1

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width).to(self.device)
        ).squeeze(1)

        timestep = self.denoising_step_list[index].to(self.device).to(self.dtype)

        # if self.extra_noise_step > 0:
        #     random_timestep = torch.randint(0, self.extra_noise_step, [
        #                                     batch_size, num_frames], device=self.device, dtype=torch.long)
        #     perturbed_noisy_input = self.scheduler.add_noise(
        #         noisy_input.flatten(0, 1),
        #         torch.randn_like(noisy_input.flatten(0, 1)),
        #         random_timestep.flatten(0, 1)
        #     ).detach().unflatten(0, (batch_size, num_frames)).type_as(noisy_input)

        #     noisy_input[timestep == 0] = perturbed_noisy_input[timestep == 0]


        noisy_input = noisy_input.transpose(2,1)

        return noisy_input, timestep
    
    @torch.no_grad()
    def _prepare_generator_input_online(self, target_latent: torch.Tensor, eval = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        target_latent = target_latent.transpose(2,1)
        batch_size, num_frames, num_channels, height, width = target_latent.shape

        # Step 1: Randomly choose a timestep for each frame
        index = self._get_timestep(
            0,
            self.denoising_step_list.shape[0],
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=False
        )

        index[:, 0] = self.denoising_step_list.shape[0] - 1

        timestep = self.denoising_step_list[index].to(self.device).to(self.dtype)
        timestep[0,0] = 0

        print(f"########### {timestep.view(-1)} {timestep.shape} ############")

        noisy_input = self.generator.scheduler.add_noise(original_samples=target_latent,
                                 noise=torch.randn_like(target_latent),
                                 timestep=timestep.view(-1))
    


        noisy_input = noisy_input.transpose(2,1)

        return noisy_input, timestep

    def generator_loss(self, ode_latent: torch.Tensor, conditional_dict: dict, unconditional_dict:dict, step) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        target_latent = ode_latent[:, -1]
        # target_latent [1,48,21,30,40]

        noisy_input, timestep_frame_level = self._prepare_generator_input_online(
            target_latent=target_latent)
        # noisy input [1,48,21,30,40]
        timestep = timestep_frame_level.clone()
        timestep = timestep.unsqueeze(-1).expand(-1, -1, int(noisy_input.shape[3]*noisy_input.shape[4]/4))
        timestep = timestep.reshape(1,-1)

        seq_len = int(noisy_input.shape[2]*noisy_input.shape[3]*noisy_input.shape[4]/4)

        pred_real_image_cond = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            seq_len=seq_len,
        ).to(torch.bfloat16)

        # pred_real_image_uncond = self.generator(
        #     noisy_image_or_video=noisy_input,
        #     conditional_dict=unconditional_dict,
        #     timestep=timestep,
        #     seq_len=seq_len,
        # ).to(torch.bfloat16)

        # pred_real_image = pred_real_image_cond + (
        #     pred_real_image_cond - pred_real_image_uncond
        # ) * 5

        pred_real_image = pred_real_image_cond

        pred_real_image = self.generator._convert_flow_pred_to_x0(flow_pred=pred_real_image,
                                                xt=noisy_input,
                                                timestep=timestep_frame_level.reshape(-1))

        # Step 2: Compute the regression loss
        mask = timestep_frame_level != (self.denoising_step_list.shape[0] - 1)
        mask = mask.view(-1)

        loss = F.mse_loss(
            pred_real_image[:,:,mask,:,:], target_latent[:,:,mask,:,:], reduction="mean").float()

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_real_image, target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
            "timestep": timestep.float().mean(dim=1).detach(),
            "input": noisy_input.detach(),
            "output": pred_real_image.detach(),
        }

        return loss, log_dict
    
    def train_multi_step(self, ode_latent: torch.Tensor, conditional_dict: dict, unconditional_dict:dict, step) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        target_latent = ode_latent[:, -1]

        noisy_input, timestep_frame_level = self._prepare_generator_input(
            ode_latent=ode_latent, eval=True)
        # noisy input [1,48,21,30,40]
        noisy_input_initial = noisy_input.clone()

        # noisy_input = torch.randn_like(noisy_input)

        timestep = timestep_frame_level.clone()
        timestep = timestep.unsqueeze(-1).expand(-1, -1, int(noisy_input.shape[3]*noisy_input.shape[4]/4))
        timestep = timestep.reshape(1,-1)

        seq_len = int(noisy_input.shape[2]*noisy_input.shape[3]*noisy_input.shape[4]/4)

        trajectory = []

        iteration = 2
        for i in range(iteration+1):

            timestep_frame_level = torch.ones_like(timestep_frame_level) * self.denoising_step_list[i]
            timestep_frame_level[:,0] = self.denoising_step_list[-1]

            timestep = timestep_frame_level.clone()
            timestep = timestep.unsqueeze(-1).expand(-1, -1, int(noisy_input.shape[3]*noisy_input.shape[4]/4))
            # timestep_frame_level [1,21]
            # timestep [1,21,300] -> [1,21*300]



            pred_real_image_cond = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                seq_len=seq_len,
            ).to(torch.bfloat16)

            pred_real_image_uncond = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                seq_len=seq_len,
            ).to(torch.bfloat16)

            pred_real_image = pred_real_image_cond + (
                pred_real_image_cond - pred_real_image_uncond
            ) * 5

            pred_real_image = self.generator._convert_flow_pred_to_x0(flow_pred=pred_real_image,
                                                    xt=noisy_input,
                                                    timestep=timestep_frame_level.reshape(-1))

            noisy_input = pred_real_image.clone()
            
            if i !=2 :
                noisy_input = self.generator.scheduler.add_noise(noisy_input,
                                                                    torch.rand_like(noisy_input),
                                                                    torch.ones_like(timestep_frame_level.view(-1)) * self.denoising_step_list[i+1])
                noisy_input = noisy_input.detach()
            
            

            # trajectory.append(pred_real_image)
        # mask = timestep_frame_level != (self.denoising_step_list.shape[0] - 1)
        # mask = mask.view(-1)

        # loss = F.mse_loss(
        #     pred_real_image[:,:,mask,:,:], target_latent[:,:,mask,:,:], reduction="mean").float()
        loss = F.mse_loss(
            pred_real_image, target_latent, reduction="mean").float()
        # trajectory = torch.cat(trajectory, dim = 0)

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_real_image, target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
            "timestep": timestep.float().mean(dim=1).detach(),
            "input": noisy_input_initial.detach(),
            "output": pred_real_image.detach(),
        }

        return loss, log_dict


    def eval_multi_step(self, ode_latent: torch.Tensor, conditional_dict: dict, unconditional_dict:dict) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        with torch.no_grad():
            target_latent = ode_latent[:, -1]

            noisy_input, timestep_frame_level = self._prepare_generator_input(
                ode_latent=ode_latent, eval=True)
            # noisy input [1,48,21,30,40]
            noisy_input_initial = noisy_input.clone()
            # noisy_input = torch.randn_like(noisy_input)

            timestep = timestep_frame_level.clone()
            timestep = timestep.unsqueeze(-1).expand(-1, -1, int(noisy_input.shape[3]*noisy_input.shape[4]/4))
            timestep = timestep.reshape(1,-1)

            seq_len = int(noisy_input.shape[2]*noisy_input.shape[3]*noisy_input.shape[4]/4)

            trajectory = []

            for i in range(3):

                timestep_frame_level = torch.ones_like(timestep_frame_level) * self.denoising_step_list[i]
                timestep_frame_level[:,0] = self.denoising_step_list[-1]

                timestep = timestep_frame_level.clone()
                timestep = timestep.unsqueeze(-1).expand(-1, -1, int(noisy_input.shape[3]*noisy_input.shape[4]/4))
                # timestep_frame_level [1,21]
                # timestep [1,21,300] -> [1,21*300]

                pred_real_image_cond = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    seq_len=seq_len,
                ).to(torch.bfloat16)

                # pred_real_image_uncond = self.generator(
                #     noisy_image_or_video=noisy_input,
                #     conditional_dict=unconditional_dict,
                #     timestep=timestep,
                #     seq_len=seq_len,
                # ).to(torch.bfloat16)

                # pred_real_image = pred_real_image_cond + (
                #     pred_real_image_cond - pred_real_image_uncond
                # ) * 5

                pred_real_image = pred_real_image_cond

                pred_real_image = self.generator._convert_flow_pred_to_x0(flow_pred=pred_real_image,
                                                        xt=noisy_input,
                                                        timestep=timestep_frame_level.reshape(-1))
                
                trajectory.append(pred_real_image)

                if i !=2 :
                    pred_real_image = self.generator.scheduler.add_noise(pred_real_image,
                                                                        torch.rand_like(pred_real_image),
                                                                        torch.ones_like(timestep_frame_level.view(-1)) * self.denoising_step_list[i+1])
                
                noisy_input = pred_real_image

                

            trajectory = torch.cat(trajectory, dim = 0)

            log_dict = {
                "unnormalized_loss": F.mse_loss(pred_real_image, target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
                "timestep": timestep.float().mean(dim=1).detach(),
                "input": noisy_input_initial.detach(),
                "output": trajectory.detach(),
            }

            return log_dict

