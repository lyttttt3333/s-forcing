from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist
import math
from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 vae,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.vae = vae
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

        self.real_guidance_scale = 5

        self.sampling_steps = 4
        num_train_timesteps = 1000
        shift = 5
        device = "cuda"

        self.scheduler = self.generator.get_scheduler()

        scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=1000,
                        shift=1,
                        use_dynamic_shifting=False)
        scheduler.set_timesteps(50, device=device, shift=5)
        full_timestep = scheduler.timesteps
        sample_step = [0,36,44,49]
        self.denoising_step_list = []
        for step in sample_step:
            self.denoising_step_list.append(full_timestep[step].to(torch.int64).unsqueeze(0))
        self.denoising_step_list = torch.cat(self.denoising_step_list, dim = 0)

        print("#######",self.denoising_step_list)
        self.sample_scheduler = scheduler


        # self.denoising_step_list = denoising_step_list

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def get_flow_pred(self, 
                    noisy_input,
                    conditional_dict,
                    unconditional_dict,
                    memory_token,
                    timestep,
                    timestep_frame,
                    t,
                    kv_cache,
                    crossattn_cache,
                    current_start,
                    seq_len,
                    idx):

        pred_real_image_cond = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=kv_cache,
            current_start=current_start,
            # memory_token=memory_token,
            seq_len=seq_len,
        )

        # pred_real_image_uncond = self.generator(
        #     noisy_image_or_video=noisy_input,
        #     conditional_dict=unconditional_dict,
        #     timestep=timestep,
        #     kv_cache=kv_cache,
        #     current_start=current_start,
        #     seq_len=seq_len
        # )

        # pred_real_image = pred_real_image_cond + (
        #     pred_real_image_cond - pred_real_image_uncond
        # ) * self.real_guidance_scale

        # pred_real_image = pred_real_image_cond.unsqueeze(0)  # [1, num_channels, num_frames, height, width]

        pred_real_image = pred_real_image_cond

        # pred_real_image = self.generator._convert_flow_pred_to_x0(flow_pred=pred_real_image,
        #                                         xt=noisy_input,
        #                                         timestep=timestep_frame.view(-1))


        if self.denoising_step_list.shape[0] - 1 >= idx + 1:
            t1 = self.denoising_step_list[idx]
            t2 = self.denoising_step_list[idx + 1]
        else:
            t1 = self.denoising_step_list[idx]
            t2 = 0
        pred_real_image = self.generator.scheduler.step_cross(model_output=pred_real_image,
                                                        sample=noisy_input,
                                                        timestep_t1= torch.ones_like(timestep_frame) * t1,
                                                        timestep_t2= torch.ones_like(timestep_frame) * t2,
                                                        )
        return pred_real_image
    
    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        frame_token: Optional[torch.Tensor] = None,
        memory_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        num_output_frames = num_frames
        seq_len = int(self.num_frame_per_block  *  height * width / 4)
        output = torch.zeros(
            [batch_size, num_channels, num_output_frames,  height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        frame_token = frame_token.unsqueeze(0) if frame_token is not None else None

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        # self._initialize_crossattn_cache(
        #     batch_size=batch_size, dtype=noise.dtype, device=noise.device
        # )
        self.state_init(conditional_dict, memory_token)

        # Step 2: Cache context feature

        self.crossattn_cache = None

        current_start_frame = 0

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, :, current_start_frame : current_start_frame + current_num_frames]

            mask = torch.ones_like(noisy_input)
            if block_index == 0 and frame_token is not None:
                mask[:, :, 0] = 0
            else:
                frame_token = torch.zeros_like(noisy_input)

            noisy_input = noisy_input * mask + frame_token * (1-mask)


            # Step 3.1: Spatial denoising loop
            denoising_step_list = self.denoising_step_list
            for index, current_timestep in enumerate(denoising_step_list):

                temp_ts = (mask[0][0][:, ::2, ::2] * current_timestep).flatten()
                timestep = temp_ts.unsqueeze(0)
                timestep_frame_level = timestep.view(1,self.num_frame_per_block,-1)[:,:,0]

                with torch.no_grad():
                    denoised_pred = self.get_flow_pred(
                        noisy_input=noisy_input,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        memory_token = memory_token,
                        timestep=timestep,
                        timestep_frame=timestep_frame_level,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        seq_len=seq_len,
                        t=current_timestep,
                        idx=index,
                    ) # output [1, num_channels, num_frames, height, width]
                    noisy_input = denoised_pred
                

                noisy_input = noisy_input * mask + frame_token * (1-mask)
            
            denoised_pred = denoised_pred * mask + frame_token * (1-mask)
            
            self.updata_3d_state(conditional_dict, idx = block_index, memory_token = memory_token)
                

            # Step 3.2: record the model's output
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            # denoised_pred = denoised_pred.transpose(1, 2).flatten(0, 1)  # [batch_size * current_num_frames, num_channels, height, width]
            # denoised_pred = self.scheduler.add_noise(
            #     denoised_pred,
            #     torch.randn_like(denoised_pred),
            #     next_timestep * torch.ones(
            #         [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            # )
            # denoised_pred = denoised_pred.unflatten(0, (batch_size, current_num_frames)).transpose(1, 2)  # [batch_size, num_channels, current_num_frames, height, width]
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    seq_len = seq_len,
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            # ### ONLY FOR TESTING
            # break

        return output

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        frame_token: Optional[torch.Tensor] = None,
        memory_token: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
    ) -> torch.Tensor:

        batch_size, num_channels, num_frames, height, width = noise.shape
        print(f"num_frames: {num_frames}, num_channels: {num_channels}, height: {height}, width: {width}")
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        num_output_frames = num_frames
        seq_len = int(self.num_frame_per_block  *  height * width / 4)
        output = torch.zeros(
            [batch_size, num_channels, num_output_frames,  height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        frame_token = frame_token.unsqueeze(0) if frame_token is not None else None

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        # self._initialize_crossattn_cache(
        #     batch_size=batch_size, dtype=noise.dtype, device=noise.device
        # )
        self.state_init(conditional_dict, memory_token)

        # Step 2: Cache context feature

        self.crossattn_cache = None

        current_start_frame = 0

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, :, current_start_frame : current_start_frame + current_num_frames]

            mask = torch.ones_like(noisy_input)
            if block_index == 0 and frame_token is not None:
                mask[:, :, 0] = 0
            else:
                frame_token = torch.zeros_like(noisy_input)

            noisy_input = noisy_input * mask + frame_token * (1-mask)


            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)

                temp_ts = (mask[0][0][:, ::2, ::2] * current_timestep).flatten()
                timestep = temp_ts.unsqueeze(0)
                timestep_frame_level = timestep.view(1,self.num_frame_per_block,-1)[:,:,0]

                if not exit_flag:
                    with torch.no_grad():
                        denoised_pred = self.get_flow_pred(
                            noisy_input=noisy_input,
                            conditional_dict=conditional_dict,
                            unconditional_dict=unconditional_dict,
                            memory_token = memory_token,
                            timestep=timestep,
                            timestep_frame=timestep_frame_level,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            seq_len=seq_len,
                            t=current_timestep,
                            idx=index
                        ) # output [1, num_channels, num_frames, height, width]
                        next_timestep = self.denoising_step_list[index + 1]
                        # denoised_pred = denoised_pred.transpose(1, 2).flatten(0, 1)  # [batch_size * current_num_frames, num_channels, height, width]
                        # noisy_input = self.scheduler.add_noise(
                        #     denoised_pred,
                        #     torch.randn_like(denoised_pred),
                        #     next_timestep * torch.ones(
                        #         [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        # )
                        # noisy_input = noisy_input.unflatten(0, (batch_size, current_num_frames)).transpose(1, 2)  # [batch_size, num_channels, current_num_frames, height, width]
                        # noisy_input = self.sample_scheduler.add_noise(
                        #     denoised_pred,
                        #     torch.randn_like(denoised_pred),
                        #     next_timestep * torch.ones([batch_size], device=noise.device, dtype=torch.long),
                        # )
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            denoised_pred = self.get_flow_pred(
                                noisy_input=noisy_input,
                                conditional_dict=conditional_dict,
                                unconditional_dict=unconditional_dict,
                                memory_token = memory_token,
                                timestep=timestep,
                                timestep_frame=timestep_frame_level,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                seq_len=seq_len,
                                t=current_timestep,
                                idx=index
                            )
                    else:
                        denoised_pred = self.get_flow_pred(
                            noisy_input=noisy_input,
                            conditional_dict=conditional_dict,
                            unconditional_dict=unconditional_dict,
                            memory_token = memory_token,
                            timestep=timestep,
                            timestep_frame=timestep_frame_level,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            seq_len=seq_len,
                            t=current_timestep,
                            idx=index
                        )
                    break

                noisy_input = noisy_input * mask + frame_token * (1-mask)

            denoised_pred = denoised_pred * mask + frame_token * (1-mask)
            
            self.updata_3d_state(conditional_dict, idx = block_index, memory_token = memory_token)
                

            # Step 3.2: record the model's output
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            # denoised_pred = denoised_pred.transpose(1, 2).flatten(0, 1)  # [batch_size * current_num_frames, num_channels, height, width]
            # denoised_pred = self.scheduler.add_noise(
            #     denoised_pred,
            #     torch.randn_like(denoised_pred),
            #     next_timestep * torch.ones(
            #         [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            # )
            # denoised_pred = denoised_pred.unflatten(0, (batch_size, current_num_frames)).transpose(1, 2)  # [batch_size, num_channels, current_num_frames, height, width]
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    seq_len = seq_len
                )
            # self.detach_kv_cache()

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def detach_kv_cache(self):
        """
        Detach all tensors in kv_cache1 from the computation graph.
        """
        for block_cache in self.kv_cache1:
            for key in block_cache:
                if torch.is_tensor(block_cache[key]):
                    block_cache[key] = block_cache[key].detach()

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 24, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def updata_3d_state(self, conditional_dict, idx, latent_to_decode = None, memory_token = None):
        pass
        # if latent_to_decode is not None:
        #     device = latent_to_decode.device
        #     dtype = latent_to_decode.dtype
        #     latent_to_decode = latent_to_decode[0].transpose(1,0)
        #     pixel_video_clip = self.vae.decode_to_pixel([latent_to_decode])
        #     state = torch.zeros([1, 1041, 2048]).to(device).to(dtype)
        #     self.state_cache = torch.zeros([1, 1041, 2048]).to(device).to(dtype)
        # elif memory_token is not None:
        #     state = memory_token[idx].unsqueeze(0)
        #     self.state_cache = None
        # conditional_dict["state"] = state

    def state_init(self, conditional_dict, memory_token):
        pass
        # device = conditional_dict["prompt_embeds"].device
        # dtype = conditional_dict["prompt_embeds"].dtype
        # # state = torch.zeros([1, 1041, 2048]).to(device).to(dtype)
        # memory_token.to(device).to(dtype)
        # state = memory_token[0].unsqueeze(0)
        # conditional_dict["state"] = state
