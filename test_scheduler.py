from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import torch

sampling_steps = 4
num_train_timesteps = 1000
shift = 5
device = "cuda"

sample_scheduler = FlowUniPCMultistepScheduler(
    num_train_timesteps=num_train_timesteps,
    shift=1,
    use_dynamic_shifting=False)
sample_scheduler.set_timesteps(
    sampling_steps, device=device, shift=shift)
denoising_step_list = sample_scheduler.timesteps

print("Denoising step list:", denoising_step_list)

denoised_pred = torch.randn(1, 48, 3, 44, 78, device=device)
timestep = torch.ones([3], device=denoised_pred.device, dtype=torch.long)*999
noisy_input = sample_scheduler.add_noise(
    denoised_pred,
    torch.randn_like(denoised_pred),
    timestep,
)