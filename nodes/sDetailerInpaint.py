import torch
import random
import comfy
import numpy as np
from PIL import Image
import cv2
from comfy.samplers import KSampler
import comfy.utils
import latent_preview
from io import BytesIO
import struct
from server import PromptServer
print("sDetailerInpaint module loaded successfully")

class SDetailerInpaintHelper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 30.0, "step": 0.1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150, "step": 1}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
            },
        }

    CATEGORY = "sDetailer"
    RETURN_TYPES = ("IMAGE", "LATENT")
    FUNCTION = "inpaint"

    @classmethod
    def IS_CHANGED(cls, strength, guidance_scale, steps, scheduler, sampler_name, seed, mask_blur=4, **kwargs):
        if seed == 0:
            return (strength, guidance_scale, steps, scheduler, sampler_name, random.randint(0, 0xffffffffffffffff), mask_blur)
        return (strength, guidance_scale, steps, scheduler, sampler_name, seed, mask_blur)

    def inpaint(self, image, mask, model, clip, vae, positive, negative,
                strength, guidance_scale, steps, scheduler, sampler_name, seed,
                mask_blur=4):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if mask is None:
            print("SDetailerInpaintHelper: Mask is None, returning original image")
            latent = vae.encode(image)
            return (image, {"samples": latent})

        mask_sum = torch.sum(mask).item()

        if mask_sum == 0:
            print("SDetailerInpaintHelper: Mask is empty (sum == 0), returning original image")
            latent = vae.encode(image)
            return (image, {"samples": latent})


        latent = vae.encode(image)

        if mask.ndim == 4:
            mask = mask.permute(0, 3, 1, 2)[:, 0, :, :]
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)

        target_device = latent.device

        if mask.device != target_device:
            mask = mask.to(target_device)

        if mask.shape[1] != latent.shape[2] or mask.shape[2] != latent.shape[3]:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1),
                size=(latent.shape[2], latent.shape[3]),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        if mask_blur > 0:
            if device == "cuda":
                try:
                    sigma = mask_blur / 3.0
                    try:
                        import kornia
                        mask_for_blur = mask.unsqueeze(1)
                        denoise_mask = kornia.filters.gaussian_blur2d(
                            mask_for_blur,
                            kernel_size=(mask_blur*2+1, mask_blur*2+1),
                            sigma=(sigma, sigma)
                        ).squeeze(1)
                        
                        compensate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        mask_np_blurred = denoise_mask[0].cpu().numpy()
                        _, mask_np_thresh = cv2.threshold(mask_np_blurred, 0.1, 1.0, cv2.THRESH_BINARY)
                        mask_np_eroded = cv2.erode((mask_np_thresh * 255).astype(np.uint8), compensate_kernel, iterations=1)
                        denoise_mask = torch.from_numpy(mask_np_eroded / 255.0).unsqueeze(0).to(target_device)

                    except ImportError:
                        raise Exception("Kornia not available")
                except:
                    blur_ksize = mask_blur * 2 + 1
                    mask_np = mask[0].cpu().numpy()
                    blurred_mask_np = cv2.GaussianBlur(mask_np, (blur_ksize, blur_ksize), 0)
                    denoise_mask = torch.from_numpy(blurred_mask_np).unsqueeze(0).to(target_device)
                    
                    compensate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    _, mask_np_thresh = cv2.threshold(blurred_mask_np, 0.1, 1.0, cv2.THRESH_BINARY)
                    mask_np_eroded = cv2.erode((mask_np_thresh * 255).astype(np.uint8), compensate_kernel, iterations=1)
                    denoise_mask = torch.from_numpy(mask_np_eroded / 255.0).unsqueeze(0).to(target_device)

            else:
                blur_ksize = mask_blur * 2 + 1
                mask_np = mask[0].cpu().numpy()
                blurred_mask_np = cv2.GaussianBlur(mask_np, (blur_ksize, blur_ksize), 0)
                
                compensate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                _, mask_np_thresh = cv2.threshold(blurred_mask_np, 0.1, 1.0, cv2.THRESH_BINARY)
                mask_np_eroded = cv2.erode((mask_np_thresh * 255).astype(np.uint8), compensate_kernel, iterations=1)
                denoise_mask = torch.from_numpy(mask_np_eroded / 255.0).unsqueeze(0).to(target_device)
        else:
            denoise_mask = mask


        noise = comfy.sample.prepare_noise(latent, seed)

        sampler = KSampler(
            model=model,
            steps=steps,
            device=model.load_device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=strength,
            model_options=model.model_options
        )

        previewer = latent_preview.get_previewer(device, model.model.latent_format)
        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps, None)
            if previewer:
                try:
                    preview_image_pil = previewer.decode_latent_to_preview_image("JPEG", x0)[1]
                    swarm_send_extra_preview(0, preview_image_pil)
                except Exception as e:
                    print(f"SDetailerInpaintHelper: Error in preview callback: {e}")

        samples = sampler.sample(
            noise=noise,
            positive=positive,
            negative=negative,
            cfg=guidance_scale,
            latent_image=latent,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            denoise_mask=denoise_mask,
            sigmas=None,
            callback=callback
        )

        result_img = vae.decode(samples)

        return (result_img, {"samples": samples})


NODE_CLASS_MAPPINGS = {
    "SDetailerInpaintHelper": SDetailerInpaintHelper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDetailerInpaintHelper": "sDetailer Inpaint Helper",
}

def swarm_send_extra_preview(id, image):
    server = PromptServer.instance
    bytesIO = BytesIO()
    num_data = 1 + (id * 16)
    header = struct.pack(">I", num_data)
    bytesIO.write(header)
    image.save(bytesIO, format="JPEG", quality=90, compress_level=4)
    preview_bytes = bytesIO.getvalue()
    server.send_sync(1, preview_bytes, sid=server.client_id)
