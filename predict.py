import os
import torch
import numpy as np
import gc
import logging
import traceback
from typing import Dict, Any
from PIL import Image

from diffusers.utils import export_to_video, load_image
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import CLIPVisionModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self):
        """Initialize the predictor"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Check CUDA availability and version
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Model IDs for the different variants
        self.t2v_model_id_small = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        self.t2v_model_id_large = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        self.i2v_model_id_480p = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        self.i2v_model_id_720p = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
        
        # Pre-load nothing to save time and memory
        self.t2v_pipe = None
        self.i2v_pipe = None
        
        # Create output directory
        os.makedirs("outputs", exist_ok=True)
        logger.info("Predictor initialized successfully")

    def load_t2v_model(self, model_size="small", resolution="720p"):
        """Load Text-to-Video model"""
        logger.info(f"Loading T2V model (size={model_size}, resolution={resolution})")
        try:
            # Clean up any previously loaded models
            if self.t2v_pipe is not None:
                del self.t2v_pipe
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
            
            # Select model based on size
            model_id = self.t2v_model_id_small if model_size == "small" else self.t2v_model_id_large
            
            # Set flow shift based on resolution
            flow_shift = 5.0 if resolution == "720p" else 3.0
            
            # Load VAE
            logger.info(f"Loading VAE from {model_id}")
            vae = AutoencoderKLWan.from_pretrained(
                model_id, 
                subfolder="vae", 
                torch_dtype=torch.float16
            )
            
            # Set up scheduler
            logger.info("Setting up scheduler with flow_shift={flow_shift}")
            scheduler = UniPCMultistepScheduler(
                prediction_type='flow_prediction',
                use_flow_sigmas=True,
                num_train_timesteps=1000,
                flow_shift=flow_shift
            )
            
            # Set up pipeline
            logger.info(f"Setting up T2V pipeline from {model_id}")
            self.t2v_pipe = WanPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=torch.bfloat16
            )
            self.t2v_pipe.scheduler = scheduler
            self.t2v_pipe.to(self.device)
            
            logger.info("T2V model loaded successfully")
            return self.t2v_pipe
        except Exception as e:
            logger.error(f"Error loading T2V model: {e}")
            logger.error(traceback.format_exc())
            raise

    def load_i2v_model(self, resolution="480p"):
        """Load Image-to-Video model"""
        logger.info(f"Loading I2V model (resolution={resolution})")
        try:
            # Clean up any previously loaded models
            if self.i2v_pipe is not None:
                del self.i2v_pipe
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
            
            # Select model based on resolution
            model_id = self.i2v_model_id_480p if resolution == "480p" else self.i2v_model_id_720p
            
            # Load image encoder
            logger.info(f"Loading image encoder from {model_id}")
            image_encoder = CLIPVisionModel.from_pretrained(
                model_id,
                subfolder="image_encoder",
                torch_dtype=torch.float32
            )
            
            # Load VAE
            logger.info(f"Loading VAE from {model_id}")
            vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch.float32
            )
            
            # Set up pipeline
            logger.info(f"Setting up I2V pipeline from {model_id}")
            self.i2v_pipe = WanImageToVideoPipeline.from_pretrained(
                model_id,
                vae=vae,
                image_encoder=image_encoder,
                torch_dtype=torch.bfloat16
            )
            
            # Enable CPU offloading to save memory
            logger.info("Enabling model CPU offloading")
            self.i2v_pipe.enable_model_cpu_offload()
            
            logger.info("I2V model loaded successfully")
            return self.i2v_pipe
        except Exception as e:
            logger.error(f"Error loading I2V model: {e}")
            logger.error(traceback.format_exc())
            raise

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Run generation pipeline based on input parameters"""
        logger.info(f"Starting prediction with inputs: {inputs}")
        try:
            # Get inputs
            task = inputs.get("task", "text-to-video")
            prompt = inputs.get("prompt", "Beautiful robot man walking, High Definition HD, High Detail, Cinematic.")
            negative_prompt = inputs.get("negative_prompt", "Low quality, blurry, pixelated, distorted, deformed, unrealistic")
            num_frames = int(inputs.get("num_frames", 81))
            fps = int(inputs.get("fps", 16))
            guidance_scale = float(inputs.get("guidance_scale", 5.0))
            seed = int(inputs.get("seed", 12))
            
            logger.info(f"Task: {task}, Frames: {num_frames}, FPS: {fps}, Seed: {seed}")
            logger.info(f"Prompt: {prompt}")
            
            # Set random seed for reproducibility
            generator = torch.Generator(device="cpu").manual_seed(seed)
            
            output_path = "outputs/output.mp4"
            
            # Text-to-Video generation
            if task == "text-to-video":
                logger.info("Running text-to-video generation")
                # Text-to-video specific parameters
                model_size = inputs.get("model_size", "small")
                width = int(inputs.get("width", 1280))
                height = int(inputs.get("height", 720))
                
                # Load the text-to-video model
                self.load_t2v_model(model_size=model_size, resolution="720p" if height >= 720 else "480p")
                
                # Make sure dimensions are multiples of 8
                width = (width // 8) * 8
                height = (height // 8) * 8
                logger.info(f"Generating video with dimensions {width}x{height}")
                
                # Generate the video
                output = self.t2v_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).frames[0]
                
                # Export the video
                logger.info(f"Exporting video to {output_path}")
                export_to_video(output, output_path, fps=fps)
            
            # Image-to-Video generation
            elif task == "image-to-video":
                logger.info("Running image-to-video generation")
                # Image-to-video specific parameters
                resolution = inputs.get("resolution", "480p")
                input_image_path = inputs.get("input_image")
                
                if input_image_path is None:
                    raise ValueError("Input image is required for image-to-video generation")
                
                logger.info(f"Input image: {input_image_path}")
                
                # Load the image-to-video model
                self.load_i2v_model(resolution=resolution)
                
                # Load the input image
                image = Image.open(input_image_path).convert("RGB")
                logger.info(f"Original image dimensions: {image.width}x{image.height}")
                
                # Calculate appropriate size for the input image
                max_area = 832 * 480 if resolution == "480p" else 832 * 720
                aspect_ratio = image.height / image.width
                
                # Get the VAE scale factor and patch size for proper adjustment
                mod_value = self.i2v_pipe.vae_scale_factor_spatial * self.i2v_pipe.transformer.config.patch_size[1]
                
                # Calculate height and width while maintaining aspect ratio
                height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
                
                # Resize the image
                image = image.resize((width, height))
                logger.info(f"Resized image dimensions: {width}x{height}")
                
                # Generate the video
                output = self.i2v_pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).frames[0]
                
                # Export the video
                logger.info(f"Exporting video to {output_path}")
                export_to_video(output, output_path, fps=fps)
            
            else:
                raise ValueError(f"Unknown task: {task}")
            
            logger.info("Prediction completed successfully")
            return {"output": output_path}
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(traceback.format_exc())
            raise