from typing import List, Union, Dict, Any, Optional
from PIL import Image
from vllm import LLM, SamplingParams

class Qwen2VL:
    def __init__(
        self,
        model_path: str = "",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.99,
        enforce_eager: bool = False,
        max_model_len: int = 8192,
        limit_mm_per_prompt: Dict[str, int] = {"image": 1, "video": 0},
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
    ):
        """
        Initialize Qwen2VL/Qwen2.5VL inference class
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of tensor parallel processes
            gpu_memory_utilization: GPU memory utilization ratio
            enforce_eager: Whether to enforce eager execution
            max_model_len: Maximum sequence length
            limit_mm_per_prompt: Limit multimodal inputs per prompt
            min_pixels: Minimum number of pixels for image processing
            max_pixels: Maximum number of pixels for image processing
            max_num_seqs: Maximum number of sequences
        """
        kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "max_model_len": max_model_len,
            "limit_mm_per_prompt": limit_mm_per_prompt,
        }
        
        if max_num_seqs:
            kwargs["max_num_seqs"] = max_num_seqs
            
        if min_pixels or max_pixels:
            kwargs["mm_processor_kwargs"] = {
                "min_pixels": min_pixels if min_pixels else 4 * 28 * 28,
                "max_pixels": max_pixels if max_pixels else 16384 * 28 * 28,
            }

        self.llm = LLM(**kwargs)
        self.system_prompt = "You are a helpful assistant."
        
        
    def _format_prompt(self, messages: List[Dict[str, Any]], images: List[Image.Image]) -> str:
        """
        Format conversation history and handle custom image positions
        
        Args:
            messages: Conversation history
            images: List of images
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        if messages[0]["role"] != "system":
            prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        
        image_idx = 0
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role in ["user", "human"]:
                # Count required images
                required_images = content.count("<image>")
                if image_idx + required_images > len(images):
                    raise ValueError(f"Not enough images provided. Required: {image_idx + required_images}, Provided: {len(images)}")
                
                # Replace all <image> tokens
                for _ in range(required_images):
                    content = content.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>", 1)
                    image_idx += 1
                    
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role in ["assistant", "gpt"]:
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                
        prompt += "<|im_start|>assistant\n"
        
        # Verify all images are used
        if image_idx < len(images):
            raise ValueError(f"Too many images provided. Used: {image_idx}, Provided: {len(images)}")
            
        return prompt

    def chat(
        self,
        messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        images: Optional[Union[Image.Image, List[Image.Image], List[List[Image.Image]]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Chat interface supporting single or batch conversations
        
        Args:
            messages: Single conversation history or list of conversation histories
            images: Single image, list of images, or batch list of images
            temperature: Sampling temperature
            max_tokens: Maximum generation length
            top_p: Top-p sampling parameter
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated response or list of responses
        """
        # Check if this is a batch request
        is_batch = isinstance(messages[0], list)
        if not is_batch:
            messages = [messages]
            images = [images] if isinstance(images, Image.Image) else ([images] if images is not None else [None])
            
        # Prepare inputs
        inputs = []
        assert len(messages) == len(images)
        for msg, img_list in zip(messages, images):
            if img_list is not None:
                img_list = [img_list] if isinstance(img_list, Image.Image) else img_list
                prompt = self._format_prompt(msg, img_list)
                
                # Build multimodal data dictionary
                input_data = {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img_list
                    }
                }
            else:
                prompt = self._format_prompt(msg, [])
                input_data = {
                    "prompt": prompt,
                    "multi_modal_data": {}
                }
                
            inputs.append(input_data)
            
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Execute inference
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        
        return responses[0] if not is_batch else responses
