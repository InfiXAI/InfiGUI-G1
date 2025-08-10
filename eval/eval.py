import argparse
import os
import json
import random
from datetime import datetime
from typing import Dict
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import smart_resize

# Import custom modules
from models.qwen2vl import Qwen2VL
from data import (
    load_benchmark_info, 
    load_dataset, 
    standardize_sample,
    resize_image,
    calculate_hierarchical_statistics,
    print_hierarchical_stats,
    reorder_stats_for_output
)
from prompts import get_prompt_processor


def prepare_sample_data(sample: Dict, benchmark_info: Dict, max_image_pixels: int, prompt_processor, think_mode: bool) -> tuple:
    """
    Prepares data for a single sample.
    """
    # Load image
    image_path = os.path.join(benchmark_info["image_root"], sample["images"])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path)
    original_width, original_height = image.size
    
    # Process bounding box coordinates
    bbox = sample["bbox"].copy()
    
    # Resize image
    new_width, new_height = resize_image(original_width, original_height, max_image_pixels)
    
    # Further resize using smart_resize
    new_height, new_width = smart_resize(new_height, new_width, max_pixels=12845056)
    image = image.resize((new_width, new_height))
    
    # Update bounding box coordinates to the new image dimensions
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    if all(key in bbox.keys() for key in ["x1", "y1", "x2", "y2"]):
        bbox["x1"] = bbox["x1"] * scale_x
        bbox["y1"] = bbox["y1"] * scale_y
        bbox["x2"] = bbox["x2"] * scale_x
        bbox["y2"] = bbox["y2"] * scale_y
    elif all(key in bbox.keys() for key in ["polygon"]):
        bbox["polygon"] = [[p[0] * scale_x, p[1] * scale_y] for p in bbox["polygon"]]
    
    # Generate prompt
    messages = prompt_processor.generate_prompt(sample, new_width, new_height, think_mode)
    
    return image, messages, bbox


def process_response(sample: Dict, response: str, bbox: Dict, prompt_processor, think_mode: bool) -> Dict:
    """
    Processes the model's response.
    """
    # Extract coordinates
    predictions = prompt_processor.extract_coordinates(response, think_mode)
    
    # Calculate metrics
    metrics = prompt_processor.calculate_metrics(sample, predictions, bbox)
    
    # Construct result
    result = {
        "image": sample["images"],
        "instruction": sample["instruction"],
        "gt_bbox": bbox,
        "response": response,
        **metrics
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="General UI Element Localization Evaluation Framework")
    
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("--benchmark", "-b", type=str, default="screenspot-pro", help="Name of the benchmark to evaluate")
    parser.add_argument("--prompt", type=str, default="infigui-g1", help="Name of the prompt processor to use")
    parser.add_argument("--tensor-parallel", "-tp", type=int, default=4, help="Tensor parallelism size")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--max-num-seqs", type=int, default=16, help="Maximum number of sequences in a batch")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--max-image-tokens", "-mit", type=int, default=5600, help="Maximum image tokens")
    parser.add_argument("--think-mode", type=int, default=1, help="Whether to enable thinking mode (1=enable, 0=disable)")
    parser.add_argument("--debug-mode", type=int, default=0, help="Whether to enable debug mode (1=enable, 0=disable)")
    parser.add_argument("--model-name", type=str, default=None, help="Output model name, extracted from model path if not specified")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="Generation temperature")
    
    args = parser.parse_args()
    
    # Convert flag arguments to boolean
    think_mode = bool(args.think_mode)
    debug_mode = bool(args.debug_mode)
    
    print(f"Starting evaluation - Benchmark: {args.benchmark}, Prompt: {args.prompt}")
    
    # Load benchmark information
    benchmark_info = load_benchmark_info(args.benchmark)
    print(f"Loaded benchmark: {benchmark_info['name']}")

    
    # Get prompt processor
    prompt_processor = get_prompt_processor(args.prompt)
    print(f"Using prompt processor: {args.prompt}")

    
    # Set output directory
    model_name = args.model_name if args.model_name else os.path.basename(os.path.normpath(args.model_path))
    output_dir = f"./output/{model_name}/{args.benchmark}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    max_image_pixels = args.max_image_tokens * 28 * 28 if args.max_image_tokens > 0 else 16384 * 28 * 28
    print(f"Initializing model, max image pixels: {max_image_pixels}")
    
    llm = Qwen2VL(
        model_path=args.model_path,
        max_model_len=max_image_pixels//28//28 + 1024,
        tensor_parallel_size=args.tensor_parallel,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
    )
    
    # Load dataset
    print("Loading dataset...")
    raw_dataset = load_dataset(benchmark_info)
    print(f"Raw dataset size: {len(raw_dataset)}")

    
    # Standardize dataset
    dataset = []
    for sample in raw_dataset:
        standardized = standardize_sample(sample, args.benchmark)
        dataset.append(standardized)
    
    # Random sampling in debug mode
    random.seed(42)
    if debug_mode:
        dataset = random.sample(dataset, len(dataset)//5)
        print(f"Debug mode enabled, using {len(dataset)} samples.")
    
    # Prepare all data
    print("Preparing data...")
    messages_list = []
    images = []
    processed_bboxes = []
    
    for i, sample in enumerate(tqdm(dataset, desc="Preparing samples")):
        # Prepare sample data
        image, messages, bbox = prepare_sample_data(
            sample, benchmark_info, max_image_pixels, prompt_processor, think_mode
        )
        messages_list.append(messages)
        images.append(image)
        processed_bboxes.append(bbox)
        
        # Update bbox information in the sample (for subsequent processing)
        dataset[i]["processed_bbox"] = bbox
    
    # Batch generate responses
    print("Generating responses...")
    responses = []
    total_samples = len(messages_list)
    progress_bar = tqdm(total=total_samples, desc="Generating responses")
    
    for i in range(0, len(messages_list), args.batch_size):
        batch_messages = messages_list[i:i+args.batch_size] 
        batch_images = images[i:i+args.batch_size]
        
        # Generate responses
        batch_responses = llm.chat(
            batch_messages, 
            batch_images, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature
        )
        responses.extend(batch_responses)
        
        progress_bar.update(len(batch_messages))
    
    progress_bar.close()
    
    # Process response results
    print("Processing response results...")
    results = {}
    
    for idx, (sample, response) in enumerate(tqdm(zip(dataset, responses), total=len(dataset), desc="Processing responses")):
        bbox = sample["processed_bbox"]
        # print(response)
        try:
            result = process_response(sample, response, bbox, prompt_processor, think_mode)
            results[idx] = result
        except Exception as e:
            print(f"Failed to process sample {idx}: {e}")
            # Add default result using prompt processor's metric definition
            default_metrics = prompt_processor.calculate_metrics(sample, None, bbox)
            results[idx] = {
                "image": sample.get("images", ""),
                "instruction": sample.get("instruction", ""),
                "gt_bbox": bbox,
                "response": response,
                **default_metrics
            }
    
    # Calculate hierarchical statistics
    print("Calculating statistics...")
    statistics = calculate_hierarchical_statistics(results, dataset, prompt_processor)
    
    # Print statistics
    print("\n=== Evaluation Results ===")
    metric_keys = prompt_processor.get_metric_keys()
    accuracy_pairs = prompt_processor.get_accuracy_pairs()
    print_hierarchical_stats(statistics, 0, metric_keys, accuracy_pairs)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Reorder statistics, moving subgroups to the end
    ordered_statistics = reorder_stats_for_output(statistics)
    
    output_data = {
        "benchmark": args.benchmark,
        "prompt": args.prompt,
        "model_path": args.model_path,
        "args": vars(args),
        "statistics": ordered_statistics,
        "detailed_results": results
    }
    
    output_file = os.path.join(
        output_dir, 
        f"{timestamp}{'_t'+str(args.temperature).replace('.', '-') if args.temperature else ''}"
        f"{'_debug' if debug_mode else ''}_{args.prompt}.json"
    )
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation results saved to: {output_file}")
    print_dict = {}
    for k, v in ordered_statistics.items():
        if k.endswith('_accuracy'):
            print_dict[k] = v
    print(f"Overall accuracy: {print_dict}")


if __name__ == "__main__":
    main()
