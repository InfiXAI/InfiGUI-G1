import json
from typing import Dict, List, Any, Tuple, Optional


def extract_and_parse_json(input_string: str, wrapper: str) -> Optional[List]:
    """
    Attempt to extract and parse a JSON array from a string using a given pair of wrapper characters.
    
    The function searches for the first occurrence of the start wrapper and the last occurrence
    of the end wrapper, and tries to parse the substring between them as JSON.
    """
    if len(wrapper) != 2:
        raise ValueError("Wrapper must be exactly two characters long")

    start_char, end_char = wrapper
    start_index = input_string.find(start_char)
    end_index = input_string.rfind(end_char)

    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None

    json_string = input_string[start_index:end_index + 1]

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None


def point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
    """
    Ray casting algorithm to determine if a point lies inside a polygon.
    
    Args:
        point: Point coordinates as [x, y].
        polygon: List of polygon vertices [[x1, y1], [x2, y2], ...].
    
    Returns:
        True if the point is inside the polygon; False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def is_point_inside_element(element: Dict, point: List[float]) -> bool:
    """
    Check whether a predicted point lies inside a ground-truth region.
    
    Args:
        element: Dictionary describing the region, may contain a bbox or a polygon.
        point: Predicted point coordinates [x, y].
    
    Returns:
        True if the point is inside the element area; False otherwise.
    """
    # Axis-aligned bounding box
    if "x1" in element and "y1" in element and "x2" in element and "y2" in element:
        return (element["x1"] <= point[0] <= element["x2"] and 
                element["y1"] <= point[1] <= element["y2"])
    
    # Polygon
    elif "polygon" in element:
        polygon = element["polygon"]
        if len(polygon) < 3:  # A polygon requires at least 3 vertices
            return False
        return point_in_polygon(point, polygon)
    
    # Unsupported format
    else:
        return False


class InfiguiG1Prompt:
    """
    Default prompt processor for infigui-g1.
    """
    
    @staticmethod
    def get_metric_keys() -> Dict[str, str]:
        """
        Return the metric keys supported by this processor and their types.
        
        Returns:
            Mapping from metric key to type, where type is one of:
            - 'sum': metrics to be summed
            - 'avg': metrics to be averaged (sum and count are handled separately)
            - 'count': count metrics
        """
        return {
            "total": "sum",
            "correct": "sum", 
            "has_correct": "sum",
            "num_answers": "avg",  # This is averaged across samples
        }
    
    @staticmethod
    def get_accuracy_pairs() -> List[Tuple[str, str]]:
        """
        Return numerator/denominator pairs for accuracy calculations.
        
        Returns:
            List of (numerator, denominator) pairs.
        """
        return [
            ("correct", "total"),       # correct_accuracy = correct / total
            ("has_correct", "total"),   # has_correct_accuracy = has_correct / total
        ]
    
    @staticmethod
    def generate_prompt(sample: Dict, image_width: int, image_height: int, think_mode: bool = True) -> List[Dict]:
        """
        Construct the message prompt for a single sample.
        """
        if think_mode:
            system_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags."
        else:
            system_prompt = "You are a helpful assistant."
            
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f'''<image>The screen's resolution is {image_width}x{image_height}.
Locate the UI element(s) for "{sample['instruction']}", output the coordinates using JSON format: [{{"point_2d": [x, y]}}, ...]'''
            }
        ]
        return messages
    
    @staticmethod
    def extract_coordinates(response: str, think_mode: bool = False) -> Optional[List]:
        """
        Extract coordinates from the model response.
        """
        if think_mode and "</think>" in response:
            response = response.split("</think>")[-1]
        
        result = extract_and_parse_json(response, "[]")
        return result
    
    @staticmethod
    def calculate_metrics(sample: Dict, predictions: Optional[List], gt_bbox: Dict) -> Dict[str, Any]:
        """
        Compute evaluation metrics for a single sample.
        
        Args:
            sample: Sample dictionary.
            predictions: List of predictions, format [{"point_2d": [x, y]}, ...].
            gt_bbox: Ground-truth bounding box, format {"x1": x1, "y1": y1, "x2": x2, "y2": y2} or {"polygon": [[x,y], ...]}.
        
        Returns:
            Dictionary containing metric values.
        """
        # Initialize metric dictionary dynamically based on metric_keys
        metric_keys = InfiguiG1Prompt.get_metric_keys()
        metrics = {}
        
        # Initialize base metrics
        for key, key_type in metric_keys.items():
            if key == "total":
                metrics[key] = 1  # Each sample counts as 1
            elif key_type in ["sum", "count"]:
                metrics[key] = 0
            elif key_type == "avg":
                metrics[key] = None
        
        # Add fixed field
        metrics["predictions"] = None
        
        # Handle samples without ground-truth (empty bbox)
        if not gt_bbox:
            if predictions is None or len(predictions) == 0:
                metrics["correct"] = 1
                metrics["has_correct"] = 1
            else:
                metrics["correct"] = 0
                metrics["has_correct"] = 0
            # Accuracies are computed later; no need to set here
            return metrics
        
        # No predictions
        if predictions is None or len(predictions) == 0:
            return metrics
        
        try:
            # Check if any predicted point is correct
            has_correct = 0
            for pred in predictions:
                point = pred["point_2d"]
                if is_point_inside_element(gt_bbox, point):
                    has_correct = 1
                    break
            
            metrics["has_correct"] = has_correct
            
            # Check the first prediction
            first_pred = predictions[0]["point_2d"]
            if is_point_inside_element(gt_bbox, first_pred):
                metrics["correct"] = 1
            
            metrics["predictions"] = first_pred
            metrics["num_answers"] = sum(1 for pred in predictions if "point_2d" in pred)
            
        except (KeyError, IndexError, TypeError) as e:
            # Parsing error; return defaults in metrics
            pass
        
        # Accuracies are computed later; no need to set here
        return metrics


# Register available prompt processors
PROMPT_PROCESSORS = {
    "infigui-g1": InfiguiG1Prompt
}


def get_prompt_processor(name: str):
    """
    Retrieve a prompt processor by name.
    """
    if name not in PROMPT_PROCESSORS:
        raise ValueError(f"Unknown prompt processor: {name}. Available: {list(PROMPT_PROCESSORS.keys())}")
    return PROMPT_PROCESSORS[name] 