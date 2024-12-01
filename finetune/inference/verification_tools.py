import random
import numpy as np
from typing import Callable, List, Tuple, Union
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError
from common import *

TOOL_TOKEN = "<|reserved_special_token_3|>"  # 128011


def load_tool(messages: List[dict], tool_code: str=None) -> Callable:
    tools = None
    
    def define_tool(tool_name: str):
        exec_globals = {
            'List': List
        }
        exec(tool_code, exec_globals)
        return exec_globals[tool_name]
    
    if tool_code:
        tool_name = [message["name"] for message in messages if message["role"] == "tool"][0]
        tools = [define_tool(tool_name)]
        
    return tools


def verify_transformation(
    source_code: str, 
    input_grids: Union[List[List[List[int]]], List[np.ndarray]] = None, 
    output_grids: Union[List[List[List[int]]], List[np.ndarray]] = None
) -> Tuple[str, List[bool], List[List[int]]]:
    """
    Verifies if a given Python implementation correctly transforms input grids to output grids.
    If input/output grids are not provided, retrieves them from the global context.
    
    Args:
        source_code: Python source code containing the transform() function implementation
        input_grids: List of 2D grids representing train/test inputs; provided by the system
        output_grids: List of 2D grids representing expected outputs; provided by the system
        
    Returns:
        A string describing whether the implementation was successful for each input grid
    """
    if input_grids is None or output_grids is None:
        return "Error: No problem context available"
        
    return _verify_transformation(source_code, input_grids, output_grids) 


def _verify_transformation(
    source_code: str, 
    input_grids: Union[List[List[List[int]]], List[np.ndarray]], 
    output_grids: Union[List[List[List[int]]], List[np.ndarray]]
) -> Tuple[str, List[bool], List[List[int]]]:
    """
    Internal implementation of the verification logic with timeout handling.
    """
    # Convert all grids to numpy arrays
    if isinstance(input_grids[0][0], list):
        input_grids = [np.array(grid) for grid in input_grids]
    if isinstance(output_grids[0][0], list):
        output_grids = [np.array(grid) for grid in output_grids]
    
    predicted_output_grids = []
    return_predicted_output_grids = []
    verdicts = []
    TIMEOUT = 8

    for input_grid, expected_grid in zip(input_grids, output_grids):
        try:
            with ProcessPool(max_workers=8) as pool:
                future = pool.schedule(execute_transform, 
                                    args=(input_grid, source_code),
                                    timeout=TIMEOUT)
                predicted_output_grid = future.result()
                predicted_output_grids.append(predicted_output_grid)
        except TimeoutError:
            predicted_output_grids.append("timeout")
        except ProcessExpired as error:
            predicted_output_grids.append("process_expired")
        except Exception as error:
            predicted_output_grids.append(None)
    
    for i, (predicted_output_grid, expected_grid) in enumerate(zip(predicted_output_grids, output_grids)):
        comparison = compare_grids(predicted_output_grid, expected_grid)
        verdicts.append(comparison)
        if isinstance(predicted_output_grid, np.ndarray):
            return_predicted_output_grids.append(predicted_output_grid.astype(int).tolist())
        else:
            return_predicted_output_grids.append(predicted_output_grid)
            
    result_str = format_verification_results(verdicts)
    return result_str, verdicts, return_predicted_output_grids

# # Use os.path.join to create path relative to script location
# with open("common.py", "r") as f:
#     COMMON_LIBRARY_CODE = f.read()


def execute_transform(input_grid, source_code):
    # Create execution environment with necessary imports and setup
    exec_globals = {
        'np': np,
        'random': random,
        'input_grid': input_grid
    }
    
    # Set random seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    
    # Execute the source code
    exec(source_code, exec_globals)
    
    # Get the transformed output
    if 'transform' not in exec_globals:
        return None
        
    predicted_output = exec_globals['transform'](input_grid)
    return predicted_output


def compare_grids(output_grid, expected_grid):
    if isinstance(output_grid, str):
        return None
        
    if not isinstance(output_grid, np.ndarray):  # type mismatch
        return None
        
    if len(output_grid.shape) != 2:  # non-2d array
        return None
        
    if np.array_equal(output_grid, expected_grid):
        return True
    
    if output_grid.shape != expected_grid.shape:  # shape mismatch
        return False
        
    # If shapes match but content doesn't
    return False


def format_verification_results(verdicts: List[bool]) -> str:
    """
    Formats the verification results into a string.
    """
    result_messages = ["Result of the transform() function for every case in the reference and test examples:"]
    for i, result in enumerate(verdicts, 1):
        if result is True:
            result_messages.append(f"Case {i}: Correct output")
        elif result is False:
            result_messages.append(f"Case {i}: Incorrect output")
        else:  # result is None
            result_messages.append(f"Case {i}: Error during execution")
    return "\n".join(result_messages)


def verify_transformation_old(input_grids: List[List[List[int]]], output_grids: List[List[List[int]]], source_code: str) -> str:
    """
    Verifies if a given Python implementation correctly transforms input grids to output grids.
    
    Args:
        input_grids: List of 2D grids representing train or test input grids
        output_grids: List of 2D grids representing expected output grids
        source_code: Python source code containing the transform() function implementation
        
    Returns:
        A string describing whether the implementation was successful for each input grid in 
    """
    return _verify_transformation(input_grids, output_grids, source_code)

'''
{
  "type": "function",
  "function": {
    "name": "verify_transformation",
    "description": "Verifies if a given Python implementation correctly transforms input grids to output grids.",
    "parameters": {
      "type": "object",
      "properties": {
        "input_grids": {
          "type": "array",
          "description": "List of 2D grids representing train or test input grids"
        },
        "output_grids": {
          "type": "array", 
          "description": "List of 2D grids representing expected output grids"
        },
        "source_code": {
          "type": "string",
          "description": "Python source code containing the transform() function implementation"
        }
      },
      "required": ["input_grids", "output_grids", "source_code"]
    }
  }
}
'''