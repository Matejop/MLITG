import MLL
from typing import List

class InputValidation():
    def check_model_input(x: List[List[float]], y: List[List[int]], dimensions: List[int], activation: str, trials: int, epochs: int, batch_size: int, learning_rate: float) -> str:
        error_message = InputValidation.check_dimensions(len(x[0]), len(y[0]), dimensions) 
        if error_message != None:
            return error_message
        error_message = InputValidation.check_activation(activation)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(trials, "Trials", 0)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(epochs, "Epochs", 0)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(batch_size, "Batch size", 0)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_float(learning_rate, "Learning rate", 0)
        if error_message != None:
            return error_message 
        return None
    
    def check_trainer_input(activation: str, max_depth: int, trials: int, epochs: int, layer_size_start: int, layer_size_end: int, batch_size_start: int, batch_size_end: int, learning_rate_start: float, learning_rate_end: float) -> str:
        error_message = InputValidation.check_activation(activation)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(max_depth, "Max depth", 2, "GrxEq")
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(trials, "Trials", 0)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(epochs, "Epochs", 0)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(layer_size_start, "Layer size start", 0)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(layer_size_end, "Layer size end", layer_size_start, "GrxEq")
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(batch_size_start, "Batch size start", 0)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(batch_size_end, "Batch size end", batch_size_start, "GrxEq")
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_float(learning_rate_start, "Learning rate start", 0, "GrxEq")
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_float(learning_rate_end, "Learning rate end", learning_rate_start, "GrxEq")
        if error_message != None:
            return error_message
        return None
        
    def check_gridsearch_input(training_x: List[List[float]], training_y: List[List[float]], validation_x: List[List[float]], validation_y: List[List[float]], layer_size_step: int, batch_size_step: int, learning_rate_step: float) -> str:
        error_message = InputValidation.check_data_len(training_x, training_y)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_data_len(validation_x, validation_y)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_data_len(validation_x, training_x)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(layer_size_step, "Layer size step", -1)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(batch_size_step, "Batch size step", -1)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_float(learning_rate_step, "Learning rate step", 0, "GrxEq")
        if error_message != None:
            return error_message
        return None
        
    def check_rndsearch_input(training_x: List[List[float]], training_y: List[List[float]], validation_x: List[List[float]], validation_y: List[List[float]], iterations: int) -> str:
        error_message = InputValidation.check_data_len(training_x, training_y)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_data_len(validation_x, validation_y)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_data_len(validation_x, training_x)
        if error_message != None:
            return error_message
        error_message = InputValidation.check_boundary_int(iterations, "Iterations", 0)
        if error_message != None:
            return error_message
        return None
        
    def check_data_len(x: List[List[float]], y: List[List[float]]) -> str:
        if len(x) != len(y):
            return f"Inputed X\'s and Y\'s  have mismatching lengths. Length of X\'s: {len(x)} Length of Y\'s: {len(y)}"
        return None
    
    def check_dimensions(input_len: int, output_len: int, dimensions: List[int]) -> str:
        for i in range(len(dimensions)):
            if i == 0:
                error_message = InputValidation.check_boundary_int(dimensions[i], "Dimension", input_len, "Eq", i)
                if error_message != None:
                    return error_message
            elif i == len(dimensions) - 1:
                error_message = InputValidation.check_boundary_int(dimensions[i], "Dimension", output_len, "Eq", i)
                if error_message != None:
                    return error_message
            else:
                error_message = InputValidation.check_boundary_int(dimensions[i], "Dimension", 0, "Gr", i)
                if error_message != None:
                    return error_message    
        raise ""

    def check_activation(activation: str) -> str:
        found = False
        if activation != "all":
            for a in MLL.activation_functions:
                if a == activation:
                    found = True
                    break
            if found == False:
                error_message = f"Inputed activation is not supported. Inputed epochs: {activation}\nSupported activations are:"
                for a in MLL.activation_functions:
                    error_message += f" {a}"
                return error_message
        return ""    
    
    def check_boundary_int(input: int, parameter_name: str, boundary: int, condition: str = "Gr", index: int = None) -> str:
        if index == None:
            if type(input) != int:
                return f"{parameter_name} must be an int. Inputed {parameter_name}: {input}, Type: {type(input)}"
            return InputValidation.resolve_condition(input, parameter_name, boundary, condition, index)
        else:
            if type(input) != int:
                raise f"Inputed {parameter_name} at index {index} must be an int but it was not. Inputed {parameter_name}: {input}, Type at index {index}: {type(input)}"
            return InputValidation.resolve_condition(input, parameter_name, boundary, condition, index)
    
    def check_boundary_float(input: float, parameter_name: str, boundary: int, condition: str = "Gr", index: int = None) -> str:
        if index == None:
            if type(input) != float:
                return f"{parameter_name} must be a float. Inputed {parameter_name}: {input}, Type: {type(input)}"
            return InputValidation.resolve_condition(input, parameter_name, boundary, condition, index)
        else:
            if type(input) != float:
                return f"Inputed {parameter_name} at index {index} must be a float but it was not. Inputed {parameter_name}: {input}, Type at index {index}: {type(input)}"
            return InputValidation.resolve_condition(input, parameter_name, boundary, condition, index)

    def resolve_condition(input: float, parameter_name: str, boundary: int, condition: str = "Gr", index: int = None) -> str:
        if index == None:
            before_text = parameter_name
            after_text = f"{boundary}. Inputed {parameter_name}: {input}"
        else:
            before_text = f"Inputed {parameter_name} at index {index}"
            after_text = f"{boundary}. Inputed {parameter_name} at index {index}: {input}"        
        if condition == "Gr":
            if boundary >= input:
                return f"{before_text} must be greater than {after_text}"
        elif condition == "Sm":
            if input >= boundary:
                return f"{before_text} must be smaller than {after_text}"
        elif condition == "Eq":
            if input != boundary:
                return f"{before_text} must be equal to {after_text}"
        elif condition == "GrxEq":
            if input < boundary:
                return f"{before_text} must be greater than or equal to {after_text}"
        elif condition == "SmxEq":
            if input > boundary:
                return f"{before_text} must be smaller than or equal to {after_text}"
        return ""