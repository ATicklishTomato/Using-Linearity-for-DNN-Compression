

def compression_ratio(original_param_count, compressed_param_count):
    """
    Compute the compression ratio
    Args:
        original_param_count:   Number of parameters in the original model.
        compressed_param_count:   Number of parameters in the compressed model.
    Returns:
        compression_ratio:   Compression ratio
    """
    if original_param_count == 0:
        return 0.0
    return compressed_param_count / original_param_count


def accuracy_loss(original_accuracy, compressed_accuracy):
    """
    Compute the accuracy loss
    Args:
        original_accuracy:   Accuracy of the original model.
        compressed_accuracy: Accuracy of the compressed model.
    Returns:
        accuracy_loss:   Accuracy loss
    """
    return original_accuracy - compressed_accuracy

def speedup(original_inference_time, compressed_inference_time):
    """
    Compute the speedup
    Args:
        original_inference_time:   Inference time of the original model.
        compressed_inference_time: Inference time of the compressed model.
    Returns:
        speedup:   Speedup
    """
    if compressed_inference_time == 0:
        return float('inf')
    return original_inference_time / compressed_inference_time

def tflop_reduction(original_tflop, compressed_tflop):
    """
    Compute the TFLOP reduction
    Args:
        original_tflop:   TFLOP of the original model.
        compressed_tflop: TFLOP of the compressed model.
    Returns:
        tflop_reduction:  TFLOP reduction
    """
    if original_tflop == 0:
        return 0.0
    return (original_tflop - compressed_tflop) / original_tflop