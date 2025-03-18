import triton
import triton.language as tl

@triton.jit
def nf4_dequantize_kernel(
    weight_ptr,
    quant_absmax_ptr,
    state2_code_ptr,
    state2_absmax_ptr,
    quant_code_ptr,
    output_ptr,
    num_blocks,
    output_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # 64 for weight blocks
    ABSMAX_BLOCK_SIZE: tl.constexpr,  # 256 for state2 blocks
):
    pid = tl.program_id(0)
    if pid >= num_blocks:
        return

    # Load the quantized absmax value for this block
    absmax_idx = pid
    code_idx = tl.load(quant_absmax_ptr + absmax_idx)
    state2_block_idx = absmax_idx // ABSMAX_BLOCK_SIZE
    state2_absmax = tl.load(state2_absmax_ptr + state2_block_idx)
    state2_code_val = tl.load(state2_code_ptr + code_idx)
    dequant_absmax = state2_code_val * state2_absmax

    # Load 32 bytes (64 4-bit weights)
    weight_offset = pid * 32
    weights = tl.load(weight_ptr + weight_offset + tl.arange(0, 32), mask=None)

    # Split each byte into two 4-bit indices
    high = (weights >> 4).to(tl.uint8)
    low = (weights & 0x0F).to(tl.uint8)
    indices = tl.zeros(64, dtype=tl.uint8)
    for i in tl.static_range(32):
        indices = tl.where(tl.arange(0, 64) == 2*i, high[i], indices)
        indices = tl.where(tl.arange(0, 64) == 2*i + 1, low[i], indices)

    # Lookup quant code and dequantize
    code_vals = tl.load(quant_code_ptr + indices)
    dequant_weights = code_vals * dequant_absmax

    # Cast to output dtype
    if output_dtype == tl.float16:
        dequant_weights = dequant_weights.to(tl.float16)
    elif output_dtype == tl.bfloat16:
        dequant_weights = dequant_weights.to(tl.bfloat16)

    # Store the result
    output_offset = pid * BLOCK_SIZE
    tl.store(output_ptr + output_offset + tl.arange(0, BLOCK_SIZE), dequant_weights, mask=None)

def triton_dequantize(weight):
    quant_state = weight.weight.quant_state
    state2 = quant_state.state2

    # Determine output dtype
    output_dtype = quant_state.dtype
    if output_dtype == torch.float16:
        tl_dtype = tl.float16
    elif output_dtype == torch.bfloat16:
        tl_dtype = tl.bfloat16
    else:
        raise ValueError("Unsupported output dtype")

    quantized_weight = weight.weight
    num_elements = quantized_weight.numel() * 2  # 4 bits per element, packed as uint8
    num_blocks = num_elements // 64  # Each block is 64 elements

    # Ensure num_blocks is correct
    assert num_blocks * 64 == num_elements, "Number of elements must be divisible by block size"

    # Output tensor
    output = torch.empty(num_elements, device='cuda', dtype=output_dtype)

    # Kernel launch parameters
    grid = (num_blocks,)
    nf4_dequantize_kernel[grid](
        quantized_weight.data_ptr(),
        quant_state.absmax.data_ptr(),
        state2.code.data_ptr(),
        state2.absmax.data_ptr(),
        quant_state.code.data_ptr(),
        output.data_ptr(),
        num_blocks,
        output_dtype=tl_dtype,
        BLOCK_SIZE=64,
        ABSMAX_BLOCK_SIZE=256,
    )

    # Reshape to original shape (out_features, in_features)
    original_shape = (quantized_weight.size(0), quantized_weight.size(1) * 2)
    return output.view(original_shape)