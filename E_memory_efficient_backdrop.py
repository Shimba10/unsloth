import torch
import torch.nn.functional as F

def transformation_function(batch, linear, labels):
    x = linear(batch).float()  # Up projection to large space
    from torch.nn import CrossEntropyLoss
    down_projection_function = CrossEntropyLoss(reduction="mean")
    # Down projection to small space
    loss = down_projection_function(x.view(-1, x.shape[-1]), labels.view(-1))
    return loss

class MemoryEfficientLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, linear, labels, forward_function):
        # Split X into batches along the batch dimension (assuming batch_first=True)
        # For simplicity, split into two batches
        batch_size = X.size(0)
        split_size = batch_size // 2
        if split_size == 0:
            split_size = batch_size
        X_batches = torch.split(X, split_size)
        labels_batches = torch.split(labels, split_size)
        
        # Save necessary tensors and parameters for backward
        ctx.save_for_backward(X, linear.weight, labels)
        ctx.linear = linear  # Save linear to access bias
        ctx.X_batches = X_batches  # Save split batches to avoid re-splitting in backward
        ctx.labels_batches = labels_batches
        ctx.forward_function = forward_function
        
        # Compute total loss by summing individual batch losses
        total_loss = 0
        for x_batch, lbl_batch in zip(X_batches, labels_batches):
            total_loss += forward_function(x_batch, linear, lbl_batch)
        return total_loss

    @staticmethod
    def backward(ctx, grad_output):
        X, weight, labels = ctx.saved_tensors
        linear = ctx.linear
        X_batches = ctx.X_batches
        labels_batches = ctx.labels_batches
        forward_function = ctx.forward_function
        
        # Initialize gradients
        dX = torch.zeros_like(X)
        dW = torch.zeros_like(weight)
        db = torch.zeros_like(linear.bias) if linear.bias is not None else None
        
        for i, (x_batch, lbl_batch) in enumerate(zip(X_batches, labels_batches)):
            # Recompute logits
            x_batch = x_batch.detach().requires_grad_(True)
            logits = torch.matmul(x_batch, weight.t())
            if linear.bias is not None:
                logits += linear.bias
            
            # Compute loss for this batch
            loss = forward_function(x_batch, linear, lbl_batch)
            
            # Compute gradients for this batch's logits
            dY_i = torch.autograd.grad(loss, logits, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]
            
            # Gradient for X
            dX_batch = torch.matmul(dY_i, weight)
            dX[i*x_batch.size(0): (i+1)*x_batch.size(0)] = dX_batch
            
            # Gradient for weight
            dW += torch.matmul(x_batch.reshape(-1, x_batch.size(-1)).transpose(0, 1), dY_i.reshape(-1, dY_i.size(-1)))
            
            # Gradient for bias
            if linear.bias is not None:
                db += dY_i.reshape(-1, dY_i.size(-1)).sum(0)
        
        # Return gradients for X, None for linear (handled by autograd), None, None
        # Note: Gradients for linear.weight and linear.bias are manually computed but need to be assigned
        # This is a workaround as the function can't directly return gradients for module parameters
        return dX, None, None, None