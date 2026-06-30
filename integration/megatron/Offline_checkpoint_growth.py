import copy
import io
import re
from collections import defaultdict

import torch

from megatron.training import ft_integration, one_logger_utils
from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import get_args, get_one_logger, get_timers
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import print_rank_0, update_use_dist_ckpt

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP  # noqa: F401
    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False


def add_growth_args(parser):
    parser.add_argument("--growth-verbose", action="store_true", help="Enable verbose logging for growth.")
    parser.add_argument("--growth-stack-method", type=str, default="interleaved", choices=["interleaved", "stacked"],
                        help="Method to stack original and growth layers. 'interleaved' means original and growth layers are interleaved, 'stacked' means original layers are stacked first, then growth layers.")
    parser.add_argument("--growth-weight-multiplier", type=float, default=1.0,
                        help="Multiplier for the weights of the growth layers. This can be used to control the initial scale of the growth layers.")
    parser.add_argument("--growth-ignore-first-num-layers", type=int, default=0,
                        help="Number of initial layers to ignore during growth. This can be useful if the first few layers are not suitable for growth.")
    parser.add_argument("--growth-ignore-last-num-layers", type=int, default=0,
                        help="Number of final layers to ignore during growth. This can be useful if the last few layers are not suitable for growth.")
    parser.add_argument("--growth-zerofy-output-init", action="store_true",
                        help="If set, the output initialization of attention output and MLP output will be set to zero. This can help in resulting identity mapping for the new layers.")
    parser.add_argument("--do-depth-growth", action="store_true",
                        help="If set, the model will be grown in depth-wise manner.")
    parser.add_argument("--do-moe-width-growth", action="store_true",
                        help="If set, the model's moe part will be grown in width-wise manner.")
    parser.add_argument("--use-ckpt-merge", action="store_true",
                        help="If set, the model will use checkpoint merging.")
    parser.add_argument("--second-ckpt-step", type=int, default=None,
                        help="Only set when using ckpt merging, specify the second ckpt to load")
    parser.add_argument("--growth-use-random-router", action="store_true",
                        help="If set, the model will use a random router for the growth layers.")
    parser.add_argument("--growth-use-interleaved-moe-cat", action="store_true",
                        help="If set, the model will use interleaved concatenation for the moe experts and router.")
    parser.add_argument("--growth-zerofy-expert-bias", action="store_true",
                        help="If set, the expert bias will be zeroed out.")
    parser.add_argument("--growth-add-expert-noise", action="store_true",
                        help="If set, add small noise to the new experts to break symmetry.")
    parser.add_argument("--growth-expert-noise-std-scaling-factor", type=float, default=0.01,
                        help="Scaling factor for the standard deviation of the noise added to new experts when --growth-add-expert-noise is set.")
    return parser


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0,
                              checkpointing_context=None):
    """Setup model and optimizer."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()
    #! when in cpu, just use this:
    model = model_provider_func()
    if not isinstance(model, list):
        model = [model]

    optimizer, opt_param_scheduler = None, None

    if (args.load is not None or args.pretrained_checkpoint is not None) and not args.moe_use_upcycling:
        one_logger and one_logger.log_metrics({
            'load_checkpoint_start_time': one_logger_utils.get_timestamp_in_ms()
        })
        timers('load-checkpoint', log_level=0).start(barrier=True)

        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                model, optimizer, opt_param_scheduler, checkpointing_context=checkpointing_context,
                skip_load_to_model_and_opt=HAVE_FSDP2 and getattr(args, "use_torch_fsdp2", False) and args.ckpt_format == "torch_dist")
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
        one_logger and one_logger.log_metrics({
            'load_checkpoint_finish_time': one_logger_utils.get_timestamp_in_ms(),
            'load_checkpoint_time': timers('load-checkpoint').active_time()
        })
    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0


    if args.use_ckpt_merge:
        print_rank_0(">>> Load the second ckpt since --use-ckpt-merge is on")
        assert args.second_ckpt_step is not None, "When using --use-ckpt-merge, --second-ckpt-step must be specified."
        
        #! some manual setting to do second ckpt loading
        args.ckpt_step = args.second_ckpt_step
        args.num_floating_point_operations_so_far = 0
        args.consumed_train_samples = 0
        args.skipped_train_samples = 0
        args.consumed_valid_samples = 0
        
        model2 = model_provider_func()
        if not isinstance(model2, list):
            model2 = [model2]
        
        optimizer, opt_param_scheduler = None, None
        if (args.load is not None or args.pretrained_checkpoint is not None) and not args.moe_use_upcycling:
            one_logger and one_logger.log_metrics({
                'load_checkpoint_start_time': one_logger_utils.get_timestamp_in_ms()
            })
            timers('load-checkpoint', log_level=0).start(barrier=True)

            args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                    model2, optimizer, opt_param_scheduler, checkpointing_context=checkpointing_context,
                    skip_load_to_model_and_opt=HAVE_FSDP2 and getattr(args, "use_torch_fsdp2", False) and args.ckpt_format == "torch_dist")
            timers('load-checkpoint').stop(barrier=True)
            timers.log(['load-checkpoint'])
            one_logger and one_logger.log_metrics({
                'load_checkpoint_finish_time': one_logger_utils.get_timestamp_in_ms(),
                'load_checkpoint_time': timers('load-checkpoint').active_time()
            })
        else:
            args.iteration = 0
            args.num_floating_point_operations_so_far = 0
    else:
        model2 = None
            
    # Convert checkpoint format.
    if args.ckpt_convert_format is not None:
        load_ckpt_format = args.ckpt_format
        args.ckpt_format = args.ckpt_convert_format
        update_use_dist_ckpt(args)

        if not args.do_depth_growth and not args.do_moe_width_growth:
            raise ValueError("At least one of --do-depth-growth or --do-moe-width-growth must be set to True for correct model growth.")
        
        if args.do_depth_growth:
            model = model_depth_growth(model, args, model_provider_func, verbose=args.growth_verbose, model2=model2)
        if args.do_moe_width_growth:
            model = model_moe_width_growth(model, args, model_provider_func, verbose=args.growth_verbose, model2=model2)

        save_checkpoint(args.iteration, model, optimizer, opt_param_scheduler,
                        args.num_floating_point_operations_so_far,
                        preprocess_common_state_dict_fn=preprocess_common_state_dict)

        print_rank_0("> converted checkpoint: %s -> %s." % (load_ckpt_format, args.ckpt_format))
        ft_integration.on_checkpointing_start()
        maybe_finalize_async_save(blocking=True, terminate=True)
        ft_integration.on_checkpointing_end(is_async_finalization=True)
        ft_integration.shutdown()
        torch.distributed.barrier()
        exit()

    return model, optimizer, opt_param_scheduler


def preprocess_common_state_dict(common_state_dict):
    # Convert args key of type namespace to dictionary
    preprocessed_common_state_dict = copy.deepcopy(common_state_dict)
    preprocessed_common_state_dict['args'] = vars(preprocessed_common_state_dict['args'])
    # Remove rank and local rank from state dict if it exists, since they are expected to be different
    preprocessed_common_state_dict['args'].pop('local_rank', None)
    preprocessed_common_state_dict['args'].pop('rank', None)
    return preprocessed_common_state_dict


#! load value func
def identify_output_layer(key):
    """Identify if the key corresponds to an output layer."""
    if_attn_output = key.endswith("self_attention.linear_proj.weight") or key.endswith("self_attention.linear_proj.bias")
    if_mlp_output = key.endswith("mlp.linear_fc2.weight") or key.endswith("mlp.linear_fc2.bias")
    if_shard_moe_output = key.endswith("mlp.shared_experts.linear_fc2.weight") or key.endswith("mlp.shared_experts.linear_fc2.bias")
    if_legacy_group_moe_output = key.endswith("mlp.experts.weight2") or key.endswith("mlp.experts.bias2")
    if_te_group_moe_output = key.endswith("mlp.experts.linear_fc2.weight") or key.endswith("mlp.experts.linear_fc2.bias")
    
    return (if_attn_output or if_mlp_output or if_shard_moe_output or if_legacy_group_moe_output or if_te_group_moe_output)

def load_in_func(key, value, multiplier=1.0, zerofy_output_init=False):
    # handle megatron distributed checkpoint format
    if isinstance(value, io.BytesIO):
        value.seek(0)
        tensor = torch.load(value, map_location='cpu')
        if tensor is None:      # for example, the extra_state
            return None
        if zerofy_output_init and identify_output_layer(key):
            tensor = torch.zeros_like(tensor)
        else:
            tensor = tensor * multiplier
        new_buf = io.BytesIO()
        torch.save(tensor, new_buf)
        new_buf.seek(0)
        return new_buf
    # handle megatron legacy checkpoint format
    else:
        if value is None:       # for example, the extra_state
            return None
        assert isinstance(value, torch.Tensor), f"Expected tensor for {key}, got {type(value)}"
        if zerofy_output_init and identify_output_layer(key):
            return torch.zeros_like(value)
        return value * multiplier
        

def add_noise_to_tensor(tensor, std_scaling_factor=0.01):
    if tensor is None:
        return None
    noise = torch.normal(mean=0.0, std=tensor.std().item() * std_scaling_factor, size=tensor.shape, device=tensor.device)
    return tensor + noise


#! main growth func
def model_depth_growth(model, args, model_provider_func, verbose=False, model2=None):
    """
    Main function to grow the model in depth-wise manner.
    
    Args:
        model: The original model to grow.
        args: Megatron args
        model_provider_func: Function to provide the model.
        verbose: If True, print detailed information about the model growth process.
        model2: If True, using model merging to grow the model.
    Returns:
        model: The grown model.
    """
    
    if model2 is not None:
        raise NotImplementedError("Model merging for depth growth is not implemented yet.")
    
    print_rank_0(f"model before growth: {model}")
        
    # 1. extract original decoder layers parameters
    layer_pattern = re.compile(r"decoder\.layers\.(\d+)\.(.+)")
    layer_params = defaultdict(dict)

    for key, value in model[0].state_dict().items():
        match = layer_pattern.match(key)
        if match:
            layer_idx, sub_key = int(match.group(1)), match.group(2)
            layer_params[layer_idx][sub_key] = value
            
    # 2. create new state_dict, and copy non-decoder layers        
    growth_state_dict = {}
    for key, value in model[0].state_dict().items():
        if not key.startswith("decoder.layers."):
            growth_state_dict[key] = value
    
    # 3. add new decoder layers
    original_layers = sorted(layer_params.keys())
    new_layer_idx = 0


    if args.growth_stack_method == "interleaved":
        print_rank_0(f">> growth stack method: interleaved")
        #! method 01: interleaved original + growth layers, total layer * 2
        for i in original_layers:
            # insert original layer
            for sub_key, value in layer_params[i].items():
                new_key = f"decoder.layers.{new_layer_idx}.{sub_key}"
                tensor_value = load_in_func(new_key, value, multiplier=args.growth_weight_multiplier, zerofy_output_init=False)     #! original layer should not zerofy output init
                growth_state_dict[new_key] = tensor_value
                if tensor_value is None:
                    print_rank_0(f"Warning: {new_key} is None, skipping initialization.") if verbose else None
                else:
                    print_rank_0(f"Initialized new_layer {new_key} with original layer {i} - shape: {tensor_value.shape}, mean: {tensor_value.mean().item():.6}, std: {tensor_value.std().item():.6}") if verbose else None

            print_rank_0(f"Inserted original layer {i} as new layer {new_layer_idx}") if (not verbose) else None
            new_layer_idx += 1
            
            # skip growth for the first few layers and the last few layers if specified
            if i < args.growth_ignore_first_num_layers or i >= (len(original_layers) - args.growth_ignore_last_num_layers):
                print_rank_0(f"Skipping growth for original layer {i} due to growth ignore settings.")
                continue

            # insert growth layer (can be a copy or a slightly perturbed version)
            for sub_key, value in layer_params[i].items():
                new_key = f"decoder.layers.{new_layer_idx}.{sub_key}"
                tensor_value = load_in_func(new_key, value, multiplier=args.growth_weight_multiplier, zerofy_output_init=args.growth_zerofy_output_init)
                growth_state_dict[new_key] = tensor_value
                if tensor_value is None:
                    print_rank_0(f"Warning: {new_key} is None, skipping initialization.") if verbose else None
                else:
                    print_rank_0(f"Initialized new_growth_layer {new_key} with original layer {i} - shape: {tensor_value.shape}, mean: {tensor_value.mean().item():.6}, std: {tensor_value.std().item():.6}") if verbose else None

            print_rank_0(f"Inserted original layer {i} as new growth layer {new_layer_idx}") if (not verbose) else None
            new_layer_idx += 1
            
    elif args.growth_stack_method == "stacked":
        print_rank_0(f">> growth stack method: stacked")
        #! method 02: stack original layers, then growth layers, total layer * 2
        for i in original_layers:
            # skip insert original layers for the last few layers if specified
            if i >= (len(original_layers) - args.growth_ignore_last_num_layers):
                print_rank_0(f"Skipping original copy for layer {i} due to growth ignore settings.")
                continue
            
            # insert original layer
            for sub_key, value in layer_params[i].items():
                new_key = f"decoder.layers.{new_layer_idx}.{sub_key}"
                tensor_value = load_in_func(new_key, value, multiplier=args.growth_weight_multiplier, zerofy_output_init=False)     #! original layer should not zerofy output init
                growth_state_dict[new_key] = tensor_value
                if tensor_value is None:
                    print_rank_0(f"Warning: {new_key} is None, skipping initialization.") if verbose else None
                else:
                    print_rank_0(f"Initialized new_layer: {new_key} with original layer {i} - shape: {tensor_value.shape}, mean: {tensor_value.mean().item():.6}, std: {tensor_value.std().item():.6}") if verbose else None
            print_rank_0(f"Inserted original layer {i} as new layer {new_layer_idx}") if (not verbose) else None
            new_layer_idx += 1
            
        # insert growth layer (can be a copy or a slightly perturbed version)
        for i in original_layers:
            # skip growth for the first few layers if specified
            if i < args.growth_ignore_first_num_layers:
                print_rank_0(f"Skipping growth for layer {i} due to growth ignore settings.")
                continue
            
            for sub_key, value in layer_params[i].items():
                new_key = f"decoder.layers.{new_layer_idx}.{sub_key}"
                tensor_value = load_in_func(new_key, value, multiplier=args.growth_weight_multiplier, zerofy_output_init=args.growth_zerofy_output_init)
                growth_state_dict[new_key] = tensor_value
                if tensor_value is None:
                    print_rank_0(f"Warning: {new_key} is None, skipping initialization.") if verbose else None
                else:
                    print_rank_0(f"Initialized new_layer {new_key} with original layer {i} - shape: {tensor_value.shape}, mean: {tensor_value.mean().item():.6}, std: {tensor_value.std().item():.6}") if verbose else None
            print_rank_0(f"Inserted original layer {i} as new growth layer {new_layer_idx}") if (not verbose) else None
            new_layer_idx += 1

    # 4. update model config and load new state_dict
    original_layers_count = args.num_layers
    new_layers_count = new_layer_idx
    print_rank_0(f">> original layers count: {original_layers_count}, new layers count: {new_layers_count}")
    
    args.num_layers = new_layers_count
    args.moe_layer_freq = [1] * (new_layers_count)
    if hasattr(args, "moe_layer_recompute_freq"):
        args.moe_layer_recompute_freq = [1] * (new_layers_count)

    model = model_provider_func()
    if not isinstance(model, list):
        model = [model]
        
    print_rank_0(f"model after growth: {model}")
    model[0].load_state_dict(growth_state_dict, strict=True)

    torch.distributed.barrier()
    return model


def interleaved_cat(a: torch.Tensor, b: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Cat two tensors in an interleaved manner along the specified dimension.
    """
    if a.shape != b.shape:
        if any([sa != sb for i,(sa,sb) in enumerate(zip(a.shape,b.shape)) if i != dim]):
            raise ValueError("Two tensors' shape must match except in cat dimension {}".format(dim))

    # stack [a,b] in dim+1
    stacked = torch.stack([a, b], dim=dim+1)
    # reshape to interleaved
    shape = list(a.shape)
    shape[dim] *= 2  # double length in cat dim
    return stacked.reshape(*shape)


def model_moe_width_growth(model, args, model_provider_func, verbose=False, model2=None):
    """
    Main function to grow the model's moe part in width-wise manner.
    
    Args:
        model: The original model to grow.
        args: Megatron args
        model_provider_func: Function to provide the model.
        verbose: If True, print detailed information about the model growth process.
        model2: If True, using model merging to grow the model.
    Returns:
        model: The grown model.
    """
    
    print_rank_0(f"model before growth: {model}")
    
    # 1. copy new state_dict     
    growth_state_dict = model[0].state_dict()
    
    use_model_merge = (model2 is not None and args.use_ckpt_merge)
    second_state_dict = model2[0].state_dict() if model2 is not None else None

    # 2. modify the state_dict for growth
    for key, value in growth_state_dict.items():
        
        if "mlp.router" in key:
            tensor_value = load_in_func(key, value, multiplier=args.growth_weight_multiplier, zerofy_output_init=False)
            if use_model_merge:
                second_tensor_value = load_in_func(key, second_state_dict[key], multiplier=args.growth_weight_multiplier, zerofy_output_init=False)
            else:
                second_tensor_value = tensor_value
            # model.decoder.layers.0.mlp.router.weight          (8, 1024) (num_experts, hidden_size)
            # model.decoder.layers.0.mlp.router.expert_bias     (8,) (num_experts,)
            if args.growth_use_random_router:
                if "weight" in key:
                    new_router_shape = torch.Size([value.shape[0] * 2, value.shape[1]])
                    growth_state_dict[key] = torch.normal(mean=0.0, std=0.02, size=new_router_shape)
                else:
                    new_router_shape = torch.Size([value.shape[0] * 2])
                    growth_state_dict[key] = torch.zeros(new_router_shape)
                print_rank_0(f"Using random router for {key} - (new/ori) shape: {growth_state_dict[key].shape}/{tensor_value.shape}, mean: {growth_state_dict[key].mean().item():.6}/{tensor_value.mean().item():.6}, std: {growth_state_dict[key].std().item():.6}/{tensor_value.std().item():.6}")
            else:
                if args.growth_add_expert_noise:
                    second_tensor_value = add_noise_to_tensor(second_tensor_value, std_scaling_factor=args.growth_expert_noise_std_scaling_factor)
                if args.growth_use_interleaved_moe_cat:
                    growth_state_dict[key] = interleaved_cat(tensor_value, second_tensor_value, dim=0)
                else:
                    growth_state_dict[key] = torch.cat([tensor_value, second_tensor_value], dim=0)  # double the router params

            if args.growth_zerofy_expert_bias and "expert_bias" in key:
                growth_state_dict[key] = torch.zeros(torch.Size([value.shape[0] * 2]))

            print_rank_0(f"Expand router {key} - (new/ori) shape: {growth_state_dict[key].shape}/{tensor_value.shape}, mean: {growth_state_dict[key].mean().item():.6}/{tensor_value.mean().item():.6}, std: {growth_state_dict[key].std().item():.6}/{tensor_value.std().item():.6}")

        elif "mlp.experts" in key:   
            if args.moe_use_legacy_grouped_gemm:
                #! legacy group gemm:
                # model.decoder.layers.0.mlp.experts.weight1        (1024, 32768) (hidden_size, num_experts * moe_ffn_hidden_size * 2(gelu factor))
                #   will be view to (num_experts, hidden_size, -1) for GroupGeMM
                # model.decoder.layers.0.mlp.experts.weight2        (16384, 1024) (num_experts * moe_ffn_hidden_size, hidden_size)
                #   will be view to (num_experts, -1, hidden_size) for GroupGeMM
                tensor_value = load_in_func(key, value, multiplier=args.growth_weight_multiplier)
                if use_model_merge:
                    second_tensor_value = load_in_func(key, second_state_dict[key], multiplier=args.growth_weight_multiplier)
                else:
                    second_tensor_value = tensor_value
                    
                if args.growth_add_expert_noise:
                    second_tensor_value = add_noise_to_tensor(second_tensor_value, std_scaling_factor=args.growth_expert_noise_std_scaling_factor)
                    
                if "weight1" in key:
                    tensor_value = tensor_value.view(args.num_experts, args.hidden_size, -1)
                    second_tensor_value = second_tensor_value.view(args.num_experts, args.hidden_size, -1)
                    if args.growth_use_interleaved_moe_cat:
                        merged_viewed_weight = interleaved_cat(tensor_value, second_tensor_value, dim=0)
                    else:
                        merged_viewed_weight = torch.cat([tensor_value, second_tensor_value], dim=0)        # cat on the num_experts dimension
                    growth_state_dict[key] = merged_viewed_weight.view(args.hidden_size, -1)
                elif "weight2" in key:
                    tensor_value = tensor_value.view(args.num_experts, -1, args.hidden_size)
                    second_tensor_value = second_tensor_value.view(args.num_experts, -1, args.hidden_size)
                    if args.growth_use_interleaved_moe_cat:
                        merged_viewed_weight = interleaved_cat(tensor_value, second_tensor_value, dim=0)
                    else:
                        merged_viewed_weight = torch.cat([tensor_value, second_tensor_value], dim=0)        # cat on the num_experts dimension
                    growth_state_dict[key] = merged_viewed_weight.view(-1, args.hidden_size)

                print_rank_0(f"Expand legacy expert weight {key} - (new/ori) shape: {growth_state_dict[key].shape}/{tensor_value.shape}, mean: {growth_state_dict[key].mean().item():.6}/{tensor_value.mean().item():.6}, std: {growth_state_dict[key].std().item():.6}/{tensor_value.std().item():.6}")
            else:
                #! te group gemm:
                # model.decoder.layers.0.mlp.experts.linear_fc1.weight<x>  (1024, 4096) (hidden_size, moe_ffn_hidden_size * 2)
                # model.decoder.layers.0.mlp.experts.linear_fc2.weight<x>  (2048, 1024) (moe_ffn_hidden_size, hidden_size)
                # <x>: the x-th expert
                if 'mlp.experts.linear_fc1' in key:
                    expert_id = int(re.search(r'\.experts\.linear_fc1\.weight(\d+)', key).group(1))
                    new_key = re.sub(r'\.experts\.linear_fc1\.weight(\d+)', f'.experts.linear_fc1.weight{expert_id + args.num_experts}', key)
                elif 'mlp.experts.linear_fc2' in key:
                    expert_id = int(re.search(r'\.experts\.linear_fc2\.weight(\d+)', key).group(1))
                    new_key = re.sub(r'\.experts\.linear_fc2\.weight(\d+)', f'.experts.linear_fc2.weight{expert_id + args.num_experts}', key)

                if use_model_merge:
                    second_tensor_value = load_in_func(key, second_state_dict[key], multiplier=args.growth_weight_multiplier)
                else:
                    second_tensor_value = load_in_func(key, value, multiplier=args.growth_weight_multiplier)
                    
                if args.growth_add_expert_noise:
                    second_tensor_value = add_noise_to_tensor(second_tensor_value, std_scaling_factor=args.growth_expert_noise_std_scaling_factor)
                    
                growth_state_dict[new_key] = second_tensor_value

                print_rank_0(f"Using original expert id {expert_id} to initialize the expert weight {new_key} - (new/ori) shape: {growth_state_dict[new_key].shape}/{tensor_value.shape}, mean: {growth_state_dict[new_key].mean().item():.6}/{tensor_value.mean().item():.6}, std: {growth_state_dict[new_key].std().item():.6}/{tensor_value.std().item():.6}")
        else:
            #! for non-moe layers, we directly copy the weights
            growth_state_dict[key] = load_in_func(key, value, multiplier=args.growth_weight_multiplier, zerofy_output_init=False)
            
            print_rank_0(f"Copying non-moe layer {key}")
            

    # 3. update model config and load new state_dict
    args.num_experts = args.num_experts * 2
    args.moe_router_topk = args.moe_router_topk * 2
    
    model = model_provider_func()
    if not isinstance(model, list):
        model = [model]
        
    print_rank_0(f"model after growth: {model}")
    model[0].load_state_dict(growth_state_dict, strict=True)
    
    torch.distributed.barrier()
    return model
    
    
def pretrain_with_model_only(
    model_provider,
    model_type,
    extra_args_provider=None,
    args_defaults={},
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
):
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks
    )
    # Model, optimizer, and learning rate.
    setup_model_and_optimizer(model_provider, model_type)


if __name__ == "__main__":
    
    from pretrain_gpt import model_provider
    from megatron.core.enums import ModelType
    
    pretrain_with_model_only(
        model_provider=model_provider,
        model_type=ModelType.encoder_or_decoder,
        extra_args_provider=add_growth_args,
    )
