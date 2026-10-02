import torch
import torch.nn as nn
import torch.nn.functional as F
import json

import transformers
from typing import Callable, List, Optional, Tuple, Union

class TransformerConfig(transformers.PretrainedConfig):
    model_type = "custom_transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: Optional[int] = None,
        max_postion_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_postion_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout

        if self.head_dim is None:
            self.head_dim = hidden_size // num_attention_heads

        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        assert self.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        assert self.hidden_size == (self.num_attention_heads * self.head_dim), "hidden_size must be equal to num_attention_heads * head_dim"

def apply_rotary_emb(x, position_embeddings):
    cos,sin = position_embeddings
    # fill here: rotate hidden states with positon angle
    # shape hint
    # cos,sin: (batch_size, seq_len, head_dim // 2)
    # x: (batch_size, num_attention_heads, seq_len, head_dim)
    #######
        
    #### YOUR CODES; TODO 
    cos = cos.unsqueeze(1)  # (batch_size, 1, seq_len, head_dim // 2): broadcast over heads
    sin = sin.unsqueeze(1)
    x1, x2 = x.chunk(2, dim=-1)  # (x1[i], x2[i]) is the 2D pair rotated by angle m * theta_i
    x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    #######
    return x

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # fill here: rms normalization
        #######
        
        #### YOUR CODES; TODO 
        input_dtype = x.dtype
        x = x.to(torch.float32)  # compute the statistics in fp32 for numerical stability
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        output = self.weight * x.to(input_dtype)
        #######
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

class RotaryEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig, device=None):
        super().__init__()
        self.config = config
        
        # fill here: rotary embedding initalization
        # shape hint
        # inv_freq: (1, head_dim // 2)
        ######
        #### YOUR CODES; TODO 
        base = config.rope_theta
        dim = config.head_dim
        # theta_i = base^(-2i / head_dim), i = 0, 1, ..., head_dim // 2 - 1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        ######

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        with torch.autocast(device_type=x.device.type, enabled=False): # disable autocasting for fp32 precision
            # fill here: calculate rotary embedding
            # shape hint
            # cos,sin: (batch_size, seq_len, head_dim // 2)
            ######
            
            #### YOUR CODES; TODO 
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  # m * theta_i
            cos = freqs.cos()
            sin = freqs.sin()
            ######

        return cos.to(x.dtype), sin.to(x.dtype)

class MultiHeadAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        #fill here: initalization of query, key, value and output projection
        ######
        
        #### YOUR CODES; TODO 
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        ######

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[List[Tuple[torch.Tensor,torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.size()
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        # batch , num_head, seq_len, head_dim
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        

        query_states = apply_rotary_emb(query_states, position_embeddings)
        key_states = apply_rotary_emb(key_states, position_embeddings)

        if past_key_value is not None:
            key_cache, value_cache = past_key_value
            key_states = torch.cat([key_cache, key_states], dim=-2)
            value_states = torch.cat([value_cache, value_states], dim=-2)
            past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # fill here: calculate attention weights
        # shape hint
        # qkv states: (batch_size, num_attention_heads, seq_len, head_dim)
        # attention_weights: (batch_size, num_attention_heads, seq_len, seq_len + cache_len)
        # attn_output: (batch_size, seq_len, hidden_size)
        #######
        
        #### YOUR CODES; TODO 
        attention_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask  # additive mask: 0 = keep, dtype min = masked
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attention_weights, value_states)  # (batch_size, num_attention_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        #######
        return attn_output, past_key_value

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size,bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size,bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,bias=False)
        self.act_fn = torch.nn.SiLU()
        self.dropout = nn.Dropout(config.ffn_dropout)

    def forward(self, x):
        # fill here: feed forward network
        ######
        
        #### YOUR CODES; TODO 
        # SwiGLU: down(SiLU(gate(x)) * up(x))
        outputs = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        outputs = self.dropout(outputs)
        ######
        return outputs

class TransformerLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MultiHeadAttention(config)

        self.feed_forward = FeedForwardNetwork(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # fill here: transformer layer
        # return layer hidden states and past key values with output of attention
        ######
        
        #### YOUR CODES; TODO 
        # Pre-Norm: h = x + Attn(RMSNorm(x)), out = h + FFN(RMSNorm(h))
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        ######
        return hidden_states, past_key_value

class TransformerPreTrainedModel(transformers.PreTrainedModel):
    config_class = TransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class TransformerModel(TransformerPreTrainedModel):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
    ):
        batch_size, seq_len = input_ids.shape
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("You cannot specify both input_ids and input_embeds at the same time")
        if inputs_embeds is None and input_ids is None:
            raise ValueError("You have to specify either input_ids or input_embeds")
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            inputs_embeds = inputs_embeds

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, device=inputs_embeds.device).unsqueeze(0)

        target_length = seq_len
        seen_token_length = 0
        if past_key_values is not None:
            seen_token_length = past_key_values[0][0].shape[-2]
            target_length += seen_token_length
        
        attention_mask = self._prepare_attention_mask(
            attention_mask=attention_mask,
            sequence_length=seq_len,
            target_length=target_length,
            seen_token_length=seen_token_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            batch_size=batch_size,
        )

        hidden_states = inputs_embeds
        position_embed = self.rotary_emb(hidden_states, position_ids)
        kv_cache_new = []
        for layer_idx, decoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values[layer_idx] if past_key_values is not None else None,
                    position_embed
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_values[layer_idx] if past_key_values is not None else None,
                    position_embeddings=position_embed,
                )

            hidden_states, kv_cache = layer_outputs
            kv_cache_new.append(kv_cache)

        hidden_states = self.norm(hidden_states)

        if past_key_values is not None:
            past_key_values = kv_cache_new

        return hidden_states, past_key_values

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        seen_token_length: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int
    ):
        # fill here: prepare attention mask
        # shape hint
        # mask: (batch_size, 1, sequence_length, target_length)
        ######
        
        #### YOUR CODES; TODO 
        min_dtype = torch.finfo(dtype).min
        # causal: query i (absolute position seen_token_length + i) can attend to key j only if j <= seen_token_length + i
        query_positions = torch.arange(sequence_length, device=device) + seen_token_length
        key_positions = torch.arange(target_length, device=device)
        allowed = key_positions[None, :] <= query_positions[:, None]  # (sequence_length, target_length)
        allowed = allowed[None, None, :, :].expand(batch_size, 1, sequence_length, target_length)
        # padding: never attend to padded keys (attention_mask == 0)
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].to(device=device, dtype=torch.bool)
        mask = torch.zeros((batch_size, 1, sequence_length, target_length), dtype=dtype, device=device)
        mask = mask.masked_fill(~allowed, min_dtype)
        ######
        return mask

class TransformerForCausalLM(TransformerPreTrainedModel):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.model = TransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs
    ):  
        if use_cache and past_key_values is None:
            batch_size, _ = input_ids.shape
            dummy = torch.empty((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim)).to(self.model.layers[0].self_attn.q_proj.weight)
            past_key_values = [(dummy.clone(),dummy.clone()) for _ in range(self.config.num_hidden_layers)]
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            logits = logits.float() # cast to fp32 for calculating softmax in high precision

            # fill here: calculate cross entropy loss
            # loss must be scalar
            ######
            
            #### YOUR CODES; TODO 
            # next-token prediction: the logits at position t predict the label at position t + 1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            ######
        return (loss,logits) if loss is not None else (logits, past_key_values)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        max_new_tokens: int = 32,
        return_response_only: bool = False,
    ):
        batch_size, init_seq_len = input_ids.shape
        device = input_ids.device
        eos = self.config.eos_token_id

        unfinish_flag = torch.ones(batch_size, dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, -1)
        
        for _ in range(max_new_tokens):
            logits, past_key_values = self.forward(
                input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids[:, -1:] if past_key_values is not None else position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = next_tokens * unfinish_flag + eos * (1 - unfinish_flag)
            unfinish_flag = unfinish_flag * next_tokens.ne(eos)

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)], dim=-1)

            if unfinish_flag.sum() == 0:
                break
        if return_response_only:
            return input_ids[:, init_seq_len:]
        return input_ids

class TransformerForSequenceClassification(TransformerPreTrainedModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = TransformerModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor=None,
        inputs_embeds: torch.FloatTensor=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits = self.classifier(hidden_states[:, -1, :])
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), reduction="mean")
        return (loss, logits) if loss is not None else (logits, past_key_values)