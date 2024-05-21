import torch
import torch.nn as nn


class EncoderProjectorConcat(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.encoder_path_hf = model_config.encoder_path_hf 
        self.k = model_config.encoder_projector_ds_rate
        self.encoder_dim = model_config.encoder_dim
        self.llm_dim = model_config.llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, model_config.llm_dim)

    def forward(self, x):
        if self.encoder_path_hf is not None:
            x = x.last_hidden_state
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class EncoderProjectorCov1d(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.encoder_path_hf = model_config.encoder_path_hf 
        self.k = model_config.encoder_projector_ds_rate
        self.encoder_dim = model_config.encoder_dim
        self.llm_dim = model_config.llm_dim
        self.conv1d = nn.Conv1d(in_channels=self.encoder_dim, out_channels=self.encoder_dim, kernel_size=self.k, stride=self.k, padding=0)
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        if self.encoder_path_hf is not None:
            x = x.last_hidden_state
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.encoder_path_hf = model_config.encoder_path_hf 
        self.encoder_dim = model_config.encoder_dim
        self.llm_dim = model_config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = 8

        self.query_len = 80
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        if self.encoder_path_hf is not None:
            x = x.last_hidden_state
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj