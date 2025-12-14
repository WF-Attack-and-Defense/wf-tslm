from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, LongformerConfig, LongformerModel, LongformerTokenizer, BigBirdConfig, BigBirdModel, \
    BigBirdTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'LONGFORMER':
            self.longformer_config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')

            self.longformer_config.num_hidden_layers = configs.llm_layers
            # Longformer requires attention_window to match num_hidden_layers
            # Default attention window is 512, replicate it for all layers
            default_window = self.longformer_config.attention_window[0] if isinstance(self.longformer_config.attention_window, list) else 512
            self.longformer_config.attention_window = [default_window] * configs.llm_layers
            self.longformer_config.output_attentions = True
            self.longformer_config.output_hidden_states = True
            try:
                self.llm_model = LongformerModel.from_pretrained(
                    'allenai/longformer-base-4096',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.longformer_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LongformerModel.from_pretrained(
                    'allenai/longformer-base-4096',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.longformer_config,
                )

            try:
                self.tokenizer = LongformerTokenizer.from_pretrained(
                    'allenai/longformer-base-4096',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LongformerTokenizer.from_pretrained(
                    'allenai/longformer-base-4096',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BIGBIRD':
            self.bigbird_config = BigBirdConfig.from_pretrained('google/bigbird-roberta-base')

            self.bigbird_config.num_hidden_layers = configs.llm_layers
            self.bigbird_config.output_attentions = True
            self.bigbird_config.output_hidden_states = True
            try:
                self.llm_model = BigBirdModel.from_pretrained(
                    'google/bigbird-roberta-base',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bigbird_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BigBirdModel.from_pretrained(
                    'google/bigbird-roberta-base',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bigbird_config,
                )

            try:
                self.tokenizer = BigBirdTokenizer.from_pretrained(
                    'google/bigbird-roberta-base',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BigBirdTokenizer.from_pretrained(
                    'google/bigbird-roberta-base',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        # CNN layer before patch embedding to capture local patterns
        # Modified to preserve packet direction patterns (-1/+1)
        # Input: (batch, seq_len, features) -> Output: (batch, seq_len, d_model)
        # First layer: Point-wise convolution to preserve individual packet directions
        self.cnn_preprocessing_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=configs.enc_in, out_channels=configs.d_model, 
                     kernel_size=1, padding=0, stride=1, bias=False),  # Point-wise: preserves each packet
            nn.BatchNorm1d(configs.d_model),
            nn.GELU()
        )
        # Second layer: Small kernel to capture local patterns without blurring directions
        self.cnn_preprocessing_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                     kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout)
        )
        # Residual connection to preserve original packet direction information
        if configs.enc_in == configs.d_model:
            self.cnn_residual = nn.Identity()
        else:
            self.cnn_residual = nn.Conv1d(in_channels=configs.enc_in, 
                                          out_channels=configs.d_model, 
                                          kernel_size=1, bias=False)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        # head_nf: d_ff features per patch, times number of patches
        self.head_nf = self.d_ff * self.patch_nums
        
        # Bidirectional projection: concatenates forward+backward features (2*d_llm) -> d_ff
        # This projects the concatenated bidirectional features to d_ff dimension
        self.bidirectional_proj = nn.Linear(self.d_llm * 2, self.d_ff)

        # Packet direction pattern module for classification task
        if self.task_name == 'classification':
            self.packet_direction_module = PacketDirectionPatternModule(
                d_model=configs.d_model,
                seq_len=configs.seq_len,
                dropout=configs.dropout
            )
            # Projection to combine packet direction features with LLM features
            # Will be initialized dynamically based on actual patch count
            self.direction_feature_proj = None
        else:
            self.packet_direction_module = None
            self.direction_feature_proj = None

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            # Classification head with residual connections
            self.act = nn.GELU()
            num_classes = getattr(configs, 'num_class', 2)
            
            # First block with residual connection
            self.class_head_block1 = nn.Sequential(
                nn.Linear(self.head_nf, self.d_ff * 2),
                nn.LayerNorm(self.d_ff * 2),
                self.act,
                self.dropout
            )
            self.class_head_residual1 = nn.Linear(self.head_nf, self.d_ff * 2) if self.head_nf != self.d_ff * 2 else nn.Identity()
            
            # Second block with residual connection
            self.class_head_block2 = nn.Sequential(
                nn.Linear(self.d_ff * 2, self.d_ff),
                nn.LayerNorm(self.d_ff),
                self.act,
                self.dropout
            )
            self.class_head_residual2 = nn.Linear(self.d_ff * 2, self.d_ff) if self.d_ff * 2 != self.d_ff else nn.Identity()
            
            # Final projection to classes
            self.class_head_final = nn.Linear(self.d_ff, num_classes)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # Apply CNN preprocessing before patch embedding
        # Modified to preserve packet direction patterns with residual connection
        x_enc_for_cnn = x_enc.permute(0, 2, 1).contiguous()  # (B, N, T) where N is enc_in
        B_N = B * N
        # Reshape to process each feature separately: (B*N, 1, T)
        # This allows processing each feature channel independently while preserving patterns
        x_enc_cnn_input = x_enc_for_cnn.reshape(B_N, 1, T)  # (B*N, 1, T) - process each feature separately
        
        # First layer: Point-wise convolution to preserve packet directions
        x_enc_cnn_l1 = self.cnn_preprocessing_layer1(x_enc_cnn_input)  # (B*N, d_model, T)
        
        # Second layer: Small kernel for local patterns
        x_enc_cnn_l2 = self.cnn_preprocessing_layer2(x_enc_cnn_l1)  # (B*N, d_model, T)
        
        # Residual connection to preserve original packet direction information
        x_enc_residual = self.cnn_residual(x_enc_cnn_input)  # (B*N, d_model, T)
        x_enc_cnn_output = x_enc_cnn_l2 + x_enc_residual  # Residual connection
        
        x_enc_cnn_output = x_enc_cnn_output.permute(0, 2, 1)  # (B*N, T, d_model)
        x_enc_processed = x_enc_cnn_output.mean(dim=-1, keepdim=True)  # (B*N, T, 1)
        
        # Patch embedding expects (batch*vars, n_vars, seq_len)
        # We have (B*N, T, 1), need to transpose to (B*N, 1, T)
        x_enc_processed = x_enc_processed.permute(0, 2, 1)  # (B*N, 1, T)
        
        # Ensure sequence length is at least patch_len
        actual_seq_len = x_enc_processed.shape[-1]
        if actual_seq_len < self.patch_len:
            # Pad to at least patch_len
            padding_size = self.patch_len - actual_seq_len
            x_enc_processed = torch.nn.functional.pad(x_enc_processed, (0, padding_size), mode='replicate')

        # Patch embedding
        enc_out, n_vars = self.patch_embedding(x_enc_processed.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        # Bidirectional processing: process forward and backward sequences
        # Process forward direction
        llama_enc_forward = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out_forward = self.llm_model(inputs_embeds=llama_enc_forward).last_hidden_state
        
        # Process backward direction (reverse the sequence)
        enc_out_backward = torch.flip(enc_out, dims=[1])
        llama_enc_backward = torch.cat([prompt_embeddings, enc_out_backward], dim=1)
        dec_out_backward = self.llm_model(inputs_embeds=llama_enc_backward).last_hidden_state
        
        # Reverse backward output back to original order
        dec_out_backward = torch.flip(dec_out_backward, dims=[1])
        # Concatenate forward and backward features along feature dimension
        dec_out_combined = torch.cat([dec_out_forward, dec_out_backward], dim=-1)  # (B, seq_len, 2*d_llm)
        # Project concatenated features to d_ff dimension
        dec_out = self.bidirectional_proj(dec_out_combined)  # (B, seq_len, d_ff)

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        Classification method for traffic classification.
        x_enc: (batch_size, seq_len, num_features)
        x_mark_enc: (batch_size, seq_len, 4) - dummy timestamps
        Returns: (batch_size, num_classes)
        """
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: Tor network traffic classification task. "
                f"Task description: classify the traffic pattern into categories based on the input sequence; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # Extract packet direction pattern features (before CNN preprocessing to preserve -1/+1)
        # Process each feature channel separately
        packet_direction_features_list = []
        if self.packet_direction_module is not None:
            for n in range(N):
                x_enc_single = x_enc[:, :, n:n+1]  # (B, T, 1) - single feature channel
                direction_features = self.packet_direction_module(x_enc_single)  # (B, T, d_model)
                packet_direction_features_list.append(direction_features)
            
            # Combine direction features from all channels
            if packet_direction_features_list:
                packet_direction_features = torch.stack(packet_direction_features_list, dim=1)  # (B, N, T, d_model)
                packet_direction_features = packet_direction_features.mean(dim=1)  # (B, T, d_model) - average over channels
            else:
                packet_direction_features = None
        else:
            packet_direction_features = None

        # Apply CNN preprocessing before patch embedding
        # Modified to preserve packet direction patterns with residual connection
        # x_enc shape: (B, T, N) -> need (B*N, 1, T) for Conv1d where N is enc_in
        x_enc_for_cnn = x_enc.permute(0, 2, 1).contiguous()  # (B, N, T) where N is enc_in
        B_N = B * N
        # Reshape to process each feature separately: (B*N, 1, T)
        # This allows processing each feature channel independently while preserving patterns
        x_enc_cnn_input = x_enc_for_cnn.reshape(B_N, 1, T)  # (B*N, 1, T) - process each feature separately
        
        # First layer: Point-wise convolution to preserve packet directions
        x_enc_cnn_l1 = self.cnn_preprocessing_layer1(x_enc_cnn_input)  # (B*N, d_model, T)
        
        # Second layer: Small kernel for local patterns
        x_enc_cnn_l2 = self.cnn_preprocessing_layer2(x_enc_cnn_l1)  # (B*N, d_model, T)
        
        # Residual connection to preserve original packet direction information
        x_enc_residual = self.cnn_residual(x_enc_cnn_input)  # (B*N, d_model, T)
        x_enc_cnn_output = x_enc_cnn_l2 + x_enc_residual  # Residual connection
        
        # Verify sequence length is preserved
        if x_enc_cnn_output.shape[-1] != T:
            # If CNN reduced length, pad it back
            if x_enc_cnn_output.shape[-1] < T:
                padding_size = T - x_enc_cnn_output.shape[-1]
                x_enc_cnn_output = torch.nn.functional.pad(x_enc_cnn_output, (0, padding_size), mode='replicate')
            else:
                x_enc_cnn_output = x_enc_cnn_output[:, :, :T]
        
        x_enc_cnn_output = x_enc_cnn_output.permute(0, 2, 1)  # (B*N, T, d_model)
        # Use mean to get single channel for patch embedding
        x_enc_processed = x_enc_cnn_output.mean(dim=-1, keepdim=True)  # (B*N, T, 1)
        
        # Patch embedding expects (batch*vars, n_vars, seq_len)
        # We have (B*N, T, 1), need to transpose to (B*N, 1, T)
        x_enc_processed = x_enc_processed.permute(0, 2, 1)  # (B*N, 1, T)
        
        # Ensure sequence length is at least patch_len
        actual_seq_len = x_enc_processed.shape[-1]
        if actual_seq_len < self.patch_len:
            # Pad to at least patch_len
            padding_size = self.patch_len - actual_seq_len
            x_enc_processed = torch.nn.functional.pad(x_enc_processed, (0, padding_size), mode='replicate')

        # Patch embedding
        enc_out, n_vars = self.patch_embedding(x_enc_processed.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        # Bidirectional processing: process forward and backward sequences
        # Process forward direction
        llama_enc_forward = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out_forward = self.llm_model(inputs_embeds=llama_enc_forward).last_hidden_state
        
        # Process backward direction (reverse the sequence)
        enc_out_backward = torch.flip(enc_out, dims=[1])
        llama_enc_backward = torch.cat([prompt_embeddings, enc_out_backward], dim=1)
        dec_out_backward = self.llm_model(inputs_embeds=llama_enc_backward).last_hidden_state
        
        # Reverse backward output back to original order and concatenate
        dec_out_backward = torch.flip(dec_out_backward, dims=[1])
        # Concatenate forward and backward outputs along feature dimension
        dec_out = torch.cat([dec_out_forward, dec_out_backward], dim=-1)  # Concatenate features
        # Project concatenated features to d_ff dimension
        dec_out = self.bidirectional_proj(dec_out)  # (B*N, seq_len_with_prompt, d_ff)
        
        # Extract only the patch tokens (skip prompt tokens)
        # dec_out shape: (B*N, seq_len_with_prompt, d_ff)
        # We need to extract the patch tokens which come after the prompt
        prompt_len = prompt_embeddings.shape[1]
        dec_out = dec_out[:, prompt_len:, :]  # (B*N, patch_seq_len, d_ff)
        
        # Reshape to separate batch and variables
        # dec_out shape: (B*N, patch_seq_len, d_ff)
        dec_out = torch.reshape(dec_out, (B, N, dec_out.shape[1], dec_out.shape[2]))  # (B, N, patch_seq_len, d_ff)
        
        # Take the last patch_nums patches for each variable
        # Ensure we don't exceed the available patches
        actual_patch_len = dec_out.shape[2]
        num_patches_to_use = min(self.patch_nums, actual_patch_len)
        dec_out = dec_out[:, :, -num_patches_to_use:, :]  # (B, N, num_patches_to_use, d_ff)
        
        # For classification, we typically have N=1 (single feature)
        # Flatten: (B, N, num_patches_to_use, d_ff) -> (B, head_nf) where head_nf = d_ff * patch_nums
        # If we have fewer patches than expected, we need to pad or project
        if N == 1:
            dec_out = dec_out.squeeze(1)  # (B, num_patches_to_use, d_ff)
            dec_out = dec_out.reshape(B, num_patches_to_use * self.d_ff)  # (B, num_patches_to_use * d_ff)
        else:
            # Average over variables, then flatten
            dec_out = dec_out.mean(dim=1)  # (B, num_patches_to_use, d_ff)
            dec_out = dec_out.reshape(B, num_patches_to_use * self.d_ff)  # (B, num_patches_to_use * d_ff)
        
        # Integrate packet direction pattern features
        if packet_direction_features is not None:
            # Aggregate packet direction features (mean pooling over time)
            direction_features_agg = packet_direction_features.mean(dim=1)  # (B, d_model)
            
            # Project to match dec_out dimension if needed
            target_dim = dec_out.shape[1]  # num_patches_to_use * d_ff
            if self.direction_feature_proj is None or self.direction_feature_proj.out_features != target_dim:
                self.direction_feature_proj = nn.Linear(self.d_model, target_dim).to(dec_out.device)
            direction_features_proj = self.direction_feature_proj(direction_features_agg)  # (B, target_dim)
            
            # Combine LLM features with packet direction features (additive combination)
            dec_out = dec_out + direction_features_proj
        
        # Project to head_nf if the dimensions don't match
        if dec_out.shape[1] != self.head_nf:
            if not hasattr(self, 'head_size_proj'):
                self.head_size_proj = nn.Linear(dec_out.shape[1], self.head_nf).to(dec_out.device)
            dec_out = self.head_size_proj(dec_out)  # (B, head_nf)
        
        # Classification head with residual connections
        # First block with residual
        x1 = self.class_head_block1(dec_out)
        x1_residual = self.class_head_residual1(dec_out)
        x1 = x1 + x1_residual  # Residual connection
        
        # Second block with residual
        x2 = self.class_head_block2(x1)
        x2_residual = self.class_head_residual2(x1)
        x2 = x2 + x2_residual  # Residual connection
        
        # Final projection
        output = self.class_head_final(x2)
        
        return output

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class PacketDirectionPatternModule(nn.Module):
    """
    Module to explicitly model packet direction patterns:
    - Direction changes (incoming ↔ outgoing)
    - Burst patterns (consecutive same-direction packets)
    - Burst positions
    - Count from different directions
    """
    def __init__(self, d_model, seq_len, dropout=0.1):
        super(PacketDirectionPatternModule, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Embedding for transition types: outgoing->outgoing, outgoing->incoming, incoming->incoming, incoming->outgoing
        self.transition_embedding = nn.Embedding(4, d_model)
        
        # Embedding for burst lengths (binned)
        self.burst_length_embedding = nn.Embedding(32, d_model)  # Support burst lengths up to 32
        
        # Embedding for burst positions (relative position in sequence)
        self.burst_position_embedding = nn.Embedding(64, d_model)  # Support 64 position bins
        
        # Direction count features projection
        self.direction_count_proj = nn.Linear(4, d_model)  # incoming_count, outgoing_count, incoming_ratio, outgoing_ratio
        
        # Attention mechanism for transition patterns
        self.transition_attention = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        
        # Burst pattern attention
        self.burst_attention = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),  # transition + burst + direction_count
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def extract_direction_features(self, x):
        """
        Extract packet direction features from input sequence.
        x: (B, T, 1) or (B*N, T, 1) with values -1 (outgoing) or +1 (incoming)
        """
        B, T, _ = x.shape
        device = x.device
        
        # Ensure values are exactly -1 or +1
        x_directions = torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))
        x_directions = x_directions.squeeze(-1)  # (B, T)
        
        # Initialize feature tensors
        transition_features = []
        burst_features = []
        direction_count_features = []
        
        for b in range(B):
            seq = x_directions[b]  # (T,)
            
            # 1. Extract direction transitions
            # Map -1 (outgoing) = 0, +1 (incoming) = 1
            dir_ids = ((seq + 1) // 2).long()  # Convert -1/+1 to 0/1
            
            # Calculate transitions: prev_dir * 2 + curr_dir
            prev_dirs = dir_ids[:-1]
            curr_dirs = dir_ids[1:]
            transition_ids = prev_dirs * 2 + curr_dirs  # 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
            
            # Pad first position (no transition before first packet)
            transition_ids = torch.cat([torch.tensor([0], device=device), transition_ids])
            
            # 2. Extract burst patterns
            burst_lengths = []
            burst_positions = []
            burst_types = []  # 0 for outgoing, 1 for incoming
            
            current_burst_length = 1
            current_burst_type = dir_ids[0].item()
            burst_start_pos = 0
            
            for i in range(1, T):
                if dir_ids[i] == current_burst_type:
                    current_burst_length += 1
                else:
                    # End of current burst
                    burst_lengths.append(min(current_burst_length, 31))  # Cap at 31 for embedding
                    burst_positions.append(min(burst_start_pos * 64 // T, 63))  # Normalize to 0-63
                    burst_types.append(current_burst_type)
                    
                    # Start new burst
                    current_burst_type = dir_ids[i].item()
                    current_burst_length = 1
                    burst_start_pos = i
            
            # Add final burst
            burst_lengths.append(min(current_burst_length, 31))
            burst_positions.append(min(burst_start_pos * 64 // T, 63))
            burst_types.append(current_burst_type)
            
            # Convert to tensors
            burst_lengths_tensor = torch.tensor(burst_lengths, device=device).long()
            burst_positions_tensor = torch.tensor(burst_positions, device=device).long()
            
            # 3. Direction counts
            incoming_count = torch.sum(dir_ids == 1).float()
            outgoing_count = torch.sum(dir_ids == 0).float()
            incoming_ratio = incoming_count / T if T > 0 else 0.0
            outgoing_ratio = outgoing_count / T if T > 0 else 0.0
            
            direction_count_features.append(torch.tensor([
                incoming_count / T,  # Normalized count
                outgoing_count / T,
                incoming_ratio,
                outgoing_ratio
            ], device=device))
            
            # Create transition sequence embedding
            transition_ids_tensor = transition_ids.long()
            transition_emb = self.transition_embedding(transition_ids_tensor)  # (T, d_model)
            transition_features.append(transition_emb)
            
            # Create burst sequence embedding
            # For each time step, find which burst it belongs to and embed it
            burst_emb_seq = torch.zeros(T, self.d_model, device=device)
            burst_idx = 0
            current_pos = 0
            
            for i, length in enumerate(burst_lengths):
                burst_len_emb = self.burst_length_embedding(burst_lengths_tensor[i])
                burst_pos_emb = self.burst_position_embedding(burst_positions_tensor[i])
                burst_combined = (burst_len_emb + burst_pos_emb) / 2
                
                # Assign to all positions in this burst
                end_pos = min(current_pos + length, T)
                burst_emb_seq[current_pos:end_pos] = burst_combined.unsqueeze(0).expand(end_pos - current_pos, -1)
                current_pos = end_pos
                if current_pos >= T:
                    break
            
            burst_features.append(burst_emb_seq)
        
        # Stack features
        transition_features = torch.stack(transition_features)  # (B, T, d_model)
        burst_features = torch.stack(burst_features)  # (B, T, d_model)
        direction_count_features = torch.stack(direction_count_features)  # (B, 4)
        
        return transition_features, burst_features, direction_count_features
    
    def forward(self, x):
        """
        x: (B, T, 1) or (B*N, T, 1) with packet direction values (-1/+1)
        Returns: (B, T, d_model) enhanced features
        """
        # Extract direction features
        transition_features, burst_features, direction_count_features = self.extract_direction_features(x)
        
        # Apply attention to transition patterns
        transition_attended, _ = self.transition_attention(
            transition_features, transition_features, transition_features
        )
        
        # Apply attention to burst patterns
        burst_attended, _ = self.burst_attention(
            burst_features, burst_features, burst_features
        )
        
        # Project direction count features and expand to sequence length
        direction_count_proj = self.direction_count_proj(direction_count_features)  # (B, d_model)
        direction_count_expanded = direction_count_proj.unsqueeze(1).expand(-1, transition_features.shape[1], -1)  # (B, T, d_model)
        
        # Concatenate all features
        combined_features = torch.cat([
            transition_attended,
            burst_attended,
            direction_count_expanded
        ], dim=-1)  # (B, T, d_model * 3)
        
        # Fuse features
        output = self.feature_fusion(combined_features)  # (B, T, d_model)
        
        return output


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
