import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from utils.load_graph_data import load_graph_data


class DFTSeriesDecomposition(nn.Module):
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, _ = torch.topk(freq, k=self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        configs.seq_len // configs.down_sampling_window ** i,
                        configs.seq_len // configs.down_sampling_window ** (i + 1),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        configs.seq_len // configs.down_sampling_window ** (i + 1),
                        configs.seq_len // configs.down_sampling_window ** (i + 1),
                    ),
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))
        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        configs.seq_len // configs.down_sampling_window ** (i + 1),
                        configs.seq_len // configs.down_sampling_window ** i,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        configs.seq_len // configs.down_sampling_window ** i,
                        configs.seq_len // configs.down_sampling_window ** i,
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))
        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.channel_independence = configs.channel_independence
        if configs.decomp_method == "moving_avg":
            self.decomposition = series_decomp(configs.moving_avg)
        else:
            self.decomposition = DFTSeriesDecomposition(configs.top_k)
        if not configs.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        lengths = [x.size(1) for x in x_list]
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        out_list = []
        for original, out_season, out_trend, length in zip(
            x_list,
            out_season_list,
            out_trend_list,
            lengths,
        ):
            out = out_season + out_trend
            if self.channel_independence:
                out = original + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TemporalExpert(nn.Module):
    """
    Input shape
        - x_enc: [B, T_in, N]
        - x_mark_enc: [B, T_in, time_feature_dim]
        - x_dec: [B, T_label + T_out, N]
        - x_mark_dec: [B, T_label + T_out, time_feature_dim]
    Output shape
        - forecast: [B, N, T_out, 1]
        - hidden_states: list of [B, N, T_in, d_model], where d_model is 64 in MoSTE_BTH.sh
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.channel_independence = configs.channel_independence
        self.mixing_blocks = nn.ModuleList(
            [PastDecomposableMixing(configs) for _ in range(configs.e_layers)]
        )
        self.series_decomposition = series_decomp(configs.moving_avg)
        if self.channel_independence:
            self.embedding = DataEmbedding_wo_pos(
                1, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
        else:
            self.embedding = DataEmbedding_wo_pos(
                configs.enc_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
        self.normalization_layers = nn.ModuleList(
            [
                Normalize(
                    self.configs.enc_in,
                    affine=True,
                    non_norm=configs.use_norm == 0,
                )
                for _ in range(configs.down_sampling_layers + 1)
            ]
        )
        self.forecast_projections = nn.ModuleList(
            [
                nn.Linear(
                    configs.seq_len // configs.down_sampling_window ** i,
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        if self.channel_independence:
            self.output_projection = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.output_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
            self.residual_layers = nn.ModuleList(
                [
                    nn.Linear(
                        configs.seq_len // configs.down_sampling_window ** i,
                        configs.seq_len // configs.down_sampling_window ** i,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
            self.residual_forecast_projections = nn.ModuleList(
                [
                    nn.Linear(
                        configs.seq_len // configs.down_sampling_window ** i,
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

    def _project_output(self, forecast, scale_index, residual):
        forecast = self.output_projection(forecast)
        residual = residual.permute(0, 2, 1)
        residual = self.residual_layers[scale_index](residual)
        residual = self.residual_forecast_projections[scale_index](residual).permute(0, 2, 1)
        return forecast + residual

    def _preprocess_inputs(self, multiscale_inputs):
        if self.channel_independence:
            return multiscale_inputs, None

        seasonal_inputs = []
        trend_inputs = []
        for values in multiscale_inputs:
            seasonal, trend = self.series_decomposition(values)
            seasonal_inputs.append(seasonal)
            trend_inputs.append(trend)
        return seasonal_inputs, trend_inputs

    def _build_multiscale_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            downsample = nn.MaxPool1d(
                self.configs.down_sampling_window,
                return_indices=False,
            )
        elif self.configs.down_sampling_method == 'avg':
            downsample = nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            downsample = nn.Conv1d(
                in_channels=self.configs.enc_in,
                out_channels=self.configs.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.configs.down_sampling_window,
                padding_mode='circular',
                bias=False,
            )
        else:
            return x_enc, x_mark_enc

        current_values = x_enc.permute(0, 2, 1)
        current_time_features = x_mark_enc
        multiscale_inputs = [current_values.permute(0, 2, 1)]
        multiscale_time_features = [x_mark_enc]
        for _ in range(self.configs.down_sampling_layers):
            current_values = downsample(current_values)
            multiscale_inputs.append(current_values.permute(0, 2, 1))
            if x_mark_enc is not None:
                current_time_features = current_time_features[
                    :, ::self.configs.down_sampling_window, :
                ]
                multiscale_time_features.append(current_time_features)
        if x_mark_enc is None:
            multiscale_time_features = None
        return multiscale_inputs, multiscale_time_features

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch_size, input_steps, num_nodes = x_enc.shape
        multiscale_inputs, multiscale_time_features = self._build_multiscale_inputs(
            x_enc,
            x_mark_enc,
        )
        normalized_inputs = []
        embedding_time_features = []
        if multiscale_time_features is not None:
            for scale_index, (values, time_features) in enumerate(
                zip(multiscale_inputs, multiscale_time_features)
            ):
                scale_batch_size, scale_steps, scale_nodes = values.size()
                values = self.normalization_layers[scale_index](values, 'norm')
                if self.channel_independence:
                    values = values.permute(0, 2, 1).contiguous().reshape(scale_batch_size * scale_nodes, scale_steps, 1)
                    time_features = (
                        time_features.unsqueeze(1)
                        .expand(scale_batch_size, scale_nodes, scale_steps, time_features.size(-1))
                        .contiguous()
                        .view(scale_batch_size * scale_nodes, scale_steps, time_features.size(-1))
                    )
                normalized_inputs.append(values)
                embedding_time_features.append(time_features)
        else:
            for scale_index, values in enumerate(multiscale_inputs):
                scale_batch_size, scale_steps, scale_nodes = values.size()
                values = self.normalization_layers[scale_index](values, 'norm')
                if self.channel_independence:
                    values = values.permute(0, 2, 1).contiguous().reshape(scale_batch_size * scale_nodes, scale_steps, 1)
                normalized_inputs.append(values)

        processed_inputs = self._preprocess_inputs(normalized_inputs)
        encoded_scales = []
        if multiscale_time_features is not None:
            for values, time_features in zip(
                processed_inputs[0],
                embedding_time_features,
            ):
                encoded_scales.append(self.embedding(values, time_features))
        else:
            for values in processed_inputs[0]:
                encoded_scales.append(self.embedding(values, None))

        hidden_states = []
        for mixing_block in self.mixing_blocks:
            encoded_scales = mixing_block(encoded_scales)
            hidden = encoded_scales[0]
            if self.channel_independence:
                _, _, hidden_size = hidden.shape
                hidden = hidden.view(batch_size, num_nodes, input_steps, hidden_size)
            else:
                current_batch_size, _, hidden_size = hidden.shape
                hidden = hidden.unsqueeze(1).expand(
                    current_batch_size, num_nodes, input_steps, hidden_size
                )
            hidden_states.append(hidden)

        scale_forecasts = self._forecast_scales(
            batch_size,
            encoded_scales,
            processed_inputs,
        )
        forecast = torch.stack(scale_forecasts, dim=-1).sum(-1)
        forecast = self.normalization_layers[0](forecast, 'denorm')
        forecast = forecast.permute(0, 2, 1).unsqueeze(-1)
        return forecast, hidden_states

    def _forecast_scales(self, batch_size, encoded_scales, processed_inputs):
        scale_forecasts = []
        if self.channel_independence:
            for scale_index, encoded in enumerate(encoded_scales):
                forecast = self.forecast_projections[scale_index](
                    encoded.permute(0, 2, 1)
                ).permute(0, 2, 1)
                forecast = self.output_projection(forecast)
                forecast = forecast.reshape(batch_size, self.configs.c_out, self.pred_len)
                forecast = forecast.permute(0, 2, 1).contiguous()
                scale_forecasts.append(forecast)
        else:
            for scale_index, (encoded, residual) in enumerate(
                zip(encoded_scales, processed_inputs[1])
            ):
                forecast = self.forecast_projections[scale_index](
                    encoded.permute(0, 2, 1)
                ).permute(0, 2, 1)
                scale_forecasts.append(
                    self._project_output(forecast, scale_index, residual)
                )
        return scale_forecasts

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(-1, -2)
        attention_scores = query @ key / self.head_dim ** 0.5
        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            attention_scores.masked_fill_(~mask, -torch.inf)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        out = attention_scores @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.output_projection(out)
        return out, attention_scores


class SelfAttentionBlock(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False):
        super().__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, x, dim: int = -2):
        x = x.transpose(dim, -2)
        residual = x
        out, attention_scores = self.attention(x, x, x)
        out = self.attention_dropout(out)
        out = self.attention_norm(residual + out)
        residual = out
        out = self.feed_forward(out)
        out = self.feed_forward_dropout(out)
        out = self.feed_forward_norm(residual + out)
        out = out.transpose(dim, -2)
        return out, attention_scores


class LatentGraphExpert(nn.Module):
    """
    Input shape
        - x: [B, T_in, N, 3]
    Output shape
        - forecast: [B, T_out, N, 1]
        - hidden_states: list of [B, N, T_in, 176]
        - latent_adjacency: [B, T_in, N, N]
    """
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        time_of_day_embedding_dim=24,
        day_of_week_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_projection=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_of_day_embedding_dim = time_of_day_embedding_dim
        self.day_of_week_embedding_dim = day_of_week_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + time_of_day_embedding_dim
            + day_of_week_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.use_mixed_projection = use_mixed_projection
        self.input_projection = nn.Linear(input_dim, input_embedding_dim)
        if time_of_day_embedding_dim > 0:
            self.time_of_day_embedding = nn.Embedding(
                steps_per_day,
                time_of_day_embedding_dim,
            )
        if day_of_week_embedding_dim > 0:
            self.day_of_week_embedding = nn.Embedding(
                7,
                day_of_week_embedding_dim,
            )
        if spatial_embedding_dim > 0:
            self.node_embedding = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_embedding)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        if use_mixed_projection:
            self.forecast_projection = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_projection = nn.Linear(in_steps, out_steps)
            self.forecast_projection = nn.Linear(self.model_dim, self.output_dim)
        self.temporal_attention_layers = nn.ModuleList(
            [
                SelfAttentionBlock(
                    self.model_dim, feed_forward_dim, num_heads, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.spatial_attention_layers = nn.ModuleList(
            [
                SelfAttentionBlock(
                    self.model_dim, feed_forward_dim, num_heads, dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        batch_size = x.shape[0]
        values = x[..., 0:1]
        mean = values.mean(1, keepdim=True).detach()
        values = values - mean
        standard_deviation = torch.sqrt(
            torch.var(values, dim=1, keepdim=True, unbiased=False) + 1e-05
        )
        values /= standard_deviation
        x = torch.cat([values, x[..., 1:]], dim=-1)
        if self.time_of_day_embedding_dim > 0:
            time_of_day = x[..., 1]
        if self.day_of_week_embedding_dim > 0:
            day_of_week = x[..., 2]
        x = x[..., :self.input_dim]
        x = self.input_projection(x)
        features = [x]
        if self.time_of_day_embedding_dim > 0:
            time_of_day_index = torch.clamp(
                (time_of_day * self.steps_per_day).long(),
                0,
                self.steps_per_day - 1,
            )
            time_of_day_embedding = self.time_of_day_embedding(time_of_day_index)
            features.append(time_of_day_embedding)
        if self.day_of_week_embedding_dim > 0:
            day_of_week_index = torch.clamp(day_of_week.long(), 0, 6)
            day_of_week_embedding = self.day_of_week_embedding(day_of_week_index)
            features.append(day_of_week_embedding)
        if self.spatial_embedding_dim > 0:
            spatial_embedding = self.node_embedding.expand(
                batch_size,
                self.in_steps,
                *self.node_embedding.shape,
            )
            features.append(spatial_embedding)
        if self.adaptive_embedding_dim > 0:
            adaptive_embedding = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adaptive_embedding)
        x = torch.cat(features, dim=-1)
        hidden_states = []
        spatial_attention_scores = []
        for attention_layer in self.temporal_attention_layers:
            x, _ = attention_layer(x, dim=1)
            hidden_states.append(x)
        for attention_layer in self.spatial_attention_layers:
            x, attention_scores = attention_layer(x, dim=2)
            hidden_states.append(x)
            spatial_attention_scores.append(attention_scores)

        latent_adjacency = spatial_attention_scores[-1]
        latent_adjacency = latent_adjacency.view(self.num_heads, batch_size, self.in_steps, self.num_nodes, self.num_nodes).mean(dim=0)
        latent_adjacency = torch.relu(latent_adjacency)
        hidden_states = [
            hidden.permute(0, 2, 1, 3).contiguous()
            for hidden in hidden_states
        ]
        if self.use_mixed_projection:
            forecast = x.transpose(1, 2)
            forecast = forecast.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            forecast = self.forecast_projection(forecast).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
            forecast = forecast.transpose(1, 2)
        else:
            forecast = x.transpose(1, 3)
            forecast = self.temporal_projection(forecast)
            forecast = self.forecast_projection(forecast.transpose(1, 3))
        forecast = forecast * standard_deviation[:, 0, :].unsqueeze(1).repeat(1, self.out_steps, 1, 1)
        forecast = forecast + mean[:, 0, :].unsqueeze(1).repeat(1, self.out_steps, 1, 1)
        return forecast, hidden_states, latent_adjacency


POSITION_EMBEDDING_INIT_GAIN = 0.01


def build_localized_adjacency(adjacency, steps=3):
    """Builds the localized spatial-temporal adjacency used by the prior graph expert."""
    num_nodes = adjacency.shape[0]
    localized_adjacency = np.zeros(
        (num_nodes * steps, num_nodes * steps),
        dtype=np.float32,
    )
    for i in range(steps):
        localized_adjacency[
            i * num_nodes:(i + 1) * num_nodes,
            i * num_nodes:(i + 1) * num_nodes,
        ] = adjacency
    for node in range(num_nodes):
        for step in range(steps - 1):
            localized_adjacency[
                step * num_nodes + node,
                (step + 1) * num_nodes + node,
            ] = 1.0
            localized_adjacency[
                (step + 1) * num_nodes + node,
                step * num_nodes + node,
            ] = 1.0
    np.fill_diagonal(localized_adjacency, 1.0)
    return localized_adjacency


def normalize_adjacency(adjacency):
    degree = adjacency.sum(axis=1)
    inverse_sqrt_degree = np.power(np.maximum(degree, 1e-12), -0.5)
    return (inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]).astype(np.float32)


class SynchronousGraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, activation='GLU'):
        super().__init__()
        self.activation = activation
        self.fc = nn.Linear(in_dim, 2 * out_dim if activation == 'GLU' else out_dim)

    def forward(self, x, adjacency):
        x = torch.einsum('ij,jbc->ibc', adjacency, x)
        x = self.fc(x)
        if self.activation == 'GLU':
            lhs, rhs = torch.chunk(x, 2, dim=-1)
            return lhs * torch.sigmoid(rhs)
        return F.relu(x)


class SynchronousGraphModule(nn.Module):
    def __init__(self, num_nodes, in_dim, filters, activation='GLU'):
        super().__init__()
        self.num_nodes = num_nodes
        layers = []
        current_dim = in_dim
        for out_dim in filters:
            layers.append(
                SynchronousGraphConvolution(
                    current_dim,
                    out_dim,
                    activation=activation,
                )
            )
            current_dim = out_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x, adjacency):
        middle_outputs = []
        for layer in self.layers:
            x = layer(x, adjacency)
            middle_outputs.append(x[self.num_nodes:2 * self.num_nodes].unsqueeze(0))
        return torch.max(torch.cat(middle_outputs, dim=0), dim=0).values


class SynchronousGraphLayer(nn.Module):
    def __init__(
        self,
        input_length,
        num_nodes,
        in_dim,
        filters,
        module_type='sharing',
        activation='GLU',
        use_temporal_embedding=True,
        use_spatial_embedding=True,
    ):
        super().__init__()
        self.filters = filters
        self.module_type = module_type
        self.temporal_embedding = (
            nn.Parameter(torch.empty(1, input_length, 1, in_dim))
            if use_temporal_embedding
            else None
        )
        self.spatial_embedding = (
            nn.Parameter(torch.empty(1, 1, num_nodes, in_dim))
            if use_spatial_embedding
            else None
        )
        if module_type == 'sharing':
            self.graph_module = SynchronousGraphModule(
                num_nodes, in_dim, filters, activation=activation
            )
        else:
            self.graph_modules = nn.ModuleList(
                [
                    SynchronousGraphModule(num_nodes, in_dim, filters, activation=activation)
                    for _ in range(input_length - 2)
                ]
            )
        self.reset_parameters()

    def reset_parameters(self):
        if self.temporal_embedding is not None:
            nn.init.xavier_uniform_(
                self.temporal_embedding,
                gain=POSITION_EMBEDDING_INIT_GAIN,
            )
        if self.spatial_embedding is not None:
            nn.init.xavier_uniform_(
                self.spatial_embedding,
                gain=POSITION_EMBEDDING_INIT_GAIN,
            )

    def _add_position_embedding(self, x):
        if self.temporal_embedding is not None:
            x = x + self.temporal_embedding
        if self.spatial_embedding is not None:
            x = x + self.spatial_embedding
        return x

    def forward(self, x, adjacency):
        x = self._add_position_embedding(x)
        batch_size, steps, num_nodes, channels = x.shape
        if self.module_type == 'sharing':
            windows = []
            for i in range(steps - 2):
                window = x[:, i:i + 3, :, :].reshape(batch_size, 3 * num_nodes, channels)
                windows.append(window.permute(1, 0, 2))
            merged = torch.cat(windows, dim=1)
            out = self.graph_module(merged, adjacency)
            out = out.reshape(num_nodes, steps - 2, batch_size, self.filters[-1])
            return out.permute(2, 1, 0, 3).contiguous()

        outputs = []
        for i, graph_module in enumerate(self.graph_modules):
            window = x[:, i:i + 3, :, :].reshape(batch_size, 3 * num_nodes, channels)
            window = window.permute(1, 0, 2)
            out = graph_module(window, adjacency).permute(1, 0, 2).unsqueeze(1)
            outputs.append(out)
        return torch.cat(outputs, dim=1).contiguous()


class PriorGraphProjection(nn.Module):
    def __init__(self, input_length, in_dim, hidden_dim=128, pred_len=12):
        super().__init__()
        self.fc1 = nn.Linear(input_length * in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        batch_size, steps, num_nodes, channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, steps * channels)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 2, 1).contiguous()


class PriorGraphExpert(nn.Module):
    """
    Input shape
        - x: [B, T_in, N]
    Output shape
        - forecast: [B, N, T_out, 1]
        - hidden_state: [B, N, T_in, hidden_size], where hidden_size is 64 in MoSTE_BTH.sh
    """
    def __init__(
        self,
        hidden_size,
        num_nodes,
        prior_graph_path=None,
        seq_len=12,
        pred_len=12,
        in_dim=1,
        layers=2,
        module_type='sharing',
        activation='GLU',
        use_mask=True,
        use_temporal_embedding=True,
        use_spatial_embedding=True,
        output_hidden=128,
    ):
        super().__init__()
        prior_adjacency = self._load_prior_adjacency(prior_graph_path)
        self.register_buffer('prior_adjacency', torch.tensor(prior_adjacency, dtype=torch.float32))
        localized_adjacency = normalize_adjacency(build_localized_adjacency(prior_adjacency, steps=3))
        self.register_buffer('localized_adjacency', torch.tensor(localized_adjacency, dtype=torch.float32))
        self.adjacency_mask = (
            nn.Parameter(torch.tensor((localized_adjacency != 0).astype(np.float32)))
            if use_mask
            else None
        )
        self.input_projection = nn.Linear(in_dim, hidden_size)
        filter_list = [[hidden_size, hidden_size, hidden_size] for _ in range(layers)]
        self.graph_layers = nn.ModuleList()
        current_length = seq_len
        current_dim = hidden_size
        for filters in filter_list:
            self.graph_layers.append(
                SynchronousGraphLayer(
                    current_length,
                    num_nodes,
                    current_dim,
                    filters,
                    module_type=module_type,
                    activation=activation,
                    use_temporal_embedding=use_temporal_embedding,
                    use_spatial_embedding=use_spatial_embedding,
                )
            )
            current_length -= 2
            current_dim = filters[-1]
        self.forecast_projection = PriorGraphProjection(
            current_length,
            current_dim,
            hidden_dim=output_hidden,
            pred_len=pred_len,
        )
        if current_length != seq_len:
            self.routing_time_projection = nn.Linear(current_length, seq_len)
        else:
            self.routing_time_projection = nn.Identity()
        if current_dim != hidden_size:
            self.routing_feature_projection = nn.Linear(current_dim, hidden_size)
        else:
            self.routing_feature_projection = nn.Identity()
        for projection in (self.routing_time_projection, self.routing_feature_projection):
            if isinstance(projection, nn.Linear):
                nn.init.xavier_uniform_(projection.weight)
                nn.init.zeros_(projection.bias)

    def _load_prior_adjacency(self, prior_graph_path):
        _, _, adjacency = load_graph_data(prior_graph_path)
        adjacency = adjacency.astype(np.float32, copy=True)
        np.fill_diagonal(adjacency, 1.0)
        return adjacency

    def _effective_adjacency(self):
        if self.adjacency_mask is None:
            return self.localized_adjacency
        return self.localized_adjacency * self.adjacency_mask

    def _routing_hidden(self, x):
        hidden = x.permute(0, 2, 3, 1).contiguous()
        hidden = self.routing_time_projection(hidden)
        hidden = hidden.permute(0, 1, 3, 2).contiguous()
        hidden = self.routing_feature_projection(hidden)
        return hidden

    def forward(self, x):
        x = x.unsqueeze(-1)
        mean = x.mean(dim=1, keepdim=True).detach()
        standard_deviation = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-05
        ).detach()
        normalized = (x - mean) / standard_deviation
        x = F.relu(self.input_projection(normalized))
        adjacency = self._effective_adjacency().to(device=x.device, dtype=x.dtype)
        for layer in self.graph_layers:
            x = layer(x, adjacency)
        hidden = self._routing_hidden(x)
        forecast = self.forecast_projection(x).unsqueeze(-1)
        forecast = forecast * standard_deviation[:, 0:1, :, :] + mean[:, 0:1, :, :]
        forecast = forecast.permute(0, 2, 1, 3).contiguous()
        return forecast, hidden


class GraphPyramidEncoder(nn.Module):
    KERNELS_BY_NUM_NODES = {24: (5, 5, 5, 5, 5, 4)}

    def __init__(self, num_nodes):
        super().__init__()
        kernels = self.KERNELS_BY_NUM_NODES[num_nodes]
        channels = [1, 16] + [32] * (len(kernels) - 1)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        bias=False,
                    ),
                    nn.GroupNorm(4, out_channels, affine=False),
                    nn.GELU(),
                )
                for in_channels, out_channels, kernel_size in zip(
                    channels[:-1],
                    channels[1:],
                    kernels,
                )
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResponseAwareRouting(nn.Module):
    """
    Input shape
        - inputs: [B, N, T_in, 1]
        - expert_hidden_states: three tensors of [B, N, T_in, hidden_size]
    Output shape
        - response_scores: [B, N, T_in, 3]
    """
    def __init__(self, hidden_size, memory_dim=64, input_dim=1, memory_size=20):
        super().__init__()
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.memory = nn.Parameter(torch.empty(memory_size, memory_dim))
        self.expert_queries = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_size, memory_dim)) for _ in range(3)]
        )
        self.expert_keys = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_size, memory_dim)) for _ in range(3)]
        )
        self.expert_values = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_size, memory_dim)) for _ in range(3)]
        )
        self.input_query = nn.Parameter(torch.empty(input_dim, memory_dim))

    def forward(self, inputs, expert_hidden_states):
        memory_context = self._query_memory(inputs)
        response_scores = []
        for expert_index, hidden_state in enumerate(expert_hidden_states):
            expert_response = self._attention(hidden_state, expert_index)
            response_scores.append(self.similarity(memory_context, expert_response))
        return torch.stack(response_scores, dim=-1)

    def _attention(self, hidden_state, expert_index):
        query = torch.matmul(hidden_state, self.expert_queries[expert_index])
        key = torch.matmul(hidden_state, self.expert_keys[expert_index])
        value = torch.matmul(hidden_state, self.expert_values[expert_index])
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, value)

    def _query_memory(self, inputs):
        query = torch.matmul(inputs, self.input_query)
        memory_scores = torch.matmul(query, self.memory.T)
        memory_weights = torch.softmax(memory_scores, dim=-1)
        return torch.matmul(memory_weights, self.memory)


class StructureAwareRouting(nn.Module):
    """
    Input shape
        - prior_adjacency: [B, 1, N, N]
        - latent_adjacency: [B * T_in, 1, N, N]
    Output shape
        - structure_scores: [B, N, T_in, 3]
    """
    def __init__(self, num_nodes, seq_len, num_experts=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.score_projection = nn.Sequential(
            nn.LayerNorm(64, elementwise_affine=False),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, num_experts),
            nn.Tanh(),
        )
        self.prior_graph_encoder = GraphPyramidEncoder(num_nodes)
        self.latent_graph_encoder = GraphPyramidEncoder(num_nodes)

    def forward(self, prior_adjacency, latent_adjacency):
        batch_size = prior_adjacency.size(0)
        prior_features = self.prior_graph_encoder(prior_adjacency).flatten(1)
        latent_features = self.latent_graph_encoder(latent_adjacency).flatten(1)
        prior_features = prior_features.unsqueeze(1).expand(-1, self.seq_len, -1)
        latent_features = latent_features.view(batch_size, self.seq_len, -1)
        graph_features = torch.cat([prior_features, latent_features], dim=-1)
        structure_scores = self.score_projection(graph_features)
        return structure_scores.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)


class DualBranchRouter(nn.Module):
    """
    Input shape
        - inputs: [B, N, T_in, 1]
        - expert_hidden_states: [temporal_hidden, prior_hidden, latent_hidden]
        - temporal_hidden and prior_hidden: [B, N, T_in, hidden_size]
        - latent_hidden: [B, N, T_in, 176]
        - prior_adjacency: [B, 1, N, N]
        - latent_adjacency: [B * T_in, 1, N, N]
    Output shape
        - routing_scores: [B, N, T_out, output_dim, 3]
    """
    def __init__(
        self,
        hidden_size,
        expert_hidden_dims,
        num_nodes,
        memory_dim=64,
        input_dim=1,
        output_dim=1,
        seq_len=96,
        pred_len=96,
        memory_size=20,
        structure_weight=0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.structure_weight = structure_weight
        self.routing_hidden_projections = nn.ModuleList(
            [
                nn.Identity()
                if expert_dim == hidden_size
                else nn.Linear(expert_dim, hidden_size, bias=False)
                for expert_dim in expert_hidden_dims
            ]
        )
        self.response_aware_routing = ResponseAwareRouting(
            hidden_size, memory_dim=memory_dim, input_dim=input_dim, memory_size=memory_size
        )
        self.structure_aware_routing = StructureAwareRouting(num_nodes, seq_len)
        self.temporal_projection = nn.Linear(self.seq_len, self.pred_len)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)

    def forward(self, inputs, expert_hidden_states, prior_adjacency, latent_adjacency):
        batch_size, num_nodes, _, _ = inputs.size()
        routing_hidden_states = [
            projection(hidden_state)
            for projection, hidden_state in zip(
                self.routing_hidden_projections,
                expert_hidden_states,
            )
        ]
        response_scores = self.response_aware_routing(inputs, routing_hidden_states)
        structure_scores = self.structure_aware_routing(prior_adjacency, latent_adjacency)
        routing_scores = response_scores + self.structure_weight * structure_scores
        num_experts = routing_scores.size(-1)
        routing_scores = routing_scores.permute(0, 1, 3, 2).contiguous()
        routing_scores = routing_scores.view(-1, self.seq_len)
        routing_scores = self.temporal_projection(routing_scores)
        routing_scores = routing_scores.view(batch_size, num_nodes, num_experts, self.pred_len)
        routing_scores = routing_scores.permute(0, 1, 3, 2).contiguous()
        return routing_scores.unsqueeze(dim=-2).expand(
            batch_size, num_nodes, self.pred_len, self.output_dim, num_experts
        )


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        hidden_size = configs.d_model
        latent_model_dim = 24 + 24 + 24 + 24 + 80
        self.temporal_expert = TemporalExpert(configs)
        self.prior_graph_expert = PriorGraphExpert(
            hidden_size=hidden_size,
            num_nodes=configs.enc_in,
            prior_graph_path=configs.prior_graph_path,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            in_dim=1,
            layers=configs.prior_layers,
            module_type=configs.prior_module_type,
            activation=configs.prior_activation,
            use_mask=configs.prior_use_mask,
            use_temporal_embedding=configs.prior_temporal_embedding,
            use_spatial_embedding=configs.prior_spatial_embedding,
            output_hidden=configs.prior_projection_hidden,
        )

        self.steps_per_day = configs.steps_per_day
        self.latent_graph_expert = LatentGraphExpert(
            num_nodes=configs.enc_in,
            in_steps=configs.seq_len,
            out_steps=configs.pred_len,
            steps_per_day=self.steps_per_day,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            time_of_day_embedding_dim=24,
            day_of_week_embedding_dim=24,
            spatial_embedding_dim=24,
            adaptive_embedding_dim=80,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
        )
        self.dual_branch_router = DualBranchRouter(
            hidden_size=hidden_size,
            expert_hidden_dims=(hidden_size, hidden_size, latent_model_dim),
            num_nodes=configs.enc_in,
            memory_dim=hidden_size,
            input_dim=1,
            output_dim=1,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            structure_weight=configs.structure_weight,
        )

    def _build_latent_expert_input(self, x_enc, x_mark_enc):
        batch_size, input_steps, num_nodes = x_enc.shape
        observations = x_enc.unsqueeze(-1)
        time_features = x_mark_enc.to(device=x_enc.device, dtype=x_enc.dtype)
        hour_of_day = torch.round((time_features[..., 0] + 0.5) * 23).clamp(0, 23)
        day_of_week = torch.round((time_features[..., 1] + 0.5) * 6).clamp(0, 6)
        time_of_day = (hour_of_day / 24.0).unsqueeze(-1).unsqueeze(-1).expand(batch_size, input_steps, num_nodes, 1)
        day_of_week = day_of_week.unsqueeze(-1).unsqueeze(-1).expand(batch_size, input_steps, num_nodes, 1)
        return torch.cat([observations, time_of_day, day_of_week], dim=-1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Temporal expert
        temporal_forecast, temporal_hidden_states = self.temporal_expert(x_enc, x_mark_enc, x_dec, x_mark_dec)
        temporal_forecast = temporal_forecast.contiguous()
        temporal_hidden = temporal_hidden_states[-1]

        # Prior graph expert
        prior_forecast, prior_hidden = self.prior_graph_expert(x_enc)

        # Latent graph expert
        latent_input = self._build_latent_expert_input(x_enc, x_mark_enc)
        latent_forecast, latent_hidden_states, latent_adjacency = self.latent_graph_expert(latent_input)
        latent_forecast = latent_forecast.permute(0, 2, 1, 3)
        latent_hidden = latent_hidden_states[-1]

        # Dual-branch routing
        batch_size, num_nodes, input_steps, _ = latent_hidden.shape
        expert_hidden_states = [temporal_hidden, prior_hidden, latent_hidden]
        router_input = x_enc.permute(0, 2, 1).unsqueeze(-1)
        prior_adjacency = self.prior_graph_expert.prior_adjacency.to(x_enc.device)
        prior_adjacency = prior_adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        prior_adjacency = prior_adjacency.unsqueeze(1)
        latent_adjacency = latent_adjacency.contiguous().view(batch_size * input_steps, 1, num_nodes, num_nodes)
        routing_scores = self.dual_branch_router(router_input, expert_hidden_states, prior_adjacency, latent_adjacency)
        routing_weights = torch.softmax(routing_scores, dim=-1)

        # Gate-weighted MoE fusion
        expert_forecasts = torch.stack([temporal_forecast, prior_forecast, latent_forecast], dim=-1)
        fused_forecast = torch.sum(routing_weights * expert_forecasts, dim=-1)
        out = fused_forecast.permute(0, 2, 1, 3).squeeze(-1)
        return out, routing_weights
