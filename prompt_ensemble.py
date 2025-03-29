import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from SupFAD_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from open_clip import tokenizer
# simple_tokenizer = tokenizer.SimpleTokenizer()
from copy import deepcopy
import torch.nn as nn

_tokenizer = _Tokenizer()  # 初始化简单分词器

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    返回给定输入字符串的标记表示

    参数
    ----------
    texts : Union[str, List[str]]
        输入字符串或字符串列表

    context_length : int
        使用的上下文长度；所有CLIP模型使用77作为上下文长度

    truncate: bool
        如果编码长度超过上下文长度，是否截断文本

    返回
    -------
    一个二维张量，包含结果标记，形状 = [输入字符串数量, context_length]。
    当torch版本小于1.8.0时返回LongTensor，因为旧的index_select需要长整型索引。
    """
    if isinstance(texts, str):
        texts = [texts]  # 如果输入是字符串，则转换为列表

    sot_token = _tokenizer.encoder["<|startoftext|>"]  # 获取开始标记
    eot_token = _tokenizer.encoder["<|endoftext|>"]  # 获取结束标记
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]  # 编码文本并添加标记
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)  # 创建长整型张量
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)  # 创建整型张量

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]  # 截断到上下文长度
                tokens[-1] = eot_token  # 确保最后一个标记是结束标记
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")  # 抛出错误
        result[i, :len(tokens)] = torch.tensor(tokens)  # 将标记填充到结果张量中

    return result  # 返回结果张量

def encode_text_with_prompt_ensemble(model, texts, device):
    # 扩展织物相关的正常状态提示
    prompt_normal = [
        '{}',  # 基础模板
        'flawless {}', 'perfect {}', 'unblemished {}',
        '{} without flaw', '{} without defect', '{} without damage',
        # 添加织物特定的正常状态描述
        'well-woven {}', 'smooth {}', 'even-textured {}',
        'properly-dyed {}', 'uniform {}', 'consistent {}',
        'high-quality {}', 'properly-finished {}',
        '{} with consistent weave', '{} with uniform texture'
    ]
    
    # 扩展织物相关的异常状态提示
    prompt_abnormal = [
        'damaged {}', 'broken {}', 
        '{} with flaw', '{} with defect', '{} with damage',
        # 添加织物特定的异常状态描述
        'torn {}', 'stained {}', 'discolored {}',
        '{} with holes', '{} with runs', '{} with snags',
        '{} with loose threads', '{} with uneven dye',
        '{} with weaving defects', '{} with fabric pilling',
        '{} with broken yarns', '{} with color bleeding',
        '{} with misaligned patterns', '{} with fabric wrinkles',
        '{} with fabric contamination', '{} with fabric creases'
    ]
    prompt_state = [prompt_normal, prompt_abnormal]  # 将正常和异常提示组合
    # 扩展织物特定的模板表述
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of the {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.',
                        # 添加织物特定的描述模板
                        'a detailed view of the {} fabric.',
                        'a closeup of the {} textile surface.',
                        'a microscopic view of the {} fabric weave.',
                        'an inspection photo of the {} material.',
                        'a quality control image of the {} fabric.',
                        'a manufacturing sample of the {} textile.',
                        'a fabric inspection photo of the {}.',
                        'a textile quality assessment of the {}.',
                        'a production line image of the {} fabric.',
                        'a quality verification photo of the {} material.']

    text_features = []  # 初始化文本特征列表
    for i in range(len(prompt_state)):
        prompted_state = [state.format(texts[0]) for state in prompt_state[i]]  # 格式化提示状态
        prompted_sentence = []  # 初始化提示句子列表
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))  # 生成提示句子
        prompted_sentence = tokenize(prompted_sentence)  # 对提示句子进行标记化
        class_embeddings = model.encode_text(prompted_sentence.to(device))  # 编码文本
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # 归一化
        class_embedding = class_embeddings.mean(dim=0)  # 计算平均嵌入
        class_embedding /= class_embedding.norm()  # 归一化
        text_features.append(class_embedding)  # 添加到文本特征列表

    text_features = torch.stack(text_features, dim=1).to(device).t()  # 堆叠文本特征并转置

    return text_features  # 返回文本特征

def _get_clones(module, N):
    # 生成N个相同的模块副本
    return nn.ModuleList([deepcopy(module) for i in range(N)])  # 返回模块副本列表

class DynamicTextureAdapter(nn.Module):
    def __init__(self, img_dim, prompt_dim, hidden_dim=512):
        super().__init__()
        # 两层MLP用于生成纹理感知的提示嵌入
        self.mlp = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_dim)
        )

    def forward(self, cls_token):
        return self.mlp(cls_token)

class MultimodalPromptFusion(nn.Module):
    def __init__(self, text_dim, img_dim, fusion_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.img_proj = nn.Linear(img_dim, fusion_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU()
        )
        
    def forward(self, text_emb, img_emb):
        text_feat = self.text_proj(text_emb)
        img_feat = self.img_proj(img_emb)
        return self.fusion(text_feat + img_feat)

class DOAPrompts(nn.Module):
    def __init__(self, img_dim, prompt_dim):
        super().__init__()
        self.static_prompt = nn.Parameter(torch.randn(prompt_dim))
        self.dynamic_adapter = DynamicTextureAdapter(img_dim, prompt_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * prompt_dim, prompt_dim),
            nn.LayerNorm(prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, cls_token, target_shape=None):
        # 生成动态提示
        dynamic = self.dynamic_adapter(cls_token)
        static = self.static_prompt.unsqueeze(0).expand(cls_token.shape[0], -1)
        
        # 如果需要调整形状以匹配目标
        if target_shape is not None:
            if len(target_shape) == 3:  # [B, L, D]
                B, L, D = target_shape
                dynamic = dynamic.unsqueeze(1).expand(-1, L, -1)
                static = static.unsqueeze(1).expand(-1, L, -1)
        
        # 计算门控权重并融合
        combined = torch.cat([static, dynamic], dim=-1)
        gate_weight = self.gate(combined)
        return gate_weight * static + (1 - gate_weight) * dynamic

class SupFAD_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        classnames = ["object"]  # 类别名称
        self.n_cls = len(classnames)  # 类别数量
        self.n_ctx = design_details["Prompt_length"]  # 提示长度
        n_ctx_pos = self.n_ctx  # 正样本上下文长度
        n_ctx_neg = self.n_ctx  # 负样本上下文长度
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]  # 可学习文本嵌入长度
        ctx_init_pos = ""  # 正样本上下文初始化
        ctx_init_neg = ""  # 负样本上下文初始化
        dtype = clip_model.transformer.get_cast_dtype()  # 获取数据类型

        ctx_dim = clip_model.ln_final.weight.shape[0]  # 上下文维度

        self.classnames = classnames  # 保存类别名称

        self.state_normal_list = [
            "{}",
            "well-woven {}",
            "properly-manufactured {}",
            "quality-checked {}"
        ]  # 正常状态列表

        self.state_anomaly_list = [
            "damaged {}",
            "defective {}",
            "{} with manufacturing defect",
            "{} with weaving flaw",
            "{} with quality issues",
            "{} with fabric contamination"
        ]  # 异常状态列表
        
        normal_num = len(self.state_normal_list)  # 正常状态数量
        anormaly_num = len(self.state_anomaly_list)  # 异常状态数量
        self.normal_num = normal_num  # 保存正常状态数量
        self.anormaly_num = anormaly_num  # 保存异常状态数量

        if ctx_init_pos and ctx_init_neg:
            # 使用给定的单词初始化上下文向量
            ctx_init_pos = ctx_init_pos.replace("_", " ")  # 替换下划线为空格
            ctx_init_neg = ctx_init_neg.replace("_", " ")  # 替换下划线为空格
            n_ctx_pos = len(ctx_init_pos.split(" "))  # 正样本上下文长度
            n_ctx_neg = len(ctx_init_neg.split(" "))  # 负样本上下文长度
            # 初始化文本为bpd编码
            prompt_pos = tokenize(ctx_init_pos)  # 对正样本上下文进行标记化
            prompt_neg = tokenize(ctx_init_neg)  # 对负样本上下文进行标记化
            with torch.no_grad():
                # 生成相应的文本嵌入
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)  # 获取正样本嵌入
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)  # 获取负样本嵌入
            # 去除EOS和CLS，获得可学习的文本提示
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]  # 正样本上下文向量
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]  # 负样本上下文向量
            prompt_prefix_pos = ctx_init_pos  # 正样本提示前缀
            prompt_prefix_neg = ctx_init_neg  # 负样本提示前缀
            if True:
                ctx_vectors_pos_ = []  # 初始化正样本上下文向量列表
                ctx_vectors_neg_ = []  # 初始化负样本上下文向量列表
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))  # 复制正样本上下文向量
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))  # 复制负样本上下文向量
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)  # 堆叠正样本上下文向量
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)  # 堆叠负样本上下文向量

        else:
            # 随机初始化
            if True:
                # cls是类的个数，n_ctx_pos代表可学习token的长度，ctx_dim表示提示的维度
                ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)  # 创建正样本上下文向量
                ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)  # 创建负样本上下文向量
            else:  
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)  # 创建正样本上下文向量
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)  # 创建负样本上下文向量
            nn.init.normal_(ctx_vectors_pos, std=0.02)  # 正样本上下文向量初始化
            nn.init.normal_(ctx_vectors_neg, std=0.02)  # 负样本上下文向量初始化
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)  # 正样本提示前缀初始化
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)  # 负样本提示前缀初始化
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]  # 可学习文本嵌入深度
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])  # 创建可学习的文本嵌入参数列表
        for single_para in self.compound_prompts_text:                 
            nn.init.normal_(single_para, std=0.02)  # 初始化参数

        single_layer = nn.Linear(ctx_dim, 896)  # 定义线性层
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)  # 生成多个线性层的副本

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # 正样本上下文向量参数
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # 负样本上下文向量参数

        classnames = [name.replace("_", " ") for name in classnames]  # 替换类别名称中的下划线
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]  # 获取类别名称的编码长度

        prompts_pos = [prompt_prefix_pos +  " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames]  # 生成正样本提示
        prompts_neg = [prompt_prefix_neg +  " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames]  # 生成负样本提示

        tokenized_prompts_pos = []  # 初始化正样本标记提示列表
        tokenized_prompts_neg = []  # 初始化负样本标记提示列表
     
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))  # 对正样本提示进行标记化
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))  # 对负样本提示进行标记化
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)  # 合并正样本标记提示
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)  # 合并负样本标记提示
        # 生成相应的文本嵌入
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)  # 获取正样本嵌入
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)  # 获取负样本嵌入
            n, l, d = embedding_pos.shape  # 获取正样本嵌入的维度
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)  # 重塑正样本嵌入
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)  # 重塑负样本嵌入

        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])  # 注册正样本前缀嵌入
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx_pos:, :])  # 注册正样本后缀嵌入
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])  # 注册负样本前缀嵌入
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])  # 注册负样本后缀嵌入

        n, d = tokenized_prompts_pos.shape  # 获取正样本标记提示的维度
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)  # 重塑正样本标记提示

        n, d = tokenized_prompts_neg.shape  # 获取负样本标记提示的维度
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)  # 重塑负样本标记提示

        self.n_ctx_pos = n_ctx_pos  # 保存正样本上下文长度
        self.n_ctx_neg = n_ctx_neg  # 保存负样本上下文长度
        # tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)  # 合并标记提示
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)  # 注册正样本标记提示
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)  # 注册负样本标记提示

        # 添加新的组件
        self.multimodal_fusion = MultimodalPromptFusion(
            text_dim=ctx_dim,
            img_dim=clip_model.visual.output_dim,
            fusion_dim=ctx_dim
        )
        
        self.doa_prompts = DOAPrompts(
            img_dim=clip_model.visual.output_dim,
            prompt_dim=ctx_dim
        )
        
        # 添加提示组合器
        self.prompt_combiner = nn.Sequential(
            nn.Linear(ctx_dim * 2, ctx_dim),
            nn.LayerNorm(ctx_dim),
            nn.GELU()
        )
        
        # 添加注意力机制来学习重要的提示
        self.prompt_attention = nn.MultiheadAttention(
            embed_dim=clip_model.ln_final.weight.shape[0],
            num_heads=8,
            dropout=0.1
        )
        
        # 添加自适应权重
        self.prompt_weight = nn.Sequential(
            nn.Linear(clip_model.ln_final.weight.shape[0], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, cls_id=None, img_cls_token=None):
        # 获取原始静态提示和tokenized_prompts
        static_prompts, tokenized_prompts, compound_prompts = self.get_static_prompts(cls_id)
        
        # 首先将静态提示分为正常和异常两组
        total_length = static_prompts.shape[0]
        split_point = total_length // 2
        normal_prompts = static_prompts[:split_point]
        abnormal_prompts = static_prompts[split_point:]
        
        # 使用注意力机制处理提示
        normal_attn = self.prompt_attention(
            normal_prompts.transpose(0, 1),
            normal_prompts.transpose(0, 1),
            normal_prompts.transpose(0, 1)
        )[0].transpose(0, 1)
        
        abnormal_attn = self.prompt_attention(
            abnormal_prompts.transpose(0, 1),
            abnormal_prompts.transpose(0, 1),
            abnormal_prompts.transpose(0, 1)
        )[0].transpose(0, 1)
        
        # 计算自适应权重
        normal_weights = self.prompt_weight(normal_attn)
        abnormal_weights = self.prompt_weight(abnormal_attn)
        
        # 加权平均
        normal_embedding = (normal_attn * normal_weights).sum(dim=0, keepdim=True)
        abnormal_embedding = (abnormal_attn * abnormal_weights).sum(dim=0, keepdim=True)
        
        # 保持提示的多样性
        normal_embedding = normal_embedding + normal_prompts.std(dim=0, keepdim=True)
        abnormal_embedding = abnormal_embedding + abnormal_prompts.std(dim=0, keepdim=True)
        
        # 不再计算 tokenized_prompts 的平均值，而是使用第一个样本
        normal_tokens = tokenized_prompts[:split_point][0:1]  # 选择第一个正常样本 [1, L]
        abnormal_tokens = tokenized_prompts[split_point:][0:1]  # 选择第一个异常样本 [1, L]
        
        # 合并为最终的两类表示
        merged_prompts = torch.cat([normal_embedding, abnormal_embedding], dim=0)  # [2, L, D]
        merged_tokens = torch.cat([normal_tokens, abnormal_tokens], dim=0)  # [2, L]
        
        if img_cls_token is not None:
            B = img_cls_token.shape[0]
            
            # 扩展提示以匹配batch size
            expanded_prompts = merged_prompts.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, L, D]
            expanded_tokens = merged_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 2, L]
            
            # 生成动态提示
            dynamic_prompts = self.doa_prompts(
                img_cls_token, 
                target_shape=(B, merged_prompts.shape[1], merged_prompts.shape[2])
            )
            
            # 多模态特征融合
            fused_features = self.multimodal_fusion(
                expanded_prompts.reshape(-1, expanded_prompts.shape[-1]),
                img_cls_token.unsqueeze(1).expand(-1, expanded_prompts.shape[1] * expanded_prompts.shape[2], -1).reshape(-1, img_cls_token.shape[-1])
            ).reshape(expanded_prompts.shape)
            
            # 组合静态和动态提示
            combined_prompts = self.prompt_combiner(
                torch.cat([fused_features, dynamic_prompts.unsqueeze(1).expand(-1, 2, -1, -1)], dim=-1)
            )
            
            # 处理compound_prompts (如果需要)
            if compound_prompts is not None:
                new_compound_prompts = []
                for i in range(len(compound_prompts)):
                    proj_features = self.compound_prompt_projections[i](img_cls_token)
                    compound_shape = compound_prompts[i].shape
                    dynamic_feat = self.doa_prompts(proj_features, target_shape=(B, compound_shape[0], compound_shape[1]))
                    dynamic_feat = dynamic_feat.mean(dim=0)
                    new_compound_prompts.append(compound_prompts[i] + dynamic_feat)
                return combined_prompts, expanded_tokens, new_compound_prompts
            
            return combined_prompts, expanded_tokens, None
            
        return merged_prompts, merged_tokens, compound_prompts

    def get_static_prompts(self, cls_id=None):
        ctx_pos = self.ctx_pos  # 获取正样本上下文
        ctx_neg = self.ctx_neg  # 获取负样本上下文
        ctx_pos = self.ctx_pos  # 冗余获取正样本上下文
        ctx_neg = self.ctx_neg  # 冗余获取负样本上下文
        # print("shape", self.ctx_pos[0:1].shape, ctx_pos.shape)
        prefix_pos = self.token_prefix_pos  # 获取正样本前缀
        prefix_neg = self.token_prefix_neg  # 获取负样本前缀
        suffix_pos = self.token_suffix_pos  # 获取正样本后缀
        suffix_neg = self.token_suffix_neg  # 获取负样本后缀

        # print(prefix_pos.shape, prefix_neg.shape)

        prompts_pos = torch.cat(
            [
                # N(模板数量), 1, dim
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=2,
        )  # 合并正样本提示

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=2,
        )  # 合并负样本提示
        _, _, l, d = prompts_pos.shape  # 获取正样本提示的维度
        prompts_pos = prompts_pos.reshape(-1, l, d)  # 重塑正样本提示
        _, _, l, d = prompts_neg.shape  # 获取负样本提示的维度
        prompts_neg = prompts_neg.reshape(-1, l, d)  # 重塑负样本提示
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)  # 合并正负样本提示

        _, l, d = self.tokenized_prompts_pos.shape  # 获取正样本标记提示的维度
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, d)  # 重塑正样本标记提示
        _, l, d = self.tokenized_prompts_neg.shape  # 获取负样本标记提示的维度
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, d)  # 重塑负样本标记提示
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim=0)  # 合并标记提示

        # 确保输出的维度一致性
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0).float()  # [B, L, D]
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim=0)  # [B, L]
    
        
        return prompts, tokenized_prompts, self.compound_prompts_text  # 返回提示和标记提示