from __future__ import annotations

import collections,copy,glob,io,lzma,math,os
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
import random,re,subprocess,sys,time,uuid,numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor,nn
try:
	from flash_attn_interface import flash_attn_func as flash_attn_3_func
except Exception:
	flash_attn_3_func=None
class Hyperparameters:data_dir=os.environ.get('DATA_DIR','./data/');seed=int(os.environ.get('SEED',1337));run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));iterations=int(os.environ.get('ITERATIONS',20000));warmdown_frac=float(os.environ.get('WARMDOWN_FRAC',.72));warmup_steps=int(os.environ.get('WARMUP_STEPS',20));train_batch_tokens=int(os.environ.get('TRAIN_BATCH_TOKENS',786432));train_seq_len=int(os.environ.get('TRAIN_SEQ_LEN',2048));train_log_every=int(os.environ.get('TRAIN_LOG_EVERY',500));max_wallclock_seconds=float(os.environ.get('MAX_WALLCLOCK_SECONDS',6e2));val_batch_tokens=int(os.environ.get('VAL_BATCH_TOKENS',524288));eval_seq_len=int(os.environ.get('EVAL_SEQ_LEN',2048));val_loss_every=int(os.environ.get('VAL_LOSS_EVERY',4000));sliding_window_enabled=bool(int(os.environ.get('SLIDING_WINDOW_ENABLED','1')));vocab_size=int(os.environ.get('VOCAB_SIZE',8192));num_layers=int(os.environ.get('NUM_LAYERS',11));xsa_last_n=int(os.environ.get('XSA_LAST_N',11));model_dim=int(os.environ.get('MODEL_DIM',512));embedding_dim=int(os.environ.get('EMBEDDING_DIM',512));num_kv_heads=int(os.environ.get('NUM_KV_HEADS',4));num_heads=int(os.environ.get('NUM_HEADS',8));mlp_mult=float(os.environ.get('MLP_MULT',4.));skip_gates_enabled=bool(int(os.environ.get('SKIP_GATES_ENABLED','1')));tie_embeddings=bool(int(os.environ.get('TIE_EMBEDDINGS','1')));logit_softcap=float(os.environ.get('LOGIT_SOFTCAP',3e1));rope_base=float(os.environ.get('ROPE_BASE',1e4));rope_dims=int(os.environ.get('ROPE_DIMS',16));rope_train_seq_len=int(os.environ.get('ROPE_TRAIN_SEQ_LEN',2048));ln_scale=bool(int(os.environ.get('LN_SCALE','1')));qk_gain_init=float(os.environ.get('QK_GAIN_INIT',5.));num_loops=int(os.environ.get('NUM_LOOPS',2));loop_start=int(os.environ.get('LOOP_START',3));loop_end=int(os.environ.get('LOOP_END',5));enable_looping_at=float(os.environ.get('ENABLE_LOOPING_AT',.35));parallel_residual_start=int(os.environ.get('PARALLEL_RESIDUAL_START',7));min_lr=float(os.environ.get('MIN_LR',.0));embed_lr=float(os.environ.get('EMBED_LR',.6));head_lr=float(os.environ.get('HEAD_LR',.008));tied_embed_lr=float(os.environ.get('TIED_EMBED_LR',.03));tied_embed_init_std=float(os.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=float(os.environ.get('MATRIX_LR',.022));scalar_lr=float(os.environ.get('SCALAR_LR',.02));muon_momentum=float(os.environ.get('MUON_MOMENTUM',.99));muon_backend_steps=int(os.environ.get('MUON_BACKEND_STEPS',5));muon_momentum_warmup_start=float(os.environ.get('MUON_MOMENTUM_WARMUP_START',.92));muon_momentum_warmup_steps=int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS',1500));muon_row_normalize=bool(int(os.environ.get('MUON_ROW_NORMALIZE','1')));beta1=float(os.environ.get('BETA1',.9));beta2=float(os.environ.get('BETA2',.95));adam_eps=float(os.environ.get('ADAM_EPS',1e-08));grad_clip_norm=float(os.environ.get('GRAD_CLIP_NORM',.3));eval_stride=int(os.environ.get('EVAL_STRIDE',64));muon_beta2=float(os.environ.get('MUON_BETA2',.95));adam_wd=float(os.environ.get('ADAM_WD',.02));muon_wd=float(os.environ.get('MUON_WD',.095));embed_wd=float(os.environ.get('EMBED_WD',.085));ema_decay=float(os.environ.get('EMA_DECAY',.9965));ttt_enabled=bool(int(os.environ.get('TTT_ENABLED','0')));ttt_lr=float(os.environ.get('TTT_LR',.005));ttt_epochs=int(os.environ.get('TTT_EPOCHS',3));ttt_momentum=float(os.environ.get('TTT_MOMENTUM',.9));ttt_chunk_tokens=int(os.environ.get('TTT_CHUNK_TOKENS',32768));compressor=os.environ.get('COMPRESSOR','brotli');gptq_calibration_batches=int(os.environ.get('GPTQ_CALIBRATION_BATCHES',64));gptq_reserve_seconds=float(os.environ.get('GPTQ_RESERVE_SECONDS',12.));matrix_bits=int(os.environ.get('MATRIX_BITS',6));embed_bits=int(os.environ.get('EMBED_BITS',8));matrix_clip_sigmas=float(os.environ.get('MATRIX_CLIP_SIGMAS',12.85));embed_clip_sigmas=float(os.environ.get('EMBED_CLIP_SIGMAS',2e1));distributed='RANK'in os.environ and'WORLD_SIZE'in os.environ;rank=int(os.environ.get('RANK','0'));world_size=int(os.environ.get('WORLD_SIZE','1'));local_rank=int(os.environ.get('LOCAL_RANK','0'));is_main_process=rank==0;grad_accum_steps=8//world_size;datasets_dir=os.path.join(data_dir,'datasets',f"fineweb10B_sp{vocab_size}");train_files=os.path.join(datasets_dir,'fineweb_train_*.bin');val_files=os.path.join(datasets_dir,'fineweb_val_*.bin');tokenizer_path=os.path.join(data_dir,'tokenizers',f"fineweb_{vocab_size}_bpe.model");logfile=f"logs/{run_id}.txt";model_path='final_model.pt';quantized_model_path='final_model.int6.ptz'
_logger_hparams=None
def set_logging_hparams(h):global _logger_hparams;_logger_hparams=h
def log(msg,console=True):
	if _logger_hparams is None:print(msg);return
	if _logger_hparams.is_main_process:
		if console:print(msg)
		if _logger_hparams.logfile is not None:
			with open(_logger_hparams.logfile,'a',encoding='utf-8')as f:print(msg,file=f)
def require_fa3(device):
	if device.type!='cuda':raise RuntimeError(f"FA3 requires CUDA, got {device}")
	if flash_attn_3_func is None:raise RuntimeError('FA3 is unavailable because flash_attn_interface could not be imported')
	major,_=torch.cuda.get_device_capability(device)
	if major<9:raise RuntimeError(f"FA3 requires Hopper or newer, got device capability sm{major}")
def build_sentencepiece_luts(sp,vocab_size,device):
	sp_vocab_size=int(sp.vocab_size());assert sp.piece_to_id('▁')!=sp.unk_id(),"Tokenizer must have '▁' (space) as its own token for correct BPB byte counting";table_size=max(sp_vocab_size,vocab_size);base_bytes_np=np.zeros((table_size,),dtype=np.int16);has_leading_space_np=np.zeros((table_size,),dtype=np.bool_);is_boundary_token_np=np.ones((table_size,),dtype=np.bool_)
	for token_id in range(sp_vocab_size):
		if sp.is_control(token_id)or sp.is_unknown(token_id)or sp.is_unused(token_id):continue
		is_boundary_token_np[token_id]=False
		if sp.is_byte(token_id):base_bytes_np[token_id]=1;continue
		piece=sp.id_to_piece(token_id)
		if piece.startswith('▁'):has_leading_space_np[token_id]=True;piece=piece[1:]
		base_bytes_np[token_id]=len(piece.encode('utf-8'))
	return torch.tensor(base_bytes_np,dtype=torch.int16,device=device),torch.tensor(has_leading_space_np,dtype=torch.bool,device=device),torch.tensor(is_boundary_token_np,dtype=torch.bool,device=device)
class RMSNorm(nn.Module):
	def __init__(self,eps=None):super().__init__();self.eps=eps
	def forward(self,x):return F.rms_norm(x,(x.size(-1),),eps=self.eps)
class CastedLinear(nn.Linear):
	def forward(self,x):w=self.weight.to(x.dtype);bias=self.bias.to(x.dtype)if self.bias is not None else None;return F.linear(x,w,bias)
class Rotary(nn.Module):
	def __init__(self,dim,base=1e4,train_seq_len=1024,rope_dims=0):super().__init__();self.dim=dim;self.base=base;self.train_seq_len=train_seq_len;self.rope_dims=rope_dims if rope_dims>0 else dim;inv_freq=1./base**(torch.arange(0,self.rope_dims,2,dtype=torch.float32)/self.rope_dims);self.register_buffer('inv_freq',inv_freq,persistent=False);self._seq_len_cached=0;self._cos_cached=None;self._sin_cached=None
	def forward(self,seq_len,device,dtype):
		if self._cos_cached is None or self._sin_cached is None or self._seq_len_cached!=seq_len or self._cos_cached.device!=device:
			rd=self.rope_dims
			if seq_len>self.train_seq_len:scale=seq_len/self.train_seq_len;new_base=self.base*scale**(rd/(rd-2));inv_freq=1./new_base**(torch.arange(0,rd,2,dtype=torch.float32,device=device)/rd)
			else:inv_freq=self.inv_freq.to(device)
			t=torch.arange(seq_len,device=device,dtype=inv_freq.dtype);freqs=torch.outer(t,inv_freq);self._cos_cached=freqs.cos()[None,:,None,:];self._sin_cached=freqs.sin()[None,:,None,:];self._seq_len_cached=seq_len
		return self._cos_cached.to(dtype=dtype),self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x,cos,sin,rope_dims=0):
	if rope_dims>0 and rope_dims<x.size(-1):x_rope,x_pass=x[...,:rope_dims],x[...,rope_dims:];half=rope_dims//2;x1,x2=x_rope[...,:half],x_rope[...,half:];x_rope=torch.cat((x1*cos+x2*sin,x1*-sin+x2*cos),dim=-1);return torch.cat((x_rope,x_pass),dim=-1)
	half=x.size(-1)//2;x1,x2=x[...,:half],x[...,half:];return torch.cat((x1*cos+x2*sin,x1*-sin+x2*cos),dim=-1)
class CausalSelfAttention(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len,rope_dims=0):
		super().__init__()
		if dim%num_heads!=0:raise ValueError('model_dim must be divisible by num_heads')
		if num_heads%num_kv_heads!=0:raise ValueError('num_heads must be divisible by num_kv_heads')
		self.num_heads=num_heads;self.num_kv_heads=num_kv_heads;self.head_dim=dim//num_heads
		if rope_dims>0:
			if rope_dims>self.head_dim:raise ValueError(f"rope_dims={rope_dims} must be <= head_dim={self.head_dim}")
			if rope_dims%2!=0:raise ValueError(f"rope_dims={rope_dims} must be even")
			self.rope_dims=rope_dims
		else:
			if self.head_dim%2!=0:raise ValueError('head_dim must be even when using full-head RoPE')
			self.rope_dims=0
		kv_dim=self.num_kv_heads*self.head_dim;self.c_q=CastedLinear(dim,dim,bias=False);self.c_k=CastedLinear(dim,kv_dim,bias=False);self.c_v=CastedLinear(dim,kv_dim,bias=False);self.proj=CastedLinear(dim,dim,bias=False);self.proj._zero_init=True;self.q_gain=nn.Parameter(torch.full((num_heads,),qk_gain_init,dtype=torch.float32));self.rotary=Rotary(self.head_dim,base=rope_base,train_seq_len=train_seq_len,rope_dims=self.rope_dims);self.use_xsa=False
	def _xsa_efficient(self,y,v):B,T,H,D=y.shape;Hkv=v.size(-2);group=H//Hkv;y_g=y.reshape(B,T,Hkv,group,D);vn=F.normalize(v,dim=-1).unsqueeze(-2);proj=(y_g*vn).sum(dim=-1,keepdim=True)*vn;return(y_g-proj).reshape(B,T,H,D)
	def forward(self,x):
		bsz,seqlen,dim=x.shape;q=self.c_q(x).reshape(bsz,seqlen,self.num_heads,self.head_dim);k=self.c_k(x).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);v=self.c_v(x).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);q=F.rms_norm(q,(q.size(-1),));k=F.rms_norm(k,(k.size(-1),));cos,sin=self.rotary(seqlen,x.device,q.dtype);q=apply_rotary_emb(q,cos,sin,self.rope_dims);k=apply_rotary_emb(k,cos,sin,self.rope_dims);q=q*self.q_gain.to(dtype=q.dtype)[None,None,:,None];y=flash_attn_3_func(q,k,v,causal=True)
		if self.use_xsa:y=self._xsa_efficient(y,v)
		y=y.reshape(bsz,seqlen,dim);return self.proj(y)
class MLP(nn.Module):
	def __init__(self,dim,mlp_mult):super().__init__();hidden=int(mlp_mult*dim);self.fc=CastedLinear(dim,hidden,bias=False);self.proj=CastedLinear(hidden,dim,bias=False);self.proj._zero_init=True
	def forward(self,x):return self.proj(F.leaky_relu(self.fc(x),negative_slope=.5).square())
class Block(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len,rope_dims=0,layer_idx=0,ln_scale=False):super().__init__();self.attn_norm=RMSNorm();self.mlp_norm=RMSNorm();self.attn=CausalSelfAttention(dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len,rope_dims);self.mlp=MLP(dim,mlp_mult);self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float());self.ln_scale_factor=1./math.sqrt(layer_idx+1)if ln_scale else 1.;self.parallel=False
	def forward(self,x,x0):
		mix=self.resid_mix.to(dtype=x.dtype);x_in=mix[0][None,None,:]*x+mix[1][None,None,:]*x0;attn_out=self.attn(self.attn_norm(x_in)*self.ln_scale_factor)
		if self.parallel:mlp_out=self.mlp(self.mlp_norm(x_in)*self.ln_scale_factor);x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out+self.mlp_scale.to(dtype=x_in.dtype)[None,None,:]*mlp_out
		else:x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out;x_out=x_out+self.mlp_scale.to(dtype=x_out.dtype)[None,None,:]*self.mlp(self.mlp_norm(x_out)*self.ln_scale_factor)
		return x_out
class GPT(nn.Module):
	def __init__(self,h):
		super().__init__()
		if h.logit_softcap<=.0:raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
		self.tie_embeddings=h.tie_embeddings;self.tied_embed_init_std=h.tied_embed_init_std;self.logit_softcap=h.logit_softcap;self.tok_emb=nn.Embedding(h.vocab_size,h.embedding_dim)
		if h.embedding_dim!=h.model_dim:self.embed_proj=CastedLinear(h.embedding_dim,h.model_dim,bias=False);self.head_proj=CastedLinear(h.model_dim,h.embedding_dim,bias=False)
		else:self.embed_proj=None;self.head_proj=None
		self.num_encoder_layers=h.num_layers//2;self.num_decoder_layers=h.num_layers-self.num_encoder_layers;self.blocks=nn.ModuleList([Block(h.model_dim,h.num_heads,h.num_kv_heads,h.mlp_mult,h.rope_base,h.qk_gain_init,h.train_seq_len,h.rope_dims,layer_idx=i,ln_scale=h.ln_scale)for i in range(h.num_layers)])
		self.final_norm=RMSNorm();self.lm_head=None if h.tie_embeddings else CastedLinear(h.embedding_dim,h.vocab_size,bias=False)
		if self.lm_head is not None:self.lm_head._zero_init=True
		if h.xsa_last_n>0:
			for i in range(max(0,h.num_layers-h.xsa_last_n),h.num_layers):self.blocks[i].attn.use_xsa=True
		if h.parallel_residual_start>=0:
			for i in range(h.parallel_residual_start,h.num_layers):self.blocks[i].parallel=True
		self.looping_active=False
		if h.num_loops>0:
			loop_seg=list(range(h.loop_start,h.loop_end+1));all_indices=list(range(h.loop_start))
			for _ in range(h.num_loops+1):all_indices.extend(loop_seg)
			all_indices.extend(range(h.loop_end+1,h.num_layers));num_enc=len(all_indices)//2;self.encoder_indices=all_indices[:num_enc];self.decoder_indices=all_indices[num_enc:]
		else:self.encoder_indices=list(range(self.num_encoder_layers));self.decoder_indices=list(range(self.num_encoder_layers,h.num_layers))
		self.num_skip_weights=min(len(self.encoder_indices),len(self.decoder_indices));self.skip_weights=nn.Parameter(torch.ones(self.num_skip_weights,h.model_dim,dtype=torch.float32));self.skip_gates=nn.Parameter(torch.zeros(self.num_skip_weights,h.model_dim,dtype=torch.float32))if h.skip_gates_enabled else None;self._init_weights()
	def _init_weights(self):
		if self.tie_embeddings:nn.init.normal_(self.tok_emb.weight,mean=.0,std=self.tied_embed_init_std)
		for(name,module)in self.named_modules():
			if isinstance(module,nn.Linear):
				if getattr(module,'_zero_init',False):nn.init.zeros_(module.weight)
				elif module.weight.ndim==2 and module.weight.shape[0]>=64 and module.weight.shape[1]>=64:nn.init.orthogonal_(module.weight,gain=1.)
	def forward_logits(self,input_ids):
		x=self.tok_emb(input_ids);x=F.rms_norm(x,(x.size(-1),))
		if self.embed_proj is not None:x=self.embed_proj(x)
		x0=x;skips=[];enc_iter=self.encoder_indices if self.looping_active else range(self.num_encoder_layers);dec_iter=self.decoder_indices if self.looping_active else range(self.num_encoder_layers,self.num_encoder_layers+self.num_decoder_layers)
		for i in enc_iter:x=self.blocks[i](x,x0);skips.append(x)
		for(skip_idx,i)in enumerate(dec_iter):
			if skip_idx<self.num_skip_weights and skips:
				scaled_skip=self.skip_weights[skip_idx].to(dtype=x.dtype)[None,None,:]*skips.pop()
				if self.skip_gates is not None:g=torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None,None,:];x=torch.lerp(scaled_skip,x,g)
				else:x=x+scaled_skip
			x=self.blocks[i](x,x0)
		x=self.final_norm(x)
		if self.head_proj is not None:x=self.head_proj(x)
		if self.tie_embeddings:logits_proj=F.linear(x,self.tok_emb.weight)
		else:logits_proj=self.lm_head(x)
		return self.logit_softcap*torch.tanh(logits_proj/self.logit_softcap)
	def forward(self,input_ids,target_ids):logits=self.forward_logits(input_ids);return F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),target_ids.reshape(-1),reduction='mean')
def classify_param(name):
	if'tok_emb'in name or'lm_head'in name:return'embed'
	if'.mlp.'in name:return'mlp'
	if'.attn.'in name or'.proj.'in name and'.mlp.'not in name:return'attn'
	return'other'
@torch.compile
def zeropower_via_newtonschulz5(G,steps=10,eps=1e-07):
	a,b,c=3.4445,-4.775,2.0315;X=G.bfloat16();X/=X.norm()+eps;transposed=G.size(0)>G.size(1)
	if transposed:X=X.T
	for _ in range(steps):A=X@X.T;B=b*A+c*A@A;X=a*X+B@X
	return X.T if transposed else X
class Muon(torch.optim.Optimizer):
	def __init__(self,params,lr,momentum,backend_steps,nesterov=True,weight_decay=.0,row_normalize=False):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay,row_normalize=row_normalize))
	@torch.no_grad()
	def step(self,closure=None):
		loss=None
		if closure is not None:
			with torch.enable_grad():loss=closure()
		distributed=dist.is_available()and dist.is_initialized();world_size=dist.get_world_size()if distributed else 1;rank=dist.get_rank()if distributed else 0
		for group in self.param_groups:
			params=group['params']
			if not params:continue
			lr=group['lr'];momentum=group['momentum'];backend_steps=group['backend_steps'];nesterov=group['nesterov'];total_params=sum(int(p.numel())for p in params);updates_flat=torch.zeros(total_params,device=params[0].device,dtype=torch.bfloat16);curr=0
			for(i,p)in enumerate(params):
				if i%world_size==rank and p.grad is not None:
					g=p.grad;state=self.state[p]
					if'momentum_buffer'not in state:state['momentum_buffer']=torch.zeros_like(g)
					buf=state['momentum_buffer'];buf.mul_(momentum).add_(g)
					if nesterov:g=g.add(buf,alpha=momentum)
					if group.get('row_normalize',False):row_norms=g.float().norm(dim=-1,keepdim=True).clamp_min(1e-07);g=g/row_norms.to(g.dtype)
					g=zeropower_via_newtonschulz5(g,steps=backend_steps);g*=max(1,g.size(0)/g.size(1))**.5;updates_flat[curr:curr+p.numel()]=g.reshape(-1)
				curr+=p.numel()
			if distributed:dist.all_reduce(updates_flat,op=dist.ReduceOp.SUM)
			wd=group.get('weight_decay',.0);curr=0
			for p in params:
				if wd>.0:p.data.mul_(1.-lr*wd)
				g=updates_flat[curr:curr+p.numel()].view_as(p).to(dtype=p.dtype);p.add_(g,alpha=-lr);curr+=p.numel()
		return loss
CONTROL_TENSOR_NAME_PATTERNS=tuple(pattern for pattern in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates').split(',')if pattern)
class Optimizers:
	def __init__(self,h,base_model):
		block_named_params=list(base_model.blocks.named_parameters());matrix_params=[p for(name,p)in block_named_params if p.ndim==2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)];scalar_params=[p for(name,p)in block_named_params if p.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]
		if base_model.skip_weights.numel()>0:scalar_params.append(base_model.skip_weights)
		if base_model.skip_gates is not None and base_model.skip_gates.numel()>0:scalar_params.append(base_model.skip_gates)
		token_lr=h.tied_embed_lr if h.tie_embeddings else h.embed_lr;tok_params=[{'params':[base_model.tok_emb.weight],'lr':token_lr,'base_lr':token_lr}];self.optimizer_tok=torch.optim.AdamW(tok_params,betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.embed_wd,fused=True);self.optimizer_muon=Muon(matrix_params,lr=h.matrix_lr,momentum=h.muon_momentum,backend_steps=h.muon_backend_steps,weight_decay=h.muon_wd,row_normalize=h.muon_row_normalize)
		for group in self.optimizer_muon.param_groups:group['base_lr']=h.matrix_lr
		self.optimizer_scalar=torch.optim.AdamW([{'params':scalar_params,'lr':h.scalar_lr,'base_lr':h.scalar_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.adam_wd,fused=True);self.optimizers=[self.optimizer_tok,self.optimizer_muon,self.optimizer_scalar]
		if base_model.lm_head is not None:self.optimizer_head=torch.optim.Adam([{'params':[base_model.lm_head.weight],'lr':h.head_lr,'base_lr':h.head_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,fused=True);self.optimizers.insert(1,self.optimizer_head)
		else:self.optimizer_head=None
	def __iter__(self):return iter(self.optimizers)
	def zero_grad_all(self):
		for opt in self.optimizers:opt.zero_grad(set_to_none=True)
	def step(self):
		for opt in self.optimizers:opt.step()
		self.zero_grad_all()
def restore_fp32_params(model):
	for module in model.modules():
		if isinstance(module,CastedLinear):module.float()
	for(name,param)in model.named_parameters():
		if(param.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))and param.dtype!=torch.float32:param.data=param.data.float()
def collect_hessians(model,train_loader,h,device,n_calibration_batches=64):
	hessians={};hooks=[]
	def _accumulate(name,x):
		if x.ndim==3:x=x.reshape(-1,x.shape[-1])
		if name not in hessians:hessians[name]=torch.zeros(x.shape[1],x.shape[1],dtype=torch.float32,device=device)
		hessians[name].addmm_(x.T,x)
	def make_hook(name):
		def hook_fn(module,inp,out):_accumulate(name,inp[0].detach().float())
		return hook_fn
	for(name,module)in model.named_modules():
		if isinstance(module,CastedLinear)and module.weight.numel()>65536:
			cat=classify_param(name+'.weight')
			if cat in('mlp','attn'):hooks.append(module.register_forward_hook(make_hook(name+'.weight')))
	if model.tie_embeddings:
		hook_module=model.head_proj if model.head_proj is not None else model.final_norm
		def make_output_hook(name):
			def hook_fn(module,inp,out):_accumulate(name,out.detach().float())
			return hook_fn
		hooks.append(hook_module.register_forward_hook(make_output_hook('tok_emb.weight')))
	model.eval()
	with torch.no_grad():
		for _ in range(n_calibration_batches):
			x,_=train_loader.next_batch(h.train_batch_tokens,h.grad_accum_steps);model.forward_logits(x)
	for hook in hooks:hook.remove()
	if dist.is_available()and dist.is_initialized()and h.world_size>1:
		for tensor in hessians.values():dist.all_reduce(tensor,op=dist.ReduceOp.SUM)
		norm=float(n_calibration_batches*h.world_size)
	else:norm=float(n_calibration_batches)
	for name in hessians:hessians[name]=hessians[name].cpu()/norm
	return hessians
def gptq_quantize_weight(w,H,clip_sigmas=3.,clip_range=63,block_size=128):
	W_orig=w.float().clone();rows,cols=W_orig.shape;H=H.float().clone();dead=torch.diag(H)==0;H[dead,dead]=1;damp=.01*H.diag().mean();H.diagonal().add_(damp);perm=torch.argsort(H.diag(),descending=True);invperm=torch.argsort(perm);W_perm=W_orig[:,perm].clone();W_perm[:,dead[perm]]=0;H=H[perm][:,perm];Hinv=torch.cholesky_inverse(torch.linalg.cholesky(H));Hinv=torch.linalg.cholesky(Hinv,upper=True);row_std=W_orig.std(dim=1);s=(clip_sigmas*row_std/clip_range).clamp_min(1e-10).to(torch.float16);sf=s.float();Q=torch.zeros(rows,cols,dtype=torch.int8);W_work=W_perm.clone()
	for i1 in range(0,cols,block_size):
		i2=min(i1+block_size,cols);W_block=W_work[:,i1:i2].clone();Hinv_block=Hinv[i1:i2,i1:i2];Err=torch.zeros(rows,i2-i1)
		for j in range(i2-i1):w_col=W_block[:,j];d=Hinv_block[j,j];q_col=torch.clamp(torch.round(w_col/sf),-clip_range,clip_range);Q[:,i1+j]=q_col.to(torch.int8);err=(w_col-q_col.float()*sf)/d;Err[:,j]=err;W_block[:,j:]-=err.unsqueeze(1)*Hinv_block[j,j:].unsqueeze(0)
		if i2<cols:W_work[:,i2:]-=Err@Hinv[i1:i2,i2:]
	return Q[:,invperm],s
def gptq_mixed_quantize(state_dict,hessians,h):
	result={};meta={}
	for(name,tensor)in state_dict.items():
		t=tensor.detach().cpu().contiguous()
		if not t.is_floating_point()or t.numel()<=65536:result[name]=t.to(torch.float16)if t.is_floating_point()else t;meta[name]='passthrough (float16)';continue
		cs=h.embed_clip_sigmas if'tok_emb'in name else h.matrix_clip_sigmas;bits=h.embed_bits if'tok_emb'in name else h.matrix_bits;q,s=gptq_quantize_weight(t,hessians[name],clip_sigmas=cs,clip_range=2**(bits-1)-1);result[name+'.q']=q;result[name+'.scale']=s;meta[name]=f"gptq (int{bits})"
	categories=collections.defaultdict(set)
	for(name,cat)in meta.items():short=re.sub('\\.\\d+$','',re.sub('blocks\\.\\d+','blocks',name));categories[cat].add(short)
	log('Quantized weights:')
	for cat in sorted(categories):log(f"  {cat}: {", ".join(sorted(categories[cat]))}")
	return result,meta
def dequantize_mixed(result,meta,template_sd):
	out={}
	for(name,orig)in template_sd.items():
		info=meta.get(name)
		if info is None:continue
		orig_dtype=orig.dtype
		if'passthrough'in info:
			t=result[name]
			if t.dtype==torch.float16 and orig_dtype in(torch.float32,torch.bfloat16):t=t.to(orig_dtype)
			out[name]=t;continue
		q,s=result[name+'.q'],result[name+'.scale']
		if s.ndim>0:out[name]=(q.float()*s.float().view(q.shape[0],*[1]*(q.ndim-1))).to(orig_dtype)
		else:out[name]=(q.float()*float(s.item())).to(orig_dtype)
	return out
_BSHF_MAGIC=b'BSHF'
def _byte_shuffle(data,stride=2):
	if stride<=1 or len(data)<stride:return data
	src=np.frombuffer(data,dtype=np.uint8);n=len(src);out=np.empty(n,dtype=np.uint8);dest_off=0
	for pos in range(stride):chunk=src[pos::stride];out[dest_off:dest_off+len(chunk)]=chunk;dest_off+=len(chunk)
	return _BSHF_MAGIC+bytes([stride])+out.tobytes()
def _byte_unshuffle(data):
	if len(data)<5 or data[:4]!=_BSHF_MAGIC:return data
	stride=data[4]
	if stride<2:return data[5:]
	payload=np.frombuffer(data,dtype=np.uint8,offset=5);n=len(payload);out=np.empty(n,dtype=np.uint8);src_off=0
	for pos in range(stride):chunk_len=n//stride+(1 if pos<n%stride else 0);out[pos::stride][:chunk_len]=payload[src_off:src_off+chunk_len];src_off+=chunk_len
	return out.tobytes()
def _compress(data,compressor):
	data=_byte_shuffle(data)
	if compressor=='lzma':return lzma.compress(data,preset=6)
	elif compressor=='brotli':import brotli;return brotli.compress(data,quality=11)
	raise ValueError(f"Unknown compressor: {compressor!r}")
def _decompress(data,compressor):
	if compressor=='lzma':raw=lzma.decompress(data)
	elif compressor=='brotli':import brotli;raw=brotli.decompress(data)
	else:raise ValueError(f"Unknown compressor: {compressor!r}")
	raw=_byte_unshuffle(raw);return raw
def serialize(h,base_model,code):
	code_bytes=len(code.encode('utf-8'))
	if h.is_main_process:torch.save(base_model.state_dict(),h.model_path);model_bytes=os.path.getsize(h.model_path);log(f"Serialized model: {model_bytes} bytes");log(f"Code size: {code_bytes} bytes")
	sd_cpu={k:v.detach().cpu()for(k,v)in base_model.state_dict().items()};device=torch.device('cuda',h.local_rank);log('GPTQ:collecting Hessians from calibration data...');t0=time.perf_counter();calib_loader=ShuffledSequenceLoader(h,device);hessians=collect_hessians(base_model,calib_loader,h,device,n_calibration_batches=h.gptq_calibration_batches);log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s");quant_result,quant_meta=gptq_mixed_quantize(sd_cpu,hessians,h);quant_buf=io.BytesIO();torch.save({'w':quant_result,'m':quant_meta},quant_buf);quant_raw=quant_buf.getvalue();quant_blob=_compress(quant_raw,h.compressor);quant_file_bytes=len(quant_blob);bytes_total=quant_file_bytes+code_bytes
	if h.is_main_process:
		with open(h.quantized_model_path,'wb')as f:f.write(quant_blob)
		log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes");log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
	return bytes_total,quant_file_bytes
def deserialize(h,device):
	eval_model=GPT(h).to(device).bfloat16();restore_fp32_params(eval_model);sd_cpu={k:v.detach().cpu()for(k,v)in eval_model.state_dict().items()}
	with open(h.quantized_model_path,'rb')as f:quant_blob_disk=f.read()
	quant_state=torch.load(io.BytesIO(_decompress(quant_blob_disk,h.compressor)),map_location='cpu');deq_state=dequantize_mixed(quant_state['w'],quant_state['m'],sd_cpu);eval_model.load_state_dict(deq_state,strict=True);return eval_model
def _loss_bpb(loss_sum,token_count,byte_count):val_loss=(loss_sum/token_count).item();val_bpb=val_loss/math.log(2.)*(token_count.item()/byte_count.item());return val_loss,val_bpb
def timed_eval(label,fn,*args,**kwargs):torch.cuda.synchronize();t0=time.perf_counter();val_loss,val_bpb=fn(*args,**kwargs);torch.cuda.synchronize();elapsed_ms=1e3*(time.perf_counter()-t0);log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms");return val_loss,val_bpb
def train_model(h,device,val_data):
	base_model=GPT(h).to(device).bfloat16();restore_fp32_params(base_model);compiled_model=torch.compile(base_model,dynamic=False,fullgraph=True)
	if h.distributed:model=DDP(compiled_model,device_ids=[h.local_rank],broadcast_buffers=False)
	else:model=compiled_model
	log(f"model_params:{sum(p.numel()for p in base_model.parameters())}");optimizers=Optimizers(h,base_model);train_loader=ShuffledSequenceLoader(h,device);max_wallclock_ms=1e3*h.max_wallclock_seconds if h.max_wallclock_seconds>0 else None
	if max_wallclock_ms is not None:max_wallclock_ms-=h.gptq_reserve_seconds*1e3;log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")
	def training_frac(step,elapsed_ms):
		if max_wallclock_ms is None:return step/max(h.iterations,1)
		return elapsed_ms/max(max_wallclock_ms,1e-09)
	def lr_mul(frac):
		if h.warmdown_frac<=0:return 1.
		if frac>=1.-h.warmdown_frac:return max((1.-frac)/h.warmdown_frac,h.min_lr)
		return 1.
	def step_fn(step,lr_scale):
		optimizers.zero_grad_all();train_loss=torch.zeros((),device=device)
		for micro_step in range(h.grad_accum_steps):
			if h.distributed:model.require_backward_grad_sync=micro_step==h.grad_accum_steps-1
			x,y=train_loader.next_batch(h.train_batch_tokens,h.grad_accum_steps)
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):loss=model(x,y)
			train_loss+=loss.detach();(loss/h.grad_accum_steps).backward()
		train_loss/=h.grad_accum_steps;frac=min(step/h.muon_momentum_warmup_steps,1.)if h.muon_momentum_warmup_steps>0 else 1.;muon_momentum=(1-frac)*h.muon_momentum_warmup_start+frac*h.muon_momentum
		for group in optimizers.optimizer_muon.param_groups:group['momentum']=muon_momentum
		for opt in optimizers:
			for group in opt.param_groups:group['lr']=group['base_lr']*lr_scale
		if h.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(base_model.parameters(),h.grad_clip_norm)
		optimizers.step();return train_loss
	if h.warmup_steps>0:
		initial_model_state={name:tensor.detach().cpu().clone()for(name,tensor)in base_model.state_dict().items()};initial_optimizer_states=[copy.deepcopy(opt.state_dict())for opt in optimizers];model.train()
		for warmup_step in range(h.warmup_steps):
			step_fn(warmup_step,1.)
			if warmup_step<=5 or(warmup_step+1)%10==0 or warmup_step+1==h.warmup_steps:log(f"warmup_step: {warmup_step+1}/{h.warmup_steps}")
		if h.num_loops>0:
			base_model.looping_active=True;log(f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
			for warmup_step in range(h.warmup_steps):
				step_fn(warmup_step,1.)
				if warmup_step<=5 or(warmup_step+1)%10==0 or warmup_step+1==h.warmup_steps:log(f"loop_warmup_step: {warmup_step+1}/{h.warmup_steps}")
			base_model.looping_active=False
		base_model.load_state_dict(initial_model_state,strict=True)
		for(opt,state)in zip(optimizers,initial_optimizer_states,strict=True):opt.load_state_dict(state)
		optimizers.zero_grad_all()
		if h.distributed:model.require_backward_grad_sync=True
		train_loader=ShuffledSequenceLoader(h,device)
	ema_state={name:t.detach().float().clone()for(name,t)in base_model.state_dict().items()};ema_decay=h.ema_decay;training_time_ms=.0;stop_after_step=None;torch.cuda.synchronize();t0=time.perf_counter();step=0
	while True:
		last_step=step==h.iterations or stop_after_step is not None and step>=stop_after_step;should_validate=last_step or h.val_loss_every>0 and step%h.val_loss_every==0
		if should_validate:torch.cuda.synchronize();training_time_ms+=1e3*(time.perf_counter()-t0);val_loss,val_bpb=eval_val(h,device,val_data,model);log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}");torch.cuda.synchronize();t0=time.perf_counter()
		if last_step:
			if stop_after_step is not None and step<h.iterations:log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}")
			break
		elapsed_ms=training_time_ms+1e3*(time.perf_counter()-t0);frac=training_frac(step,elapsed_ms);scale=lr_mul(frac)
		if h.num_loops>0 and not base_model.looping_active and frac>=h.enable_looping_at:base_model.looping_active=True;log(f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
		train_loss=step_fn(step,scale)
		with torch.no_grad():
			for(name,t)in base_model.state_dict().items():ema_state[name].mul_(ema_decay).add_(t.detach().float(),alpha=1.-ema_decay)
		step+=1;approx_training_time_ms=training_time_ms+1e3*(time.perf_counter()-t0);should_log_train=h.train_log_every>0 and(step<=5 or step%h.train_log_every==0 or stop_after_step is not None)
		if should_log_train:tok_per_sec=step*h.train_batch_tokens/(approx_training_time_ms/1e3);log(f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}")
		reached_cap=max_wallclock_ms is not None and approx_training_time_ms>=max_wallclock_ms
		if h.distributed and max_wallclock_ms is not None:reached_cap_tensor=torch.tensor(int(reached_cap),device=device);dist.all_reduce(reached_cap_tensor,op=dist.ReduceOp.MAX);reached_cap=bool(reached_cap_tensor.item())
		if stop_after_step is None and reached_cap:stop_after_step=step
	log(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")
	log('ema:applying EMA weights')
	current_state=base_model.state_dict()
	avg_state={name:t.to(dtype=current_state[name].dtype)for(name,t)in ema_state.items()}
	base_model.load_state_dict(avg_state,strict=True)
	return base_model,compiled_model
def main():
	world_size=int(os.environ.get('WORLD_SIZE','1'));local_rank=int(os.environ.get('LOCAL_RANK','0'));distributed='RANK'in os.environ and'WORLD_SIZE'in os.environ
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	if world_size<=0:raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
	if 8%world_size!=0:raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
	device=torch.device('cuda',local_rank);torch.cuda.set_device(device)
	require_fa3(device)
	if distributed:dist.init_process_group(backend='nccl',device_id=device);dist.barrier()
	torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('high');torch._dynamo.config.optimize_ddp=False;h=Hyperparameters();set_logging_hparams(h)
	if h.is_main_process:
		os.makedirs('logs',exist_ok=True);log(100*'=',console=False);log('Hyperparameters:',console=True)
		for(k,v)in sorted(vars(type(h)).items()):
			if not k.startswith('_'):log(f"  {k}: {v}",console=True)
		log('='*100,console=False)
		log(f"Running Python {sys.version}",console=False)
		log(f"Running PyTorch {torch.__version__}",console=False)
		log("attention_backend: fa3",console=True)
		log(subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False).stdout,console=False)
		log('='*100,console=False)
	train_and_eval(h,device)
	if distributed:dist.destroy_process_group()


HEADER_BYTES = 256 * np.dtype("<i4").itemsize
IGNORE_INDEX = -100


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


Hyperparameters.shuffle_docs = env_bool("SHUFFLE_DOCS", True)
Hyperparameters.train_context_burnin = int(os.environ.get("TRAIN_CONTEXT_BURNIN", "64"))
Hyperparameters.train_token_limit = int(os.environ.get("TRAIN_TOKEN_LIMIT", "0"))
Hyperparameters.val_token_limit = int(os.environ.get("VAL_TOKEN_LIMIT", "0"))
Hyperparameters.train_doc_limit = int(os.environ.get("TRAIN_DOC_LIMIT", "0"))
Hyperparameters.val_doc_limit = int(os.environ.get("VAL_DOC_LIMIT", "0"))


@dataclass(frozen=True)
class ShardSpec:
    path: str
    global_start: int
    num_tokens: int


@dataclass(frozen=True)
class DocSpan:
    doc_id: int
    start: int
    stop: int
    has_closing_bos: bool

    @property
    def raw_len(self) -> int:
        return self.stop - self.start

    @property
    def pred_len(self) -> int:
        return self.raw_len - 1


@dataclass(frozen=True)
class DocChunk:
    chunk_id: int
    doc_indices: tuple[int, ...]
    pred_tokens: int


@dataclass(frozen=True)
class ScoreWindowRef:
    doc_index: int
    start: int
    valid_len: int
    score_from: int
    score_count: int


@dataclass(frozen=True)
class TrainWindowRef:
    doc_index: int
    start: int
    valid_len: int


def shard_token_count(path: Path) -> int:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    return int(header[2])


def build_shard_specs(pattern: str) -> tuple[list[ShardSpec], int]:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}")
    specs: list[ShardSpec] = []
    total = 0
    for path in files:
        count = shard_token_count(path)
        specs.append(ShardSpec(str(path), total, count))
        total += count
    if total < 2:
        raise ValueError(f"Need at least two tokens from {pattern}")
    return specs, total


class MemmapTokenStore:
    def __init__(self, specs: list[ShardSpec]):
        self.specs = specs
        self.starts = [spec.global_start for spec in specs]
        self.ends = [spec.global_start + spec.num_tokens for spec in specs]
        self.total_tokens = self.ends[-1] if self.ends else 0
        self.arrays = [
            np.memmap(spec.path, dtype="<u2", mode="r", offset=HEADER_BYTES, shape=(spec.num_tokens,))
            for spec in specs
        ]

    def read_span(self, start: int, end: int) -> np.ndarray:
        if not (0 <= start < end <= self.total_tokens):
            raise ValueError(f"Invalid span [{start}, {end}) for total_tokens={self.total_tokens}")
        out = np.empty((end - start,), dtype=np.uint16)
        cursor = 0
        shard_idx = max(bisect_right(self.starts, start) - 1, 0)
        pos = start
        while pos < end:
            shard_start = self.starts[shard_idx]
            shard_end = self.ends[shard_idx]
            take = min(end, shard_end) - pos
            local_start = pos - shard_start
            out[cursor : cursor + take] = self.arrays[shard_idx][local_start : local_start + take]
            pos += take
            cursor += take
            shard_idx += 1
        return out


def scan_bos_positions(specs: list[ShardSpec], bos_id: int) -> np.ndarray:
    starts: list[int] = []
    for spec in specs:
        arr = np.memmap(spec.path, dtype="<u2", mode="r", offset=HEADER_BYTES, shape=(spec.num_tokens,))
        rel = np.flatnonzero(arr == bos_id)
        if rel.size:
            starts.extend((spec.global_start + rel.astype(np.int64)).tolist())
    starts = sorted(set(int(x) for x in starts if x >= 0))
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    return np.asarray(starts, dtype=np.int64)


def build_doc_spans(
    specs: list[ShardSpec],
    bos_id: int,
    doc_limit: int,
    token_limit: int,
    keep_last_open_doc: bool,
) -> list[DocSpan]:
    bos_positions = scan_bos_positions(specs, bos_id)
    total_tokens = specs[-1].global_start + specs[-1].num_tokens
    spans: list[DocSpan] = []
    for doc_id, start_pos in enumerate(bos_positions.tolist()):
        if token_limit > 0 and start_pos >= token_limit:
            break
        if doc_limit > 0 and len(spans) >= doc_limit:
            break
        if doc_id + 1 < len(bos_positions):
            stop = min(int(bos_positions[doc_id + 1]) + 1, total_tokens)
            has_closing_bos = True
        else:
            if not keep_last_open_doc:
                break
            stop = total_tokens
            has_closing_bos = False
        if stop - start_pos >= 2:
            spans.append(DocSpan(doc_id=doc_id, start=int(start_pos), stop=int(stop), has_closing_bos=has_closing_bos))
    return spans


def build_doc_chunks(doc_spans: list[DocSpan], target_pred_tokens: int) -> list[DocChunk]:
    target_pred_tokens = max(int(target_pred_tokens), 1)
    chunks: list[DocChunk] = []
    current_docs: list[int] = []
    current_pred_tokens = 0
    for doc_index, span in enumerate(doc_spans):
        pred_len = span.pred_len
        if pred_len <= 0:
            continue
        if current_docs and current_pred_tokens + pred_len > target_pred_tokens:
            chunks.append(
                DocChunk(
                    chunk_id=len(chunks),
                    doc_indices=tuple(current_docs),
                    pred_tokens=current_pred_tokens,
                )
            )
            current_docs = []
            current_pred_tokens = 0
        current_docs.append(doc_index)
        current_pred_tokens += pred_len
    if current_docs:
        chunks.append(
            DocChunk(
                chunk_id=len(chunks),
                doc_indices=tuple(current_docs),
                pred_tokens=current_pred_tokens,
            )
        )
    return chunks


def phase_for_doc(doc_id: int, seed: int, epoch: int) -> int:
    base_seed = ((doc_id * 0x9E3779B1) ^ seed) & 0xFFFFFFFF
    return int((base_seed + epoch) & 1)


def train_window_starts(pred_len: int, seq_len: int, doc_id: int, seed: int, epoch: int) -> list[int]:
    if pred_len <= seq_len:
        return [0]
    phase = phase_for_doc(doc_id, seed, epoch)
    if pred_len <= 2 * seq_len:
        return [0] if phase == 0 else [pred_len - seq_len]
    num_windows = max(pred_len // seq_len, 1)
    phase_offset = 0 if phase == 0 else pred_len - num_windows * seq_len
    return [phase_offset + idx * seq_len for idx in range(num_windows)]


def eval_window_specs(pred_len: int, seq_len: int, stride: int) -> list[tuple[int, int, int, int]]:
    if pred_len <= 0:
        return []
    if pred_len <= seq_len:
        return [(0, pred_len, 0, pred_len)]
    starts = [0]
    final_start = pred_len - seq_len
    cursor = stride
    while cursor < final_start:
        starts.append(cursor)
        cursor += stride
    if starts[-1] != final_start:
        starts.append(final_start)
    specs: list[tuple[int, int, int, int]] = []
    scored_until = 0
    for start in starts:
        valid_len = min(seq_len, pred_len - start)
        window_end = start + valid_len
        new_start = max(scored_until, start)
        score_count = max(window_end - new_start, 0)
        score_from = max(new_start - start, 0)
        if score_count > 0:
            specs.append((start, valid_len, score_from, score_count))
            scored_until = window_end
    return specs


def build_score_window_refs(
    doc_spans: list[DocSpan],
    doc_indices: tuple[int, ...],
    seq_len: int,
    stride: int,
) -> list[ScoreWindowRef]:
    refs: list[ScoreWindowRef] = []
    for doc_index in doc_indices:
        span = doc_spans[doc_index]
        for start, valid_len, score_from, score_count in eval_window_specs(span.pred_len, seq_len, stride):
            refs.append(
                ScoreWindowRef(
                    doc_index=doc_index,
                    start=start,
                    valid_len=valid_len,
                    score_from=score_from,
                    score_count=score_count,
                )
            )
    return refs


def build_train_window_refs(
    doc_spans: list[DocSpan],
    doc_indices: tuple[int, ...],
    seq_len: int,
    seed: int,
    epoch: int,
) -> list[TrainWindowRef]:
    refs: list[TrainWindowRef] = []
    for doc_index in doc_indices:
        span = doc_spans[doc_index]
        for start in train_window_starts(span.pred_len, seq_len, span.doc_id, seed, epoch):
            valid_len = min(seq_len, span.pred_len - start)
            if valid_len > 0:
                refs.append(TrainWindowRef(doc_index=doc_index, start=start, valid_len=valid_len))
    return refs


def pack_train_window(
    doc_tokens: np.ndarray,
    start: int,
    valid_len: int,
    seq_len: int,
    pad_id: int,
    train_context_burnin: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.full((seq_len,), pad_id, dtype=np.int64)
    y = np.full((seq_len,), IGNORE_INDEX, dtype=np.int64)
    x[:valid_len] = doc_tokens[start : start + valid_len].astype(np.int64, copy=False)
    y[:valid_len] = doc_tokens[start + 1 : start + 1 + valid_len].astype(np.int64, copy=False)
    if start > 0 and train_context_burnin > 0:
        burn = min(valid_len, train_context_burnin)
        y[:burn] = IGNORE_INDEX
    return x, y


def pack_eval_window(
    doc_tokens: np.ndarray,
    start: int,
    valid_len: int,
    score_from: int,
    score_count: int,
    seq_len: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.full((seq_len,), pad_id, dtype=np.int64)
    y = np.full((seq_len,), IGNORE_INDEX, dtype=np.int64)
    x[:valid_len] = doc_tokens[start : start + valid_len].astype(np.int64, copy=False)
    y[score_from : score_from + score_count] = doc_tokens[
        start + 1 + score_from : start + 1 + score_from + score_count
    ].astype(np.int64, copy=False)
    return x, y


def load_doc_tokens(store: MemmapTokenStore, doc_spans: list[DocSpan], doc_index: int, cache: dict[int, np.ndarray]) -> np.ndarray:
    cached = cache.get(doc_index)
    if cached is not None:
        return cached
    span = doc_spans[doc_index]
    cached = store.read_span(span.start, span.stop)
    cache[doc_index] = cached
    return cached


def pack_score_batch(
    refs: list[ScoreWindowRef],
    store: MemmapTokenStore,
    doc_spans: list[DocSpan],
    seq_len: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.full((len(refs), seq_len), pad_id, dtype=np.int64)
    y = np.full((len(refs), seq_len), IGNORE_INDEX, dtype=np.int64)
    cache: dict[int, np.ndarray] = {}
    for row, ref in enumerate(refs):
        doc_tokens = load_doc_tokens(store, doc_spans, ref.doc_index, cache)
        window_x, window_y = pack_eval_window(
            doc_tokens,
            ref.start,
            ref.valid_len,
            ref.score_from,
            ref.score_count,
            seq_len,
            pad_id,
        )
        x[row] = window_x
        y[row] = window_y
    return x, y


def pack_train_batch(
    refs: list[TrainWindowRef],
    store: MemmapTokenStore,
    doc_spans: list[DocSpan],
    seq_len: int,
    pad_id: int,
    train_context_burnin: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.full((len(refs), seq_len), pad_id, dtype=np.int64)
    y = np.full((len(refs), seq_len), IGNORE_INDEX, dtype=np.int64)
    cache: dict[int, np.ndarray] = {}
    for row, ref in enumerate(refs):
        doc_tokens = load_doc_tokens(store, doc_spans, ref.doc_index, cache)
        window_x, window_y = pack_train_window(
            doc_tokens,
            ref.start,
            ref.valid_len,
            seq_len,
            pad_id,
            train_context_burnin,
        )
        x[row] = window_x
        y[row] = window_y
    return x, y


class ValidationData:
    def __init__(self, h, device: torch.device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}")
        self.bos_id = int(self.sp.bos_id())
        if self.bos_id < 0:
            raise ValueError("Tokenizer must define a BOS token for document-local sampling")
        self.pad_id = int(self.sp.pad_id())
        if self.pad_id < 0:
            self.pad_id = 0
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = build_sentencepiece_luts(
            self.sp,
            h.vocab_size,
            device,
        )
        self.train_specs, self.train_total_tokens = build_shard_specs(h.train_files)
        self.val_specs, self.val_total_tokens = build_shard_specs(h.val_files)
        all_train_docs = build_doc_spans(
            self.train_specs,
            self.bos_id,
            h.train_doc_limit,
            h.train_token_limit,
            keep_last_open_doc=False,
        )
        all_val_docs = build_doc_spans(
            self.val_specs,
            self.bos_id,
            h.val_doc_limit,
            h.val_token_limit,
            keep_last_open_doc=True,
        )
        if not all_train_docs:
            raise ValueError("No training documents found for document-local sampling")
        if not all_val_docs:
            raise ValueError("No validation documents found for document-local sampling")
        self.all_train_doc_spans = all_train_docs
        self.all_val_doc_spans = all_val_docs
        self.val_ttt_chunks = build_doc_chunks(all_val_docs, h.ttt_chunk_tokens)
        self.train_doc_spans = all_train_docs[h.rank::h.world_size]
        self.val_doc_spans = all_val_docs[h.rank::h.world_size]
        if not self.train_doc_spans:
            raise ValueError(f"Rank {h.rank} received zero training documents")
        if not self.val_doc_spans:
            raise ValueError(f"Rank {h.rank} received zero validation documents")
        h._bos_id = self.bos_id
        h._pad_id = self.pad_id
        h._train_specs = self.train_specs
        h._val_specs = self.val_specs
        h._train_doc_spans = self.train_doc_spans
        h._val_doc_spans = self.val_doc_spans
        h._all_train_doc_spans = all_train_docs
        h._all_val_doc_spans = all_val_docs
        h._val_ttt_chunks = self.val_ttt_chunks


class ShuffledSequenceLoader:
    def __init__(self, h, device: torch.device):
        self.h = h
        self.device = device
        self.seq_len = int(h.train_seq_len)
        self.pad_id = int(h._pad_id)
        self.seed = int(h.seed + h.rank)
        self.shuffle_docs = bool(h.shuffle_docs)
        self.train_context_burnin = max(int(h.train_context_burnin), 0)
        self.store = MemmapTokenStore(h._train_specs)
        self.doc_spans = list(h._train_doc_spans)
        self.next_cycle = 0
        self.current_cycle = 0
        self.doc_order = np.empty((0,), dtype=np.int64)
        self.doc_cursor = 0
        self.current_doc_idx = -1
        self.current_window_specs: list[tuple[int, int]] = []
        self.current_window_cursor = 0
        self.cached_doc_idx = -1
        self.cached_doc_tokens: np.ndarray | None = None
        self._start_new_cycle()

    def _start_new_cycle(self) -> None:
        self.current_cycle = self.next_cycle
        self.next_cycle += 1
        self.doc_order = np.arange(len(self.doc_spans), dtype=np.int64)
        if self.shuffle_docs and self.doc_order.size > 1:
            rng = np.random.default_rng(self.seed + 7919 * self.current_cycle)
            rng.shuffle(self.doc_order)
        self.doc_cursor = 0
        self.current_doc_idx = -1
        self.current_window_specs = []
        self.current_window_cursor = 0

    def _doc_tokens(self, doc_idx: int) -> np.ndarray:
        if self.cached_doc_idx != doc_idx or self.cached_doc_tokens is None:
            span = self.doc_spans[doc_idx]
            self.cached_doc_tokens = self.store.read_span(span.start, span.stop)
            self.cached_doc_idx = doc_idx
        return self.cached_doc_tokens

    def _advance_doc(self) -> None:
        while True:
            if self.doc_cursor >= self.doc_order.size:
                self._start_new_cycle()
            doc_idx = int(self.doc_order[self.doc_cursor])
            self.doc_cursor += 1
            span = self.doc_spans[doc_idx]
            window_specs = [
                (start, min(self.seq_len, span.pred_len - start))
                for start in train_window_starts(span.pred_len, self.seq_len, span.doc_id, self.seed, self.current_cycle)
                if span.pred_len - start > 0
            ]
            if window_specs:
                self.current_doc_idx = doc_idx
                self.current_window_specs = window_specs
                self.current_window_cursor = 0
                return

    def _next_window(self) -> tuple[np.ndarray, np.ndarray]:
        if self.current_window_cursor >= len(self.current_window_specs):
            self._advance_doc()
        start, valid_len = self.current_window_specs[self.current_window_cursor]
        self.current_window_cursor += 1
        doc_tokens = self._doc_tokens(self.current_doc_idx)
        return pack_train_window(doc_tokens, start, valid_len, self.seq_len, self.pad_id, self.train_context_burnin)

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.h.world_size * grad_accum_steps)
        batch_size = device_tokens // self.seq_len
        if batch_size <= 0:
            raise ValueError("TRAIN_BATCH_TOKENS is too small for document-local loader")
        x = np.full((batch_size, self.seq_len), self.pad_id, dtype=np.int64)
        y = np.full((batch_size, self.seq_len), IGNORE_INDEX, dtype=np.int64)
        for batch_idx in range(batch_size):
            window_x, window_y = self._next_window()
            x[batch_idx] = window_x
            y[batch_idx] = window_y
        return (
            torch.from_numpy(x).to(self.device, non_blocking=True),
            torch.from_numpy(y).to(self.device, non_blocking=True),
        )


ORIG_GPT = GPT


class GPT(ORIG_GPT):
    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        if target_ids.dtype != torch.long:
            target_ids = target_ids.long()
        flat_targets = target_ids.reshape(-1)
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            flat_targets,
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        valid = (flat_targets != IGNORE_INDEX).to(dtype=per_token.dtype)
        denom = valid.sum().clamp_min(1.0)
        return (per_token * valid).sum() / denom


def iter_eval_batches(h, doc_spans: list[DocSpan], specs: list[ShardSpec], batch_size: int):
    store = MemmapTokenStore(specs)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for span in doc_spans:
        doc_tokens = store.read_span(span.start, span.stop)
        pred_len = int(doc_tokens.size - 1)
        if pred_len <= 0:
            continue
        for start, valid_len, score_from, score_count in eval_window_specs(pred_len, h.eval_seq_len, h.eval_stride):
            window_x, window_y = pack_eval_window(doc_tokens, start, valid_len, score_from, score_count, h.eval_seq_len, h._pad_id)
            xs.append(window_x)
            ys.append(window_y)
            if len(xs) == batch_size:
                yield np.stack(xs), np.stack(ys)
                xs.clear()
                ys.clear()
    if xs:
        yield np.stack(xs), np.stack(ys)


def masked_byte_count(val_data: ValidationData, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    mask = target_ids != IGNORE_INDEX
    prev_ids = input_ids[mask]
    tgt_ids = target_ids[mask]
    token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
    token_bytes += (val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
    return token_bytes.to(torch.float64).sum()


def evaluate_doc_local(h, device: torch.device, val_data: ValidationData, model) -> tuple[float, float]:
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    batch_size = max(local_batch_tokens // h.eval_seq_len, 1)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for x_np, y_np in iter_eval_batches(h, h._val_doc_spans, h._val_specs, batch_size):
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.int64, non_blocking=True)
            y = torch.from_numpy(y_np).to(device=device, dtype=torch.int64, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                batch_loss = model(x, y).detach()
            batch_tokens = (y != IGNORE_INDEX).sum().to(dtype=torch.float64)
            loss_sum += batch_loss.to(torch.float64) * batch_tokens
            token_count += batch_tokens
            byte_count += masked_byte_count(val_data, x, y)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)


def eval_val(h, device, val_data, model):
    return evaluate_doc_local(h, device, val_data, model)


def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    return evaluate_doc_local(h, device, val_data, base_model)


def eval_val_ttt(h, device, val_data, base_model, batch_seqs=32):
    rank = h.rank
    world_size = h.world_size
    global_batch_windows = batch_seqs * world_size
    doc_spans = h._all_val_doc_spans
    chunks = h._val_ttt_chunks
    store = MemmapTokenStore(h._val_specs)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    ttt_params = [p for p in base_model.parameters()]
    for p in ttt_params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum)
    pending_train_refs: list[TrainWindowRef] = []
    scored_windows = 0
    trained_windows = 0
    dropped_windows = 0
    train_steps = 0
    log(
        f"ttt:start chunks={len(chunks)} chunk_tokens={h.ttt_chunk_tokens} "
        f"ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs} global_batch_windows={global_batch_windows}"
    )
    for chunk in chunks:
        score_refs = build_score_window_refs(doc_spans, chunk.doc_indices, h.eval_seq_len, h.eval_stride)
        my_s = len(score_refs) * rank // world_size
        my_e = len(score_refs) * (rank + 1) // world_size
        my_score_refs = score_refs[my_s:my_e]
        base_model.eval()
        with torch.no_grad():
            for bi in range(0, len(my_score_refs), batch_seqs):
                batch_refs = my_score_refs[bi : bi + batch_seqs]
                x_np, y_np = pack_score_batch(batch_refs, store, doc_spans, h.eval_seq_len, h._pad_id)
                if len(batch_refs) != batch_seqs:
                    x_pad = np.full((batch_seqs, h.eval_seq_len), h._pad_id, dtype=np.int64)
                    y_pad = np.full((batch_seqs, h.eval_seq_len), IGNORE_INDEX, dtype=np.int64)
                    x_pad[: len(batch_refs)] = x_np
                    y_pad[: len(batch_refs)] = y_np
                    x_np, y_np = x_pad, y_pad
                x = torch.from_numpy(x_np).to(device=device, dtype=torch.int64, non_blocking=True)
                y = torch.from_numpy(y_np).to(device=device, dtype=torch.int64, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = compiled_logits(x)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                ).reshape_as(y)
                valid = y != IGNORE_INDEX
                loss_sum += nll[valid].to(torch.float64).sum()
                token_count += valid.sum().to(dtype=torch.float64)
                byte_count += masked_byte_count(val_data, x, y)
        scored_windows += len(score_refs)
        is_last_chunk = chunk.chunk_id == len(chunks) - 1
        if is_last_chunk or h.ttt_epochs <= 0:
            continue
        for epoch in range(h.ttt_epochs):
            pending_train_refs.extend(
                build_train_window_refs(doc_spans, chunk.doc_indices, h.train_seq_len, h.seed, epoch)
            )
        full_batches = len(pending_train_refs) // global_batch_windows
        if full_batches <= 0:
            continue
        cos_lr = h.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * chunk.chunk_id / max(len(chunks) - 1, 1)))
        for group in optimizer.param_groups:
            group["lr"] = cos_lr
        base_model.train()
        for batch_idx in range(full_batches):
            batch_start = batch_idx * global_batch_windows
            batch_end = batch_start + global_batch_windows
            global_refs = pending_train_refs[batch_start:batch_end]
            local_start = rank * batch_seqs
            local_end = local_start + batch_seqs
            local_refs = global_refs[local_start:local_end]
            x_np, y_np = pack_train_batch(
                local_refs,
                store,
                doc_spans,
                h.train_seq_len,
                h._pad_id,
                h.train_context_burnin,
            )
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.int64, non_blocking=True)
            y = torch.from_numpy(y_np).to(device=device, dtype=torch.int64, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = base_model(x, y)
            loss.backward()
            if world_size > 1:
                for p in ttt_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
            optimizer.step()
            train_steps += 1
        trained_now = full_batches * global_batch_windows
        trained_windows += trained_now
        pending_train_refs = pending_train_refs[trained_now:]
        if h.is_main_process and (
            chunk.chunk_id < 4
            or (chunk.chunk_id + 1) % 100 == 0
            or chunk.chunk_id + 1 == len(chunks)
        ):
            log(
                f"ttt:chunk {chunk.chunk_id + 1}/{len(chunks)} scored_windows={len(score_refs)} "
                f"trained_windows={trained_now} pending_windows={len(pending_train_refs)}"
            )
    dropped_windows = len(pending_train_refs)
    if dropped_windows and h.is_main_process:
        log(
            f"ttt:dropped_remainder_windows={dropped_windows} "
            f"threshold={global_batch_windows} reason=below_full_global_batch"
        )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    if h.is_main_process:
        log(
            f"ttt:done scored_windows={scored_windows} trained_windows={trained_windows} "
            f"train_steps={train_steps} dropped_windows={dropped_windows}"
        )
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    return _loss_bpb(loss_sum, token_count, byte_count)


def submission_source_text() -> str:
    return Path(__file__).read_text(encoding="utf-8")


def train_and_eval(h, device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = ValidationData(h, device)
    log(f"train_shards: {len(h._train_specs)} train_docs_local/global: {len(h._train_doc_spans)}/{len(h._all_train_doc_spans)}")
    log(f"val_shards: {len(h._val_specs)} val_docs_local/global: {len(h._val_doc_spans)}/{len(h._all_val_doc_spans)}")
    log(f"doc_windows: train_seq={h.train_seq_len} eval_seq={h.eval_seq_len} eval_stride={h.eval_stride}")
    log(f"train_context_burnin: {h.train_context_burnin}")
    if h.ttt_enabled:
        log(f"ttt_doc_chunks: {len(h._val_ttt_chunks)} target_chunk_tokens={h.ttt_chunk_tokens}")
    base_model, compiled_model = train_model(h, device, val_data)
    torch._dynamo.reset()
    timed_eval("pre-quantization post-ema", eval_val, h, device, val_data, compiled_model)
    serialize(h, base_model, submission_source_text())
    if h.distributed:
        dist.barrier()
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_eval_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval("quantized", eval_val, h, device, val_data, compiled_eval_model)
    if h.sliding_window_enabled:
        timed_eval("quantized_doc_rolling", eval_val_sliding, h, device, val_data, eval_model)
    if h.ttt_enabled and h.sliding_window_enabled:
        del eval_model, compiled_eval_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        timed_eval("quantized_ttt", eval_val_ttt, h, device, val_data, ttt_model)
        del ttt_model


if __name__ == "__main__":
    main()
