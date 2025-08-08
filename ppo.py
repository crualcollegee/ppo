from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 构建dataset
class PromptDataset(Dataset):
    # 继承 Dataset 是创建自定义数据集的标准做法，它要求我们必须实现 __init__、__len__ 和 __getitem__ 这三个方法
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        # prompts: 这是一个列表（list），包含了所有原始的、未经处理的文本提示词字符串。例如 ['你好', '请写一首诗']。
        # 一个参数，接收一个从 transformers 库加载的分词器对象
        self.prompts = prompts
        self.tokenizer = tokenizer
        
        self.final_prompts = []
        
        for prompt in prompts:
            if apply_chat_template:
                # 想使用对话模板来格式化提示词
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            else:
                prompt = self.tokenizer.bos_token + prompt
                # 它只是在原始 prompt 字符串的前面加上一个特殊的“句子开始”标记
                
            self.final_prompts.append(prompt)
        
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.final_prompts[index]

# 价值（评论家）模型，用于预测每一步（生成token）的动作产生的收益，使用演员模型进行初始化，并外加一个回归头，输出shape为：(batch_size, seq_len， 1)
class Critic(nn.Module):
    # 这个 Critic 模型的设计很巧妙：它不从头训练一个巨大的新模型，而是复用“演员”（Actor）模型的主体，
    # 只在上面加一个非常小的、可训练的“头”，用来输出价值
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()
        # 将 base_model 设置为评估模式
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        # 线性层 value_head 用于将 base_model 的输出转换为一个标量值，表示每个输入序列的价值
    def forward(self, input_ids, attention_mask, num_actions):
        # input_ids: 输入的 token ID 序列，形状通常是 (batch_size, seq_len)
        # attention_mask: 注意力掩码，用于告诉模型哪些 token 是真实的、需要关注的，哪些是填充的、需要忽略的。形状与 input_ids 相同
        
        hidden_state = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        #  将输入的 input_ids 和 attention_mask 传递给我们之前存储的 base_model。
        # 从 base_model 的输出中，提取最后一层的隐藏状态。这个 hidden_state 张量的形状是 (batch_size, seq_len, hidden_size)
        # hidden_size 指的是每一个词token的 语义向量 
        value_model_output = self.value_head(hidden_state)
        # 将上一步得到的 hidden_state 特征向量传递给我们定义的 value_head 线性层。
        # value_model_output: 线性层的输出。它的形状是 (batch_size, seq_len, 1)
        values = value_model_output.squeeze(-1)[:, -num_actions:]
        # 使张量形状从 (batch_size, seq_len, 1) 变为 (batch_size, seq_len)
        # 我们只取从后往前数 num_actions 个元素  整个 PPO 过程只关心对**新生成的动作（tokens）**的价值评估，而不需要评估输入提示（prompt）部分的价值
        return values


# 计算的是**演员（Actor）**的损失，它的目标是更新策略网络，让能够带来高“优势（Advantage）”的动作（actions）出现的概率更高
def compute_policy_loss(log_probs, old_log_probs, advantages, action_mask=None, clip_eps=0.2):
    # log_probs: 当前策略网络计算出的动作的对数概率  形状: (B, S_a) 张量中位于 (i, j) 位置的元素，表示批次中第 i 条数据
    # 、其生成的第 j 个动作（token）的对数概率
    
    # old_log_probs: 在进行本次更新之前，旧的策略网络计算出的同一个动作的对数概
    ratio = (log_probs - old_log_probs).exp()# 计算新旧策略的概率比 (probability ratio)
    surr1 = ratio * advantages
    # 即概率比 r_t(theta) 乘以优势hatA_t。如果优势为正，我们希望增大这个动作的概率（即增大 ratio）；如果优势为负，则希望减小其概率。
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    # 计算第二个、也是 PPO 最具特色的裁剪后 (clipped) 的代理目标。
    # 它将概率比 ratio 强行限制在 [1−epsilon,1+epsilon] 的区间内。
    loss = -torch.min(surr1, surr2)
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()
    # loss * action_mask: 将损失与掩码相乘，把填充位置的损失清零 
    # 对每个序列的有效损失求和，再除以该序列的有效动作数量，得到每个序列的正确平均损失。
    # .mean(): 最后，对整个批次中所有序列的平均损失再求一次平均，得到最终的标量损失值。

# 计算的是**评论家（Critic）**的损失
def compute_value_loss(values, old_values, returns, action_mask=None, clip_eps: float = None):
    # values: 当前价值网络对状态的价值预测值 V_phi(s_t)。
    #old_values: 更新前的价值网络的预测值 V_phi_old(s_t)。
    #returns: 真实的回报，也叫目标价值 (target value)，通常由 GAE 计算得出。这是价值网络学习的目标。
    if clip_eps is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns) ** 2
        
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()

 #Actor (演员/策略家) 的目标是学习一个**“行为策略”，它需要弄清楚在某个状态下，执行“每一个动作”**的相对好坏，以便调整选择它们的概率。

 #Critic (评论家/裁判) 的目标是学习一个**“打分标准”，它只需要评估当前“这一个状态”**本身有多好，而不需要关心具体采取了哪个动作。


class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []
    
    def append(self, experiences):
        # limit: 一个整数参数，用于指定这个缓冲区的最大容量（最多能存储多少条经验）。
        batch = [{} for _ in range(len(experiences))]
        keys = (
        "seqs",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "num_actions"
    )
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
          
        self.buffer.extend(batch)
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer)-self.limit:]
        
    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer = []
        
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]
    

@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor

@dataclass
class Experience:

    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None

def compute_approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
):
    # 计算策略模型（Actor）的输出概率与参考模型（Reference Model）输出概率之间的差异，这种差异通常用 KL 散度 (KL Divergence) 来衡量
    # 。在 PPO 微调大模型的实践中，这个 KL 散度值会作为一个惩罚项加入到奖励函数中，以防止训练后的模型偏离原始模型太远

    log_ratio = log_probs.float() - ref_log_probs.float()
    # 这个值反映了两个策略在对数空间中的差异。
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio

# A(t) = R(t) + gam*V(t+1) - V(t)
# gae:A(t) = R(t) + gam*V(t+1) - V(t) + gam*lam*A(t+1)
# 最后一个时刻的未来优势和未来收益为0：A(T+1) = 0, V(T+1) = 0,  则A(T) = R(T) - V(T), 得出A(T)
# A(T-1) = R(T-1) + gam*V(T) - V(T-1) + gam*lam*A(T) 知道A(T)可计算A(T-1) 依次类推
# returns(t) = A(t) + V(t) = = R(t) + gam * (V(t+1) + lam * A(t+1))
def get_advantages_and_returns(
    # 实现了泛化优势估计 (Generalized Advantage Estimation, GAE) 算法。GAE 是一种在
    # 偏差和方差之间进行权衡的高级技巧，用于估算在每个时间步（timestep）采取动作的“优势” (Advantage) 和“回报” (Return)，
        values: torch.Tensor, #评论家（Critic）模型对每个时间步的状态价值 V(s_t) 的预测
        rewards: torch.Tensor, # 每个时间步获得的即时奖励 R_t。
        action_mask: torch.Tensor,
        gamma: float, # 折扣因子，用于计算未来奖励的当前价值，通常取值如 0.99。
        lambd: float #偏差和方差之间进行权衡
        ):
    
    lastgaelam = 0 # 初始化一个变量，用于存储下一个时间步的 GAE 值。因为我们是倒序计算的，所以它代表了“未来”的优势信息
    advantages_reversed = [] # 初始化一个空列表，用于按时间倒序存储计算出的每个时间步的优势值。
    response_length = rewards.size(1) #  获取序列的长度（即动作的总数）。
    
    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask * rewards

    for t in reversed(range(response_length)):
        # 它从最后一个时间步 T-1 开始，倒序遍历到第一个时间步 0。这是因为计算当前时间步的优势需要用到下一个时间步的信息
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        # nextvalues: 获取下一个时间步的状态价值 V(s_t+1)
        # 如果当前是最后一个时间步，那么序列已经结束，其“下一个状态”不存在，我们定义其价值为 0。
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        # “近视眼”的优势
        # delta_t=R_t+gammaV(s_t+1)−V(s_t)。它衡量了在当前时间步 t，我们实际获得的（部分）回报与我们预期的回报之间的差距。
        lastgaelam = delta + gamma * lambd * lastgaelam 
        # 既采用近视眼，也采用“未来所有优势的总和” 
        # 它将当前时间步的 TD 优势 delta 与来自未来的、经过折扣的 GAE 优势 lastgaelam（即 hatA GAE
        #_t+1）结合起来，得到当前时间步 t 的 GAE 优势。
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    # [::-1] 将这个列表翻转过来将 Python 列表中的多个张量堆叠成一个单独的张量
    returns = advantages + values
    # hatA_t=G_t−V(s_t)（其中 G_t 是回报），我们可以反推出回报 G_t=hatA_t+V(s_t)。这个 returns 将作为训练评论家（Critic）网络的目标。
    return advantages.detach(), returns

# 它接收一批初始提示（prompts），然后利用当前的策略模型（Actor Model）**来生成续写文本（responses）。
# 最后，它将生成的结果（包括原始序列、掩码、长度等信息）打包成标准化的 Samples 对象，为下一步的“评估”（计算奖励和优势）做准备
def generate_samples(prompts, model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size):
    
    samples_list = []
    model.eval()
    all_prompts = sum([[prompt]*n_samples_per_prompt for prompt in prompts], [])
    # [prompt]*n_samples_per_prompt: 这是一个列表推导式，对于 prompts 列表中的每一个 prompt，
    # 都复制 n_samples_per_prompt 次。例如，如果 n_samples_per_prompt=2，'你好' 就变成了 ['你好', '你好']。
    
    # sum(..., []): 这是一个巧妙的技巧，用于将一个二维列表（列表的列表）“压平”成一维列表。最终，all_prompts 会变成一个包含了所有复制后提示的一维长列表。
    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        prompts = all_prompts[i:i+micro_rollout_batch_size]
        # 取出当前循环需要处理的小批量 prompts
        inputs = actor_tokenizer(prompts, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        # 将所有短于 max_length 的序列填充到 max_length  将所有长于 max_length 的序列截断 返回pytorch张量
        input_ids = inputs['input_ids']
        seqs = model.generate(**inputs.to(device), 
                            max_new_tokens = max_new_tokens, 
                            eos_token_id = eos_token_id, 
                            pad_token_id = pad_token_id)
        if seqs.size(1) >= max_new_tokens + max_length:
            seqs = seqs[:, :max_new_tokens + max_length]
        else:
            seqs = torch.cat([seqs, torch.full((seqs.size(0), max_new_tokens + max_length - seqs.size(1)), fill_value=pad_token_id, device=seqs.device)], dim=1)
            
        attention_mask = (seqs.ne(pad_token_id)).to(dtype=torch.long)
        ans = seqs[:, input_ids.size(1):]
        action_mask = (ans.ne(eos_token_id) & ans.ne(pad_token_id)).to(dtype=torch.long)
       

        samples = Samples(
            seqs=seqs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),
            total_length=attention_mask.float().sum(dim=-1),
        )
        samples_list.append(samples)

    return samples_list


def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):
        # 这个 rewards 张量中，大部分位置的值是 KL 散度惩罚（一个小的负数）
        # ，而只有一个特殊的位置（通常是 <eos> 的位置）的值是 KL 惩罚 + 奖励模型分数。
        kl_divergence_estimate = -kl_ctl * kl
        rewards = kl_divergence_estimate

        ends = action_mask.sum(1) + 1
        
        if not isinstance(clip_reward_value, torch.Tensor):
            clip_reward_value = torch.tensor(clip_reward_value).to(r.device)
    
        reward_clip = torch.clamp(r, -clip_reward_value,
                                  clip_reward_value)
        batch_size = r.size(0)
        for j in range(batch_size):
            rewards[j, :ends[j]][-1] += reward_clip[j, 0]

        return rewards

def generate_experiences(samples_list):

    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()

    experiences = []
    
    for samples in samples_list:
        seqs = samples.seqs
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        with torch.no_grad():
            # 计算策略模型输出token的概率
            output = actor_model(seqs, attention_mask=attention_mask)
            logits = output.logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
            #计算参考模型输出token的概率
            ref_output = ref_model(seqs, attention_mask=attention_mask)
            ref_logits = ref_output.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_log_probs_labels = ref_log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]
            # 计算价值
            value = critic_model.forward(seqs, attention_mask, num_actions).to(device)
            # 转换成文本
            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            # 计算奖励模型的奖励值
            reward_model_inputs = reward_tokenizer(seq_texts, return_tensors="pt", padding=True)
            r = reward_model(**reward_model_inputs.to(device)).logits # 奖励模型的输出，相当于生成最后一个token的奖励（结果奖励模型）
            # 计算kl散度
            kl = compute_approx_kl(
                    action_log_probs,
                    ref_action_log_probs,
                    action_mask=action_mask).to(device)
            # 计算实际奖励
            rewards = compute_rewards(kl, r, action_mask, kl_ctl=0.4, clip_reward_value=0.2)
            # 计算优势和回报
            advantages, returns = get_advantages_and_returns(value, rewards, action_mask, gamma=0.1, lambd=0.2)
        # actor_model.train()
        # critic_model.train()

        experiences.append(Experience(seqs,
                    action_log_probs.detach(),
                    value.detach(),
                    returns.detach(),
                    advantages.detach(),
                    attention_mask,
                    action_mask,
                    r.detach(),
                    samples.response_length,
                    samples.total_length,
                    num_actions,
                    kl.detach(),
        ))

    return experiences

@dataclass
class BufferItem:

    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]

def collate_fn(batch):

    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []
    
    for x in batch:
        seqs.append(x['seqs'])
        action_log_probs.append(x['action_log_probs'])
        values.append(x['values'])
        returns.append(x['returns'])
        advantages.append(x['advantages'])
        attention_mask.append(x['attention_mask'])
        action_mask.append(x['action_mask'])

    seqs = torch.cat(seqs, dim=0)
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)
    
    return BufferItem(seqs, action_log_probs, values, returns, advantages, attention_mask, action_mask, action_mask.size(1))
    
def train_step(experience, steps):
    
    actor_model.train()
    optimizer_actor.zero_grad()

    
    sequences = experience.seqs
    old_action_log_probs = experience.action_log_probs
    advantages = experience.advantages
    num_actions = experience.num_actions
    attention_mask = experience.attention_mask
    action_mask = experience.action_mask
    old_values = experience.values
    returns = experience.returns
    
    logits = actor_model(
            sequences,
            attention_mask=attention_mask).logits
    
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=sequences[:, 1:].unsqueeze(-1))
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
  

    
    policy_loss = compute_policy_loss(action_log_probs, old_action_log_probs, advantages,action_mask=action_mask)
    policy_loss.backward()
    optimizer_actor.step()  
    writer.add_scalar("policy_loss", policy_loss.item(), steps)
    
    critic_model.train()
    optimizer_critic.zero_grad()
    values = critic_model.forward(sequences, attention_mask, num_actions)
    value_loss = compute_value_loss(values, old_values, returns, action_mask)
    value_loss.backward()
    optimizer_critic.step()
    writer.add_scalar("value_loss", value_loss.item(), steps)
    print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}")
    

def train():
    # 初始化经验池
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:
            # 生成样本（获取模型推理结果）
            samples = generate_samples(rand_prompts, actor_model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size)
            # 生成经验（获取优势、奖励、回报等）
            experiences = generate_experiences(samples)
            buffer.append(experiences)
            dataloader = DataLoader(buffer, batch_size=micro_train_batch_size, shuffle=True, collate_fn=collate_fn)
            torch.cuda.empty_cache()
            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1
            
            buffer.clear()
        
            torch.cuda.empty_cache()
            

# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # 一共迭代多少轮
#     episodes = 3
#     # 生成一次经验，训练的轮数
#     max_epochs = 5
#     # 一次从提示词数据集中取多少条数据用于生成经验
#     rollout_batch_size = 8
#     # 一次取多少条数据生成经验（生成经验需要多个模型推理，对显存要求高）
#     micro_rollout_batch_size = 2
#     # 一个提示词生成多少个样本
#     n_samples_per_prompt = 2
#     # 生成的最大长度，相当于最大动作数，数值越大，模型探索的可能性越多
#     max_new_tokens = 50
#     # 最大长度
#     max_length = 256
#     # 实际训练的batch_size大小，一次取多少条数据用于更新参数
#     micro_train_batch_size = 2
#     # 记录日志
#     writer = SummaryWriter('./runs')
#     # 策略模型
#     actor_model = AutoModelForCausalLM.from_pretrained('/home/hz/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct').to(device)
#     # 参考模型
#     ref_model = AutoModelForCausalLM.from_pretrained('/home/hz/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct').to(device)
#     # 奖励模型
#     reward_model = AutoModelForSequenceClassification.from_pretrained('/home/hz/code/baird/model/reward-model-deberta-v3-large-v2').to(device)
#     actor_tokenizer = AutoTokenizer.from_pretrained('/home/hz/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct')
#     reward_tokenizer = AutoTokenizer.from_pretrained('/home/hz/code/baird/model/reward-model-deberta-v3-large-v2')
#     # 价值模型
#     critic_model = Critic(actor_model.base_model).to(device)
    
#     # 初始化优化器
#     optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.00002)
#     optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00002)
    
#     # 填充方式为左填充
#     actor_tokenizer.padding_side = 'left'
#     eos_token_id = actor_tokenizer.eos_token_id
#     pad_token_id = actor_tokenizer.pad_token_id
#     prompt_list = [
#     # 1. Simple Q&A
#     'What is the capital of China?',
#     'Who discovered penicillin?',
#     'What is the average distance from the Earth to the Moon in kilometers?',
#     'How high is Mount Everest?',
#     'What is the Pythagorean theorem?',
#     'Please explain what photosynthesis is.',
#     'What are the first 10 digits of Pi (π)?',
#     'Who wrote "One Hundred Years of Solitude"?',
#     'How many elements are in the periodic table?',
#     'How does the binary system in computers work?',
#     'What is a black hole?',
#     'When did the dinosaurs go extinct?',
#     'Explain the economic concept of "inflation".',
#     'Who was the first person to walk on the moon?',
#     'What is the largest ocean in the world?',

#     # 2. Knowledge & Information Extraction
#     'Extract all the person names mentioned in the following paragraph: "John and Michael have been friends for years, and they work together at Robert\'s company. Yesterday, Emily invited them for dinner."',
#     'Summarize the main pros and cons from this product review: "This phone\'s battery life is outstanding, lasting a full day, and the camera quality is stunning. However, it feels a bit heavy, and the price is on the high side."',
#     'Extract the province, city, and district from the following address: "Address: No. 1 Keji Road, Nanshan District, Shenzhen City, Guangdong Province".',
#     'Is the sentiment of this sentence positive, negative, or neutral? "The weather is beautiful today, sunny and bright, and I\'m really enjoying this walk."',
#     'Find the time and location in the following sentence: "The meeting is scheduled for next Wednesday at 3 PM in the main conference room at the headquarters."',
#     'Based on the following description, what type of animal is this? "It is a large feline, has black stripes on its body, and lives mainly in Asia."',
#     'From the sentence "Apple Inc. was founded in 1976 by Steve Jobs and Steve Wozniak," extract the company\'s founding year.',
#     'What is the core message of this text? "Although artificial intelligence is developing rapidly, we must still pay attention to its ethical risks and potential social impacts to ensure technology is used for good."',
#     'In the following sentence, what are the subject, verb, and object? "The kitten is chasing a butterfly."',
#     'Extract all the numbers from the following text: "The order number is 88015, the total price is 350 dollars, and it includes 3 items."',

#     # 3. Creative Writing & Content Generation
#     'Please write a short four-line poem about a summer night.',
#     'Write the beginning of a short story with the main character being "a forgotten robot".',
#     'Write a catchy slogan for a new coffee product.',
#     'Write an email inviting a friend to your birthday party.',
#     'Continue this story: "When he opened the long-forgotten box, a strange light beamed out..."',
#     'Come up with three different English titles for a movie about time travel.',
#     'Describe a picture of a future city that you imagine.',
#     'Write a short article introducing the local cuisine of your hometown.',
#     'Create a short fable about friendship.',
#     'Design a slogan for an environmental protection campaign.',
#     'Write a dialogue between a young person and an old person about the meaning of life.',
#     'If animals could talk, write the minutes of a meeting among the animals in a forest.',
#     'Write a product description for a perfume named "Stardust".',
#     'Write a short commentary on a current social phenomenon in the style of Mark Twain.',
#     'Create a fairy tale where the main character is a flying cloud.',

#     # 4. Role-playing
#     'You are an experienced world traveler. Please recommend three destinations suitable for solo travel and explain why.',
#     'You are a meticulous scientist. Please explain what quantum entanglement is in simple terms.',
#     'You are now a Shakespearean actor. Seeing the rainy scene outside your window, please perform a short soliloquy.',
#     'You are a senior software engineer. Please explain what an API is to a beginner.',
#     'You are a robot butler living in the 22nd century. Describe your typical day.',
#     'You are a witty food critic. Please review the dish "Pineapple Pizza".',
#     'You are a cat. Describe your opinion of your human owner.',
#     'You are a professional fitness coach. Create a simple one-week workout plan for an office worker who wants to lose weight.',
#     'You are a marketing expert. Write 3 key marketing points for a new smartwatch.',
#     'You are Sherlock Holmes. Please analyze why a wealthy man would go to a park alone late at night.',

#     # 5. Logical Reasoning & Chain-of-Thought
#     'A warehouse has three types of fruit: apples, bananas, and oranges. There are 5 more apples than bananas, and 2 fewer oranges than apples. If there are 10 bananas, how many oranges are there? Please explain step-by-step.',
#     'John\'s house is 3 km east of the school, and Jane\'s house is 2 km west of the school. If they both walk from their homes to the school, who walks a longer distance?',
#     'If the statement "all flying things are birds" is false, can you give a counterexample?',
#     'There are 4 corners in a room. A cat is sitting in each corner. In front of each cat, there are 3 other cats. How many cats are in the room in total?',
#     'Why does the saying "dripping water pierces the stone" hold true? Briefly explain from a physics perspective.',
#     'A person says: "This statement I am making is false." Is the statement true or false? Why?',
#     'If A > B and B > C, what is the relationship between A and C?',
#     'Please explain why salting roads in winter helps to melt ice.',
#     'A farmer has 17 sheep. All but 9 die. How many sheep does he have left?',
#     'If today is Wednesday, what day of the week will it be in 100 days? Please show your calculation.',

#     # 6. Code Generation
#     'Write a Python function to calculate the sum of all even numbers in a list.',
#     'Write an SQL query to select the names of all students with an "age" greater than 20 from a table named "Students".',
#     'Explain the difference between `*args` and `**kwargs` in Python.',
#     'Write a JavaScript function to reverse a string.',
#     'What is a RESTful API? Briefly explain its design principles.',
#     'Write a regular expression to match a valid email address.',
#     'Explain "inheritance" and "polymorphism" in object-oriented programming.',
#     'How do you use the `grep` command in Linux to find lines containing a specific word in a file?',
#     'Write a simple "Hello, World!" program in C++.',
#     'What is Git, and what is its relationship with GitHub?',

#     # 7. Open-ended & Brainstorming
#     'How can city traffic be made more efficient? Propose three innovative ideas.',
#     'If humans could hibernate, how would the world be different?',
#     'If you could have any superpower, what would you choose and why?',
#     'Propose some ideas for the future of education.',
#     'If you could have dinner with any historical figure, who would you choose and what would you ask them?',
#     'How can you celebrate a holiday in an eco-friendly way?',
#     'Imagine an animal that doesn\'t exist and describe its appearance and habits.',
#     'What are some effective methods to improve personal creativity?',
#     'If time travel were possible, which era would you most want to visit and what would you do?',
#     'List 10 examples of how AI could improve everyday life.',

#     # 8. Instruction Following & Constrained Tasks
#     'Write a sentence that includes the words "summer", "beach", and "ice cream", but do not use the word "hot".',
#     'Summarize the plot of the movie "Interstellar" in no more than 50 words.',
#     'Draft a social media post announcing that you got a new kitten. The tone should be cute and include a cat emoji.',
#     'Describe "nostalgia" using a metaphor.',
#     'Generate a list of 5 words that start with the letter "A" and are all related to food.',
#     'Write a paragraph praising the season of spring, and you must use personification.',
#     'Rephrase the sentence "I am very happy today" in a very formal and official manner.',
#     'Explain what "procrastination" is without using the word "procrastinate" or "procrastination".',
#     'Write a three-line poem where the last word of each line rhymes.',
#     'List three planets in our solar system, but do not include Earth.',

#     # 9. Counterfactual & Absurd Questions
#     'If the moon were made of cheese, what effect would it have on the mice of Earth?',
#     'Why are bananas curved instead of straight?',
#     'If humans had wings, would we still need cars?',
#     'Why are the letters on a keyboard arranged the way they are?',
#     'Which would we find at the roots of an infected plant, ozone or gold?',
#     'Why do mirrors reverse left and right, but not top and bottom?',
#     'Why is the sky blue instead of green?',
#     'If Superman existed, would the world still need a police force?',
#     'What would happen if a week had eight days?',
#     'Why do people like to swim in aquariums instead of swimming pools?',
#     'How many ping-pong balls can fit inside a standard basketball?',
#     'Why does one plus one equal two? Argue from a philosophical perspective.',
#     ]
#     prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
#     prompts_dataloader = DataLoader(prompts_dataset, batch_size=rollout_batch_size, shuffle=True)
   
#     train()
    
#     print("训练完成，正在保存模型...")

#     # 1. 定义一个你想保存模型的路径
#     output_dir = "./my_qwen_ppo_model"

#     # 2. 保存策略模型（Actor）和分词器（Tokenizer）
#     # 使用 save_pretrained 会保存模型权重和配置文件，方便后续加载
#     actor_model.save_pretrained(output_dir)
#     actor_tokenizer.save_pretrained(output_dir)

#     # 3. 保存价值模型（Critic）
#     # Critic是一个自定义的nn.Module，所以我们用标准PyTorch方式保存它的状态字典
#     critic_model_path = f"{output_dir}/critic_model.pth"
#     torch.save(critic_model.state_dict(), critic_model_path)
    
#     print(f"模型已成功保存到: {output_dir}")
#     print(f"价值模型已成功保存到: {critic_model_path}")


# [保留您代码上半部分所有的类和函数定义，从 PromptDataset 到 train_step]
# ...
# def train():
# ...

def interactive_chat(model_path, tokenizer_path):
    """
    启动一个交互式聊天循环，用于在终端实时输入和生成。
    """
    print("正在加载模型用于交互式聊天...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载你训练好的模型
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 对于某些模型，需要设置 padding_side 和 pad_token
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval() # 设置为评估模式
    print("模型加载完成！现在您可以开始提问了。")
    print(" (输入 'quit' 或 'exit' 来结束对话)")
    print("-" * 30)

    while True:
        try:
            # 1. 从终端获取用户输入
            user_prompt = input("您: ")

            # 2. 检查退出命令
            if user_prompt.lower() in ["quit", "exit"]:
                print("模型: 再见！")
                break

            # 3. 准备模型输入（应用聊天模板）
            content = [{"role": "user", "content": user_prompt}]
            final_prompt = tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(final_prompt, return_tensors='pt').to(device)

            # 4. 生成回复
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256, # 可以根据需要调整生成长度
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # 5. 解码并只打印生成的部分
            input_length = inputs.input_ids.shape[1]
            response_ids = outputs[0][input_length:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            print(f"模型: {response.strip()}")

        except KeyboardInterrupt:
            # 允许使用 Ctrl+C 优雅地退出
            print("\n模型: 再见！")
            break
        except Exception as e:
            print(f"发生了一个错误: {e}")
            break


if __name__ == "__main__":
    # <<<<<<<<<<<<<<<< 之前的所有代码都已被上面的 interactive_chat 函数替代 >>>>>>>>>>>>>>>>
    
    # 这里填写你训练完成后保存的 actor_model 的文件夹路径
    TRAINED_MODEL_PATH = '/home/hz/code/baird/PPO/my_qwen_ppo_model'
    TOKENIZER_PATH = '/home/hz/code/baird/PPO/my_qwen_ppo_model'

    # 启动交互式聊天
    interactive_chat(TRAINED_MODEL_PATH, TOKENIZER_PATH)
