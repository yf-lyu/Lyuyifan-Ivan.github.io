# 论文解读：P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks

## 简要信息

| 序号 |    属性    |                  值                  |
| :--: | :--------: | :----------------------------------: |
|  1   |  模型名称  |             P-Tuning V2              |
|  2   |  核心内容  |       Prompt-based Fine-tuning       |
|  3   |  论文PDF   | https://arxiv.org/pdf/2110.07602.pdf |
|  4   | GitHub源码 | https://github.com/THUDM/P-tuning-v2 |

## 核心要点

在原始P-tuning基础上，提出deep prompt tuning，对pseudo token采用更深的表征，该方法可以视为Prefix-tuning的拓展版本（Prefix-tuning本身用于生成任务，作者将其拓展为NLU任务）

## 动机

- Fine-tuning需要微调整个预训练语言模型，且额外添加了新的参数，而Prompting则可以将整个预训练语言模型的参数保持固定，而只需要添加prompt来预测结果即可；因此无需微调，但容易陷入局部最优
- Prompt-tuning则是将Prompt加入到微调过程中，此时只对Prompt部分的参数进行训练，而语言模型的参数固定不变。例如，在P-tuning中引入continuous template，此时template是连续可微调的token，而不是离散的token，模型可以学习一个pseudo token
- 因此本文提出P-tuning-V2模型（对Prefix-tuning的扩展），发现稍微改进优化后的prompt-tuning依然可以达到与传统Fine-tuning一样甚至更好的效果；提出Deep Prompt Tuning，进一步对pseudo continuous embedding进行深度表征

## 方法

先前的P-tuning用了一层BiLSTM来表征pseudo token，显然是推理能力不足的原因之一，因此该部分提出Deep Prompt Tuning，替换原来的BiLSTM而使用Prefix-tuning中的深层模型

![P-tuning v2](./多模态NER/img/P-tuning v2.jpg)

（个人理解）根据bert隐藏层数设计Prompt层数，如hidden_layer为12，Prompt设计为shape: (batch_size, pre_seq_len)，pre_seq_len为设定Prompr长度（自定义长度），**prefix_attention_mask**[shape:(batch_size, pre_seq_len)]与**attention_mask**进行拼接作为bert的attention_mask输入

```python3
prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
```

作者设计shape: (batch_size, pre_seq_len, 2 * layer * hidden)作为bert的past_key_values输入，past_key_values根据prefix tokens shape: (batch_size, pre_seq_len)通过embedding获得，经过不断训练去优化past_key_values。对于past_key_values设计为2 * layer * hidden，主要是存在key和value

如何设计past_key_values

```python3
self.prefix_tokens = torch.arange(self.pre_seq_len).long()	# pre_seq_len=11
---------------------------------------------------------------
def get_prompt(self, batch_size):
    prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size,-1).to(
        													self.bert.device)
    past_key_values = self.prefix_encoder(prefix_tokens)	# Embedding 
    # shape:(b_sz, pre_seq_len, 2*layer*hidden)
    past_key_values = past_key_values.view(
                                            batch_size,
                                            self.pre_seq_len,
                                            self.n_layer * 2, 
                                            self.n_head,
                                            self.n_embd
                                        	) 
    # shape:(b_sz, pre_seq_len, 2*layer, n_head, n_embd) 
    # n_embd = hidden_size // num_attention_heads
    past_key_values = self.dropout(past_key_values)
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
    # shape:(2, b_sz, n_head, pre_seq_len, n_embd)*layer
    return past_key_values
----------------------------------------------------------------
past_key_values = self.get_prompt(batch_size=batch_size)
```

**past_key_values：**BertSelfAttention(nn.Module)中对past_key_values存在时，将本身hidden_states得到的key和value与past_key_values拼接起来

```python3
elif past_key_value is not None:
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
    value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
```

而当我们想要训练下游任务时，将bert参数进行冻结只需训练Prompt的参数

### P-tuning v2关键所在

P-tuning v2在引入Prefix-Tuning基础上，提升小模型上的Prompt Tuning

<img src="./多模态NER/img/Prefix-tuning.jpg" alt="Prefix-tuning" style="zoom:80%;" />

Prefix-tuning（前缀微调）最开始应用在NLG任务上，由[Prefix, x, y]三部分构成，如上图所示：Prefix为前缀，x为输入，y为输出。Prefix-tuning将预训练参数固定，Prefix参数进行微调：不仅只在embedding上进行微调，也在TransFormer上的embedding输入每一层进行微调

## 实验

实体识别（CoNLL03、OntoNotes 5.0、CoNLL04）

Baseline选择：

- PT-2：本文提出的P-tuning V2
- MPT-2：本文提出的P-tuning V2，并采用Multi-task learning方法
- PT：P-tuning
- FT：传统的Fine-tuning方法

![ner_实验结果](D:\Paper\论文解读\img\ner_实验结果.jpg)

## 代码实现

```python
class BertPrefixForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        from_pretrained = False
        if from_pretrained:
            self.classifier.load_state_dict(torch.load('model/checkpoint.pkl'))
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)
 
        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param)) # 9860105
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        attention_mask = attention_mask[:,self.pre_seq_len:].contiguous()
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,)
```

```python
class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
```