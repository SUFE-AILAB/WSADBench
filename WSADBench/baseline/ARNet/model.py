import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
# from utils import fill_context_mask
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

#权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

#单层全连接模型
class Model_single(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_single, self).__init__()   
        self.fc = nn.Linear(n_feature, n_feature)    
        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        #inputs: (B, T, F)
        x = F.relu(self.fc(inputs))          #(B,T,F) -> (B,T,F)[120,32,2048]
        if is_training:
            x = self.dropout(x)             
        return x, self.sigmoid(self.classifier(x))  #输出特征[120,32,2048] + 异常分数(120, 32, 1)   


class Filter_Module(nn.Module):
    def __init__(self, len_feature):
        super(Filter_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=1,
                      stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1,
                      stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):      #通过两层1D卷积生成每个片段的前景权重（重要性分数），用于加权原始特征
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)    #(B, 512, T)
        out = self.conv_2(out)    #(B, 1, T)
        out = out.permute(0, 2, 1) 
        # out: (B, T, 1)
        return out

#多卷积层生成类别激活分数
class CAS_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(CAS_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes + 1, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.drop_out = nn.Dropout(p=0.7)

    def forward(self, x):   #用于生成每个clip的类别激活分数（含异常类），支持多分类
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)     # (B, 2048, T)
        out = self.conv_2(out)     # (B, 2048, T)
        out = self.drop_out(out)   
        out = self.conv_3(out)    # (B, C + 1, T)
        out = out.permute(0, 2, 1)
        # out: (B, T, C + 1)
        return out


class BaS_Net(nn.Module):        #这个模型输出有点特别，运行时需要重赋值变量
    def __init__(self, len_feature, num_classes, num_segments):
        super(BaS_Net, self).__init__()
        self.filter_module = Filter_Module(len_feature)    #前景权重生成
        self.len_feature = len_feature                      
        self.num_classes = num_classes

        self.cas_module = CAS_Module(len_feature, num_classes)      # 类别激活

        self.softmax = nn.Softmax(dim=1)

        self.num_segments = num_segments
        self.k = num_segments // 8                #得分最高的k个片段取平均

    def forward(self, x):
        # Step1: 生成片段重要性权重
        fore_weights = self.filter_module(x)       # [B, T, 1]
        # Step2: 前景增强特征
        x_supp = fore_weights * x           # 加权重要片段  # [B, T, F]

        # Step3: 双分支异常检测
        cas_base = self.cas_module(x)       # 原始特征分支        (B, T, C + 1)
        cas_supp = self.cas_module(x_supp)      # 前景增强分支    (B, T, C + 1)

        # Step4: Top-k片段聚合
        score_base = torch.mean(torch.topk(cas_base, self.k, dim=1)[0], dim=1)   #  (B, C + 1)
        score_supp = torch.mean(torch.topk(cas_supp, self.k, dim=1)[0], dim=1)   # (B, C + 1)

        score_base = self.softmax(score_base)   
        score_supp = self.softmax(score_supp)

        return score_base, cas_base, score_supp, cas_supp, fore_weights




class Model_mean(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_mean, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)
        #使用三个不同的卷积核的1维卷积层来处理输入特征
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')

        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)
        self.mean_pooling = nn.AvgPool2d((3, 1))      #3*1平均池化层融合
        # self.weight_conv1 = nn.Conv2d(n_channels, out_channels, kernel_size, stride=1,
        #          padding=0, dilation=1, groups=1,
        #          bias=True, padding_mode='zeros')

    def forward(self, inputs, is_training=True):    #卷积需要的输入维度：[B,channel, T]  channel = F
        segments = 10
        segments_n = inputs.shape[0] // segments  #计算输入的段数
        if inputs.ndim == 2:  # 如果输入是二维张量(B, F)，则添加一个维度
            inputs = inputs[:segments * segments_n].reshape(segments_n, segments, inputs.shape[1])  # (B, T, F)
        inputs = inputs.permute(0, 2, 1)     #将输入特征从(B, T, F)转换为(B, F, T)   [120,32,2048] -> [120, 2048, 32]
        x_1 = F.relu(self.conv1(inputs)).permute(0, 2, 1).unsqueeze(2)    # (B,T,F,1)
        x_2 = F.relu(self.conv2(inputs)).permute(0, 2, 1).unsqueeze(2)    # (B,T,F,1)
        x_3 = F.relu(self.conv3(inputs)).permute(0, 2, 1).unsqueeze(2)    # (B,T,F,1)
        x = torch.cat((x_1, x_2, x_3), dim=2)   #拼接特征  # (B,T,F,3)
        x = self.mean_pooling(x)    # (B,T,F,1)  #3*1平均池化层融合
        # x = x_3 + x_2
        # x = F.relu(self.conv_b2(x))
        # x = x_1 + x
        # x = F.relu(self.conv_b1(x))
        x = x.squeeze(2)   # (B,T,F)  #[120,32,2048]   第二次 [200,10,2048]
        if is_training:
            x = self.dropout(x)
        return x, self.sigmoid(self.classifier(x))   #输出特征 + 异常分数(B, T, 1)


class Model_sequence(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_sequence, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')

        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):            #残差连接融合多尺度特征
        segments = 10
        segments_n = inputs.shape[0] // segments  #计算输入的段数
        if inputs.ndim == 2:  # 如果输入是二维张量(B, F)，则添加一个维度
            inputs = inputs[:segments * segments_n].reshape(segments_n, segments, inputs.shape[1])  # (B, T, F)
        inputs = inputs.permute(0, 2, 1)
        x_1 = F.relu(self.conv1(inputs))
        x_2 = F.relu(self.conv2(inputs))
        x_3 = F.relu(self.conv3(inputs))
        x = x_3 + x_2
        x = F.relu(self.conv_b2(x))
        x = x_1 + x
        x = F.relu(self.conv_b1(x))

        if is_training:
            x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x, self.sigmoid(self.classifier(x))

class Model_concatcate(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_concatcate, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b1 = nn.Conv1d(in_channels=n_feature * 3, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        # self.conv_b2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
        #          padding=0, dilation=1, groups=1,
        #          bias=True, padding_mode='zeros')

        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):           #全连接层融合特征/拼接结构
        segments = 10
        segments_n = inputs.shape[0] // segments  #计算输入的段数
        if inputs.ndim == 2:  # 如果输入是二维张量(B, F)，则添加一个维度
            inputs = inputs[:segments * segments_n].reshape(segments_n, segments, inputs.shape[1])  # (B, T, F)
        inputs = inputs.permute(0, 2, 1)
        x_1 = F.relu(self.conv1(inputs))
        x_2 = F.relu(self.conv2(inputs))
        x_3 = F.relu(self.conv3(inputs))
        x = torch.cat((x_1, x_2, x_3), dim=1)   # [B,F*3,T] 通道维数拼接
        x = self.conv_b1(x)    # [B,F,T] 通过1D卷积层融合特征 
        
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc(x))

        # x = x_3 + x_2
        # x = F.relu(self.conv_b2(x))
        # x = x_1 + x
        # x = F.relu(self.conv_b1(x))

        if is_training:
            x = self.dropout(x)

        return x, self.sigmoid(self.classifier(x))

class model_lstm(torch.nn.Module):
    def __init__(self, n_feature,seq_len):
        super(model_lstm, self).__init__()
        self.bidirectlstm = nn.LSTM(
            input_size=n_feature,
            hidden_size=n_feature,    #输入/输出同维
            num_layers=1,
            batch_first=True)            #支持变长序列
        self.seq_len = seq_len
        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(0.7)

    def forward(self, inputs, seq_len, is_training=True):
        self.bidirectlstm.flatten_parameters()   #降低显存占用
        if is_training:
            seq_len_list = seq_len.tolist()
            x = pack_padded_sequence(input=inputs, lengths=seq_len_list, batch_first=True, enforce_sorted=False)  #支持变长序列输入，抽取时序特征
            x, _ = self.bidirectlstm(x)               #双向LSTM
            x, _ = pad_packed_sequence(x, batch_first=True)    #填充回标准张量  [120,32,2048]
            # x = self.dropout(x)
        else:
            x, _ = self.bidirectlstm(inputs)
        return x, self.sigmoid(self.classifier(x))


#统一接口  包含6中预定义架构
def model_generater(model_name, feature_size,seq_len=None):    #seq_len为lstm添加  #目前是认定feature_size与n_feature相同
    print(f'model_name:{model_name}')
    if model_name == 'model_single':
        model = Model_single(feature_size)  # for anomaly detection, only one class, anomaly, is needed.
    elif model_name == 'model_mean':
        model = Model_mean(feature_size)
    elif model_name == 'model_sequence':
        model = Model_sequence(feature_size)
    elif model_name == 'model_concatcate':
        model = Model_concatcate(feature_size)
    elif model_name == 'model_lstm':
        model = model_lstm(feature_size,seq_len)
    elif model_name == 'model_bas':
        model = BaS_Net(feature_size)
    else:
        raise ('model_name is out of option')
    return model


def get_vars(self):
        """获取模型参数列表"""
        return self.vars



def init_weights_xavier(module):
    """
    Xavier权重初始化函数
    
    Args:
        module: 神经网络模块
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def count_parameters(model):
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 兼容性别名
# Learner = model_generater  # 兼容原始实现的命名