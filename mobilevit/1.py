import torch
import torch.nn as nn
from model import mobile_vit_small

device = torch.device( "cpu")

pretrained_dict34 = torch.load("./mobilevit_s.pt",map_location='cpu')  # feiyan_mobilenetv3.pth是在基础网络mobilenetv3上训练肺炎数据的预训练参数
#net1 = efficientnetv2_m(num_classes= 5)  # 原基础网络
net1=mobile_vit_small(num_classes=5)
#net1.
#print(net1)
model_dict34 = net1.state_dict()
# 重新制作预训练的权重，主要是减去参数不匹配的层，楼主这边层名为“fc”
# pretrained_dict34 = {k: v for k, v in pretrained_dict34.items() if k in model_dict34 and 'last_linear' not in k}
pretrained_dict34 = {k: v for k, v in pretrained_dict34.items() if k in model_dict34 and 'classifier' not in k}
# pretrained_dict34 = {k: v for k, v in pretrained_dict34.items() if (k in model_dict34 )}
# 更新权重
model_dict34.update(pretrained_dict34)
net1.load_state_dict(model_dict34)

#in_channel = net1.head.in_features
# net.last_linear = nn.Linear(in_channel, 5)
#net1.head= nn.Linear(in_channel, 5)
net1.to(device)
#print(net1)

a= torch.load("./mobilevit1/mobilevit_base1.pth",map_location='cpu')
b= torch.load("./mobilevit1/mobilevit_base2.pth",map_location='cpu')
print(a)
print('---------')
print(b.classifier.fc.bias)