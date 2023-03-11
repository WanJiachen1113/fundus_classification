import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
# from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,roc_auc_score,auc,precision_score,recall_score,f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import time
import math
import seaborn as sns
import numpy as np
import pandas as pd
# from scipy import interp
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn import metrics
# from model_resnet_se_cancanei import resnet18_se_cancanei,resnet34_se_cancanei, resnet50_se_cancanei,resnet101_se_cancanei,resnet152_se_cancanei
# from model import resnet18,resnet34, resnet50,resnet101, resnet152
# from model_mobilenetv2 import mobilenet_v2
# from model_eca_mobilenetv2 import eca_mobilenet_v2
# from model_cabm_mobilenetv import eca_mobilenet_v2
# # from model_mobilenetv3 import mobilenet_v3_small,mobilenet_v3_large
# # from model_mobilenetv3 import mobilenet_v3_small,mobilenet_v3_large
# from model_v3 import mobilenet_v3_large
#from model_csam_efficientv2 import efficientnetv2_m
from model import resnet50
from model import resnext101_32x8d
# from senet import se_resnet152
from torchsummary import summary

def torch_numpy_list(y):
    y0 = y.cpu()
    y0 = y0.detach().numpy()
    y0 = list(np.ravel(y0))
    return y0

# def duofenlei_zhibiao_weighted(Y_train0123,pre_y_train0123,pre_y_train_gailv):
#     confusion_matrix_sifenlei = confusion_matrix(Y_train0123, pre_y_train0123)
#     y_one_hot =[]
#     for i in range(len(Y_train0123)):
#         if Y_train0123[i]==0:
#             y_one_hot.append([1,0])
#         else:
#             y_one_hot.append([0, 1])
def duofenlei_zhibiao_weighted(Y_train0123, pre_y_train0123, pre_y_train_gailv):
    # print('10', Y_train0123)
    # print('11', len(pre_y_train0123))
    # print('12',  len(pre_y_train_gailv))
    confusion_matrix_sifenlei = confusion_matrix(Y_train0123, pre_y_train0123)
    y_one_hot = []

    for i in range(len(Y_train0123)):
        if Y_train0123[i] == 0:
            y_one_hot.append([1, 0, 0, 0,0])
        elif Y_train0123[i] == 1:
            y_one_hot.append([0, 1, 0, 0,0])
        elif Y_train0123[i] == 2:
            y_one_hot.append([0, 0, 1, 0,0])
        elif Y_train0123[i] == 3:
            y_one_hot.append([0, 0, 0, 1,0])
        else:
            y_one_hot.append([0, 0, 0, 0, 1])

    #y_one_hot = label_binarize(Y_train0123, classes=[0, 1, 2, 3])  # 装换成类似二进制的编码, average="weighted"

    y_one_hot = np.array(y_one_hot)
    pre_y_train_gailv = np.array(pre_y_train_gailv)
    pre_y_train0123 = list(np.ravel(pre_y_train0123))

    accuracy=accuracy_score(np.array(Y_train0123), np.array(pre_y_train0123))
    precision = precision_score(np.array(Y_train0123), np.array(pre_y_train0123), average='macro')
    recall = recall_score(np.array(Y_train0123), np.array(pre_y_train0123),average='macro')

    # f1 = f1_score(np.array(Y_train0123), np.array(pre_y_train0123))
    f1 = f1_score(np.array(Y_train0123), np.array(pre_y_train0123), average='macro')

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        # print('1', y_one_hot.shape)
        # print('11111', pre_y_train_gailv.shape)
        # fpr[i], tpr[i], _ = roc_curve(y_one_hot[:, i], pre_y_train_gailv[:,i])
        fpr[i], tpr[i], _ = roc_curve(y_one_hot[:, i], pre_y_train_gailv[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # 计算宏观平均ROC曲线和ROC面积（方法一）
    # 首先汇总所有假阳性率
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))
    # 然后在此点对所有ROC曲线进行插值
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(5):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= 5
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # 计算微平均ROC曲线和ROC面积（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_one_hot.ravel(), pre_y_train_gailv.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return confusion_matrix_sifenlei,accuracy,precision, recall,f1,roc_auc["macro"], roc_auc["micro"],fpr["macro"], tpr["macro"]



def main():
    global best_valnet_confusion_matrix_sifenlei, best_trainnet_fpr, best_trainnet_tpr, best_valnet_fpr, best_valnet_tpr, best_trainnet_roc_auc_macro, best_valnet_roc_auc_macro
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # device = torch.device("cpu")
    # print("using {} device.".format(device))
    # 数据预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随即裁剪到224×224
                                     transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                     transforms.ToTensor(),  # 转为tensor
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),   # 标准化
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),  # 中心裁剪到224×224
                                   
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    # image_path = os.path.join(data_root, "data_set", "Ruxianai-dataset")
    #image_path = os.path.join(data_root, "newdata_pro")  # flower data set path
    image_path="C:/Users/wanjiachen/Desktop/new_pro"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'covid-19':0, 'normal':1, 'viral pneumonia':2}
    pneumonia_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in pneumonia_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=5)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # net1 = senet154()
    # # load pretrain weights
    # # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "./senet154-pre.pth"
    pretrained_dict34 = torch.load("./resnext101.pth")  # feiyan_mobilenetv3.pth是在基础网络mobilenetv3上训练肺炎数据的预训练参数
    net1 = resnext101_32x8d(num_classes= 1000)  # 原基础网络
    print(net1)
    model_dict34 = net1.state_dict()
    # 重新制作预训练的权重，主要是减去参数不匹配的层，楼主这边层名为“fc”
    # pretrained_dict34 = {k: v for k, v in pretrained_dict34.items() if k in model_dict34 and 'last_linear' not in k}
    pretrained_dict34 = {k: v for k, v in pretrained_dict34.items() if k in model_dict34 and 'classifier' not in k}
    # pretrained_dict34 = {k: v for k, v in pretrained_dict34.items() if (k in model_dict34 )}
    # 更新权重
    model_dict34.update(pretrained_dict34)
    net1.load_state_dict(model_dict34)
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net1.load_state_dict(torch.load(model_weight_path, map_location=device))
    # in_channel = net1.last_linear.in_features
    #in_channel = net1.head.classifier.in_features
    in_channel = net1.fc.in_features
    # net.last_linear = nn.Linear(in_channel, 5)
    net1.fc= nn.Linear(in_channel, 5)
    net1.to(device)

    # summary(net1, input_size=(3, 224, 224))
    # print(net1)
    # for param in net1.load.features.parameters():
    #     param.requires_grad = False
    # in_channel = net1.last_channel
    # net1.classifier = nn.Sequential(
    #     nn.Dropout(0.25),
    #     nn.Linear(in_channel, 2),
    # )
    # freeze features weights
    # for param in net1.features.parameters():
    #     param.requires_grad = False
    epoch_n = 20
    if epoch_n <= 50:
        lr = 0.0001
    else:
        lr = 0.00001


    loss_f_1 = torch.nn.CrossEntropyLoss()
    # loss_f_2 = torch.nn.CrossEntropyLoss()
    # optimizer_1 = torch.optim.Adam(net1.parameters(), lr=lr)
    optimizer_1 = torch.optim.Adam(net1.parameters(), lr=lr)
    # weight_1 = 0.7
    # weight_2 = 0.3

    time_open = time.time()

    ep = []
    train_loss_list1 = []
    train_acc_list1 = []
    # train_loss_list2 = []
    # train_acc_list2 = []
    # train_integration_acc_list = []
    # train_integration_loss_list = []
    valid_loss_list1 = []
    valid_acc_list1 = []
    # valid_loss_list2 = []
    # valid_acc_list2 = []
    # valid_integration_acc_list = []
    # valid_integration_loss_list = []
    save_path1 = './jieguo/resnext101_new.pth'
    # save_path2 = './checkpoints/resNet50_gai.pth'
    for epoch in range(epoch_n):

        ep.append(epoch+1)

        print('Epoch {}/{}'.format(epoch+1, epoch_n))
        print('---' * 10)
        print('Training...')
        # net1.train(True)
        net1.train(True)

        running_loss_1 = 0.0
        # running_loss_2 = 0.0
        # running_integration_loss=0.0

        train_bar = tqdm(train_loader)
        pre_y_train0123_1 = []
        # pre_y_train0123_2 = []
        # integration_y_pred_train0123=[]
        pre_y_train_gailv_1 = []
        # pre_y_train_gailv_2 = []
        # train_integration_y_pred_gailv=[]
        Y_train0123_1 = []
        Y_train0123_2 = []
        # integration_y_train0123=[]
        for batch, data in enumerate(train_bar):
            #X, y =data[0].to(device), data[1].to(device)
            X, y = data
            # X, y = Variable(X.cuda()), Variable(y.cuda())
            X, y = Variable(X.to(device)), Variable(y.to(device))
            # optimizer_1.zero_grad()
            y_pred_1 = net1(X)
            # print("20", y_pred_1.shape)
            pred_gailv_1 = torch_numpy_list(y_pred_1)
            for i in range(len(pred_gailv_1)//5):
                pre_y_train_gailv_1.append(pred_gailv_1[5*i:(5*i+5)])
            _, pred_1 = torch.max(y_pred_1.data, 1)
            # if loss_1<0.45:
            #     loss_1 = (loss_1 - b).abs() + b
            b = 0.2
            loss_1 = loss_f_1(y_pred_1, y)
            loss_1 = (loss_1 - b).abs() + b
            # loss_2 = (loss_2 - b).abs() + b
            optimizer_1.zero_grad()
            loss_1.backward()
            # loss_2.backward()
            optimizer_1.step()

            running_loss_1 += loss_1.data.item()
            train_bar.desc = "train epoch[{}/{}] ".format(epoch + 1,
                                                          epoch_n)
            if batch % 100 == 0 and batch != 0:
                print('Batch {},Model_34 Loss:{:.4f}\
                    '.format(batch, running_loss_1 / batch))

            pred_1_zhibiao = torch_numpy_list(pred_1)
            for i in range(len(pred_1_zhibiao)):
                pre_y_train0123_1.append(pred_1_zhibiao[i])
            # print("300", y.shape)
            y = torch_numpy_list(y)

            for i in range(len(y)):

                Y_train0123_1.append(y[i])

        trainnet_confusion_matrix_sifenlei,trainnet_accuracy, trainnet_precision, trainnet_recall,trainnet_f1,trainnet_roc_auc_macro, trainnet_roc_auc_micro,trainnet_fpr, trainnet_tpr=duofenlei_zhibiao_weighted(Y_train0123_1, pre_y_train0123_1, pre_y_train_gailv_1)
        # train50_confusion_matrix_sifenlei, train50_accuracy, train50_precision, train50_recall,train50_f1, train50_roc_auc_macro, train50_roc_auc_micro,train50_fpr, train50_tpr = duofenlei_zhibiao_weighted(Y_train0123_2,pre_y_train0123_2, pre_y_train_gailv_2)
        # train_integration_confusion_matrix_sifenlei, train_integration_accuracy, train_integration_precision, train_integration_recall, train_integration_f1, train_integration_roc_auc_macro, train_integration_roc_auc_micro, train_integration_fpr, train_integration_tpr = duofenlei_zhibiao_weighted( integration_y_train0123, integration_y_pred_train0123, train_integration_y_pred_gailv)
        print('trainnet_accuracy: %f,trainnet_precision: %f,trainnet_recall: %f,trainnet_f1: %f,trainnet_auc["macro"]: %f,trainnet_auc["micro"]: %f'
              % ( trainnet_accuracy,  trainnet_precision,  trainnet_recall, trainnet_f1, trainnet_roc_auc_macro, trainnet_roc_auc_micro))
        # print('train50_accuracy: %f,train50_precision: %f,train50_recall: %f,train50_f1: %f,train50_auc["macro"]: %f,train50_auc["micro"]: %f'
        #     % (train50_accuracy, train50_precision, train50_recall, train50_f1, train50_roc_auc_macro,train50_roc_auc_micro))
        # print('train_integration_accuracy: %f,train_integration_precision: %f,train_integration_recall: %f,train_integration_f1: %f,train_integration_auc["macro"]: %f,train_integration_auc["micro"]: %f'
        #     % (train_integration_accuracy, train_integration_precision, train_integration_recall, train_integration_f1, train_integration_roc_auc_macro,train_integration_roc_auc_micro))

        train_loss_1 = running_loss_1 * batch_size / train_num

        print('Model_net Loss: %.4f  ;' %
              (train_loss_1))

        train_loss_list1.append(train_loss_1)
        train_acc_list1.append(trainnet_accuracy)

        print('Validing...')
        # net1.eval()
        net1.eval()
        best_acc = 0.0
        running_loss_1 = 0.0
        # running_loss_2 = 0.0
        # running_integration_loss = 0.0
        pre_y_val0123_1 = []
        # pre_y_val0123_2 = []
        # integration_y_pred_val0123 = []
        pre_y_val_gailv_1 = []
        # pre_y_val_gailv_2 = []
        # val_integration_y_pred_gailv = []
        Y_val0123_1 = []
        # Y_val0123_2 = []
        # integration_y_val0123 = []
        with torch.no_grad():
            validate_bar = tqdm(validate_loader)

            for batch, data in enumerate(validate_bar):
                #X, y =data[0].to(device), data[1].to(device)
                X, y = data
                # X, y = Variable(X.cuda()), Variable(y.cuda())
                X, y = Variable(X.to(device)), Variable(y.to(device))
                y_pred_1 = net1(X)
                # print("2",y_pred_1.shape)
                pred_gailv_1 = torch_numpy_list(y_pred_1)

                for i in range(len(pred_gailv_1) // 5):
                    pre_y_val_gailv_1.append(pred_gailv_1[5 * i:(5 * i + 5)])
                _, pred_1 = torch.max(y_pred_1.data, 1)
                loss_1 = loss_f_1(y_pred_1, y)
                running_loss_1 += loss_1.data.item()
                pred_1_zhibiao = torch_numpy_list(pred_1)
                for i in range(len(pred_1_zhibiao)):
                    pre_y_val0123_1.append(pred_1_zhibiao[i])
                y = torch_numpy_list(y)
                for i in range(len(y)):
                    Y_val0123_1.append(y[i])
                    # Y_val0123_2.append(y[i])
                    # integration_y_val0123.append(y[i])
            # print("220", Y_val0123_1.shape)
            # print("22", pre_y_val_gailv_1.shape)
            valnet_confusion_matrix_sifenlei, valnet_accuracy, valnet_precision, valnet_recall, valnet_f1, valnet_roc_auc_macro, valnet_roc_auc_micro, valnet_fpr, valnet_tpr = duofenlei_zhibiao_weighted(Y_val0123_1, pre_y_val0123_1, pre_y_val_gailv_1)
            # val50_confusion_matrix_sifenlei, val50_accuracy, val50_precision, val50_recall, val50_f1, val50_roc_auc_macro, val50_roc_auc_micro, val50_fpr, val50_tpr = duofenlei_zhibiao_weighted(Y_val0123_2, pre_y_val0123_2, pre_y_val_gailv_2)
            # val_integration_confusion_matrix_sifenlei, val_integration_accuracy, val_integration_precision, val_integration_recall, val_integration_f1, val_integration_roc_auc_macro, val_integration_roc_auc_micro, val_integration_fpr, val_integration_tpr = duofenlei_zhibiao_weighted(integration_y_val0123, integration_y_pred_val0123, val_integration_y_pred_gailv)
            print('valnet_accuracy: %f,valnet_precision: %f,valnet_recall: %f,valnet_f1: %f,valnet_auc["macro"]: %f,valnet_auc["micro"]: %f'
                % (valnet_accuracy, valnet_precision, valnet_recall, valnet_f1, valnet_roc_auc_macro,valnet_roc_auc_micro))
            # print('val50_accuracy: %f,val50_precision: %f,val50_recall: %f,val50_f1: %f,val50_auc["macro"]: %f,val50_auc["micro"]: %f'
            #     % (val50_accuracy, val50_precision, val50_recall,val50_f1,val50_roc_auc_macro, val50_roc_auc_micro))
            # print('val_integration_accuracy: %f,val_integration_precision: %f,val_integration_recall: %f,val_integration_f1: %f,val_integration_auc["macro"]: %f,val_integration_auc["micro"]: %f'
            #     % (val_integration_accuracy, val_integration_precision, val_integration_recall, val_integration_f1,val_integration_roc_auc_macro, val_integration_roc_auc_micro))

            valid_loss_1 = running_loss_1 * batch_size / val_num
            # valid_loss_2 = running_loss_2 * batch_size / val_num
            # valid_integration_loss = running_integration_loss * batch_size / val_num


            validate_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       epoch_n)

        print('Model_val_net Loss: %.4f ; ' %
              (valid_loss_1))
        # if epoch%5==0:
        #     Coderfilepath_34=os.path.join("",'Codercheckpoint_model34_epoch_{}.pth'.format(epoch))
        #     Coderfilepath_34=r'G:\xbx - erfenl\model_jieguo\CC\model34/'+Coderfilepath_34
        #     Coderfilepath_50 = os.path.join("", 'Codercheckpoint_model50_epoch_{}.pth'.format(epoch))
        #     Coderfilepath_50 = r'G:\xbx - erfenl\model_jieguo\CC\model50/' + Coderfilepath_50
        #     torch.save(net1.state_dict(), Coderfilepath_34)
        '''if valnet_accuracy > 0.88 and valnet_roc_auc_macro > 0.9:


                if trainnet_roc_auc_macro > valnet_roc_auc_macro and trainnet_accuracy > valnet_accuracy and trainnet_roc_auc_macro - valnet_roc_auc_macro < 0.6:'''
        if True:
            if epoch_n==epoch+1:
                outputfile_fpr = './jieguo/FPR.xlsx'
                outputfile_tpe = './jieguo/TPR.xlsx'
                writer_fpr = pd.ExcelWriter(outputfile_fpr)
                writer_tpe = pd.ExcelWriter(outputfile_tpe)
                pd.DataFrame(trainnet_fpr).to_excel(writer_fpr, sheet_name='train_fpr')
                pd.DataFrame(valnet_fpr).to_excel(writer_fpr, sheet_name='val_fpr')
                pd.DataFrame(trainnet_tpr).to_excel(writer_tpe, sheet_name='train_tpr')
                pd.DataFrame(valnet_tpr).to_excel(writer_tpe, sheet_name='val_tpr')
                writer_fpr.save()
                writer_tpe.save()
                plt.figure(11)
                lw = 2
                plt.plot(trainnet_fpr, trainnet_tpr, color='blue',
                         lw=lw, label='ROC curve train (area = %0.3f)' % trainnet_roc_auc_macro)
                plt.plot(valnet_fpr, valnet_tpr, color='darkorange',
                         lw=lw, label='ROC curve test (area = %0.3f)' % valnet_roc_auc_macro)
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                # plt.title('ROC curves of CC and MLO')
                plt.title('ROC curves of  view')
                plt.legend(loc="lower right")

                plt.savefig('./jieguo/roc.png')

                plt.show()
                # 混淆矩阵
                plt.figure(123)
                f, ax = plt.subplots()
                sns.heatmap(valnet_confusion_matrix_sifenlei, annot=True, ax=ax)  # 画热力图
                ax.set_title('confusion matrix of Net')  # 标题
                ax.set_xlabel('predict')  # x轴
                ax.set_ylabel('true')  # y轴

                plt.savefig('./jieguo/hunxiaojuzhen.png')

                plt.show()

                time_end = time.time() - time_open
                print(time_end)
                print('Finished')

            best_trainnet_roc_auc_macro = trainnet_roc_auc_macro
            best_valnet_roc_auc_macro=valnet_roc_auc_macro
            best_valnet_confusion_matrix_sifenlei=valnet_confusion_matrix_sifenlei
            best_acc = valnet_accuracy
            best_trainnet_fpr, best_trainnet_tpr = trainnet_fpr, trainnet_tpr
            best_valnet_fpr, best_valnet_tpr=valnet_fpr,valnet_tpr
            # torch.save(net1.state_dict(), save_path1)
            torch.save(net1.state_dict(), save_path1)
        valid_loss_list1.append(valid_loss_1)
        valid_acc_list1.append(valnet_accuracy)
    plt.figure(111)
    plt.plot(ep, train_loss_list1, 'g', label='Net Training loss')
    # plt.plot(ep, train_loss_list2, 'y', label='model_2 Training loss')
    # plt.plot(ep, train_integration_loss_list, 'm', label='integration_model Training loss')
    plt.plot(ep, valid_loss_list1, 'b', label='Net validation loss')
    # plt.plot(ep, valid_loss_list2, 'r', label='model_2 validation loss')
    # plt.plot(ep, valid_integration_loss_list, 'k', label='integration_model validation loss')
    plt.title('Training and Validation loss for each epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Model Loss')
    plt.legend()
    plt.savefig('./jieguo/loss2_34_zhibiao.png')

    plt.figure(222)
    plt.plot(ep, train_acc_list1, 'g', label='net Training accuracy')
    # plt.plot(ep, train_acc_list2, 'y', label='model_2 Training accuracy')
    # plt.plot(ep, train_integration_acc_list, 'm', label='integration_model Training accuracy')
    plt.plot(ep, valid_acc_list1, 'b', label='net validation accuracy')
    # plt.plot(ep, valid_acc_list2, 'r', label='model_2 validation accuracy')
    # plt.plot(ep, valid_integration_acc_list, 'k', label='integration_model validation accuracy')
    plt.title('Training and Validation Accuracy for each epoch')
    plt.xlim([0.0, epoch_n])
    plt.ylim([0.5, 1.0])

    plt.xlabel('Epoch')
    plt.ylabel('Model Accuracy per Epoch')
    plt.legend()
    plt.savefig('./jieguo/accuarcy2_34_zhibiao.png')

    # plt.figure(3)
    # lw = 2
    # plt.plot(best_trainnet_fpr, best_trainnet_tpr, color='blue',
    #          lw=lw, label='ROC curve train (area = %0.2f)' %  best_trainnet_roc_auc_macro)
    # plt.plot(best_valnet_fpr, best_valnet_tpr, color='darkorange',
    #          lw=lw, label='ROC curve test (area = %0.2f)' %  best_valnet_roc_auc_macro)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # # plt.title('ROC curves of CC and MLO')
    # plt.title('ROC curves of net')
    # plt.legend(loc="lower right")
    # plt.show()

    plt.figure(321)
    f, ax = plt.subplots()
    sns.heatmap(valnet_confusion_matrix_sifenlei, annot=True, ax=ax)  # 画热力图
    ax.set_title('confusion matrix of train34')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.savefig('./jieguo/relitu.png')
    plt.show()

    time_end = time.time() - time_open
    print(time_end)
    print('Finished')

    plt.show()
if __name__ == '__main__':
    main()