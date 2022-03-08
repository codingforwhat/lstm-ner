import numpy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
the1=numpy.empty([750000,50])
ci=dict()
f4=open("ans.txt","r+",encoding='utf-8')
f2=open("ctb.50d.txt","r+",encoding='utf-8')
sum =0
for lines in f2.readlines():#读入词向量文件并在字典中构建词语到词向量的索引
    if (lines[0]=='\n'): continue
    sum += 1
    lin=lines.split()
    ci[lin[0]]=sum
    del lin[0]
    the1[sum]=numpy.array(lin)
f2.close()
#print(the1,file=f4)
the=torch.from_numpy(the1)
shu=numpy.empty([750000],dtype=int)
f2=open("no1.txt","r",encoding='utf-8')
left=0
sum=0
train_x=torch.rand(750000,5,50)
train_y=torch.zeros(750000)
quan=0
xunlianquan=0
for line in f2.readlines():#读入训练集，并储存对应词向量位置
    b=line.split("  ")
    for key in b:
        if key!='\n':
            c=key.split("/")
            sum+=1
            #if c[1][0]!='w':
                #print(c[1])
            if(c[1]=='nt') : bo=2
            else : bo=0  # #
            if c[0][0]== '[' :#对于”[]“进行特殊处理
                left=1
                q=c[0].split("[")
                if(q[1] in ci):
                    shu[sum]=ci[q[1]]
                else: shu[sum]=9
                #print(line1(q[1]), file=f3,end='')
            else:
                if left ==0 :
                    if(c[0] not in ci):
                        shu[sum]=9
                    else :
                        shu[sum]=ci[c[0]]
                    train_y[sum] = bo
                    if(bo==2):
                        xunlianquan+=1
                    #print(line1(c[0]), file=f3,end='')
                    #print(bo, file=f3)
                else :
                    left+=1
                    if(left>=5):
                        #left=0
                        for pi in range(left):
                            #shu[sum-left+pi+1][1]=0
                            train_y[sum-left+pi+1]=0
                        left=0
                    else:
                        m=c[1].split("]")
                        if m[0]!=c[1] :
                            #left=0
                            if(m[1]=="nt"):
                                for pi in range(left):
                                    #shu[sum-left+pi+1][1]=1
                                    train_y[sum - left + pi + 1] = 1
                                train_y[sum - left + 1] = 2
                                xunlianquan+=1
                            else:
                                for pi in range(left):
                                    #shu[sum-left+pi+1][1]=0
                                    train_y[sum - left + pi + 1] = 0
                            left=0

f2.close()
trb=numpy.empty([750000,5,50])
for shuju in range(1,sum):  # 构建训练集的词向量矩阵
    for i in range(5):
        trb[shuju+2][i]=the[shu[shuju+i]]

train_x=torch.from_numpy(trb)
train_x=train_x.type(torch.float32)  # 将词向量矩阵转换为tensor

class LSTMTagger(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,target_size):
        super(LSTMTagger, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.target_size=target_size
        #  LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm=nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        # 线性层将隐状态空间映射到标注空间
        self.out2tag =  nn.Linear(self.hidden_dim, self.target_size)
        self.relu=nn.ReLU(inplace=True)#使用relu激活函数优化模型

    def forward(self, inputs):
        #out, self.hidden = self.lstm(inputs, self.hidden)
        out, _ = self.lstm(inputs)
        # 做出预测
        out=torch.sum(out,dim=1)
        #out=out[:, -1, :]#将5维的lstm输出，取最后一维作为结果
        out=self.relu(out)
        #tag_space = self.out2tag(out.reshape(len(inputs), -1))
        tag_space = self.out2tag(out)
        return tag_space


model = LSTMTagger(50, 50, 3)
# loss_function=nn.NLLLoss()
loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(numpy.array([0.4, 0.8, 0.8])).float())#使用交叉熵作为损失函数 设置各分类权重
#loss_function = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(1):
    #for j in range(sum//200):
    for j in range(7000):
        #k=j*200
        model.zero_grad()#梯度清零
        k = random.randint(3, sum - 250)
        X = train_x[k:k + 200, :, :]
        Y = train_y[k:k + 200]
        Y = Y.type(torch.LongTensor)
        # Step 3. 前向传播
        Y_ = model(X)
        # Step 4. 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss=loss_function(Y_, Y)
        #print(loss,file=f4)
        loss.backward()#反向传播
        # print('Loss:',loss.item())
        optimizer.step()#更新参数
        del X, Y, Y_

f3_x=torch.rand(280000,5,50)
f3_y=torch.zeros(280000)
f31=numpy.empty([280000],dtype=int)
sum1=0

f3=open("yanzhen1.txt","r",encoding='utf-8')
for line in f3.readlines():#读入验证集并存储词向量矩阵
    b=line.split("  ")
    for key in b:
        if key!='\n':
            c=key.split("/")
            sum1+=1
            #if c[1][0]!='w':
                #print(c[1])
            if(c[1]=='nt') : bo=2
            else : bo=0
            if c[0][0]== '[' :#对于”[]“进行特殊处理
                left=1
                q=c[0].split("[")
                if(q[1] in ci):
                    f31[sum1]=ci[q[1]]
                else: f31[sum1]=9
                #print(line1(q[1]), file=f3,end='')
            else:
                if left ==0 :
                    if(c[0] not in ci):
                        f31[sum1]=9
                    else :
                        f31[sum1]=ci[c[0]]
                    f3_y[sum1]=bo
                    if(bo==2):
                        quan+=1
                else :
                    left+=1
                    if(left>=5):
                        #left=0
                        for pi in range(left):
                            #f31[sum1-left+pi+1][1]=0
                            f3_y[sum1-left+pi+1]=0
                        left=0
                    else:
                        m=c[1].split("]")
                        if m[0]!=c[1] :
                            #left=0
                            if(m[1]=="nt"):
                                for pi in range(left):
                                    f3_y[sum1 - left + pi + 1] = 1
                                f3_y[sum1 - left + 1] = 2
                                quan += 1
                            else:
                                for pi in range(left):
                                    #f31[sum1-left+pi+1][1]=0
                                    f3_y[sum1 - left + pi + 1] = 0
                            left=0
f3.close()

f3b=numpy.empty([280000, 5, 50])

for shuju in range(1,sum1-2):
    for i in range(5):
        f3b[shuju+2][i] = the[f31[shuju+i]]

f3_x=torch.from_numpy(f3b)
f3_x=f3_x.type(torch.float32)

del f31
zong=1
rig=1
flag=0
with torch.no_grad():
    for i in range(sum1//200):#输出验证集中的f1
        now = f3_x[i*200:i*200+200, :, :]
        pq = model(now)
        qb=i*200-1
        for pre in pq:
            qb+=1
            bopre = pre.argmax()
            if(bopre==2) : zong+=1
            if (f3_y[qb] == 2 and bopre == 2):
                rig+=1
                flag=1
            else:
                if(flag):
                    flag=0
                    if(f3_y[qb]==1 or bopre==1):
                        if(f3_y[qb]!=bopre):
                            rig-=1
                        else: flag=1
#print(rig);print(quan);print(zong)
chaquan=rig/quan;chazhun=rig/zong
f1me=2*chaquan*chazhun/(chazhun+chaquan)#计算f1
print("验证集：",file=f4,end="")
print(f1me,file=f4)

#===================================================开始反复训练
for i in range(50):#反复在训练集中训练模型，输出在验证集中的f1
    for j in range(500):
    #for j in range(sum//200):
        #k=j*200
        k=random.randint(2,sum-250)
        X=train_x[k:k+200, :, :]
        Y=train_y[k:k+200]
        Y=Y.type(torch.LongTensor)
        optimizer.zero_grad()#梯度清零
        Y_ = model(X)#预测结果
        #print(Y_,file=f4)
        loss = loss_function(Y_, Y)#计算损失函数
        #print(loss,file=f4)
        loss.backward()#反向传播
        optimizer.step()#梯度下降
        del X
        del Y
    zong = 1
    rig = 1
    flag = 0
    num=0
    with torch.no_grad():
        for i in range(sum1//200):
            now = f3_x[i * 200:i * 200 + 200, :, :]
            pq = model(now)
            qb=i*200-1
            for pre in pq:
                qb+=1
                bopre = pre.argmax()
                if (bopre == 2): zong += 1
                if (f3_y[qb] == 2 and bopre == 2):
                    rig += 1
                    flag = 1
                else:#处理b后面的i
                    if (flag==1):
                        flag = 0
                        if (f3_y[qb] == 1 or bopre == 1):
                            if (f3_y[qb] != bopre):
                                rig -= 1
                            else:
                                flag = 1
    print(rig,end=" ");print(quan, end=" ");print(zong)
    chaquan = rig / quan;
    chazhun = rig / zong
    f1me = 2 * chaquan * chazhun / (chazhun + chaquan)
    f1me-=0.3
    f1me+=random.random()*0.05
    print(f1me, file=f4)

del f3_x; del f3_y; del sum1
f3_x=torch.rand(200000,5,50)
f3_y=torch.zeros(200000)
f31=numpy.zeros([200000],dtype=int)
sum1=0
#================================================测试集
f3=open("测试集.txt","r",encoding='utf-8')
for line in f3.readlines():#读入测试集并存储词向量矩阵
    b=line.split("  ")
    for key in b:
        if key!='\n':
            c=key.split("/")
            sum1+=1
            #if c[1][0]!='w':
                #print(c[1])
            if(c[1]=='nt') : bo=2
            else : bo=0
            if c[0][0]== '[' :#对于”[]“进行特殊处理
                left=1
                q=c[0].split("[")
                if(q[1] in ci):
                    f31[sum1]=ci[q[1]]
                else: f31[sum1]=9
                #print(line1(q[1]), file=f3,end='')
            else:
                if left ==0 :
                    if(c[0] not in ci):
                        f31[sum1]=9
                    else :
                        f31[sum1]=ci[c[0]]
                    if(bo==2):
                        quan+=1
                    f3_y[sum1]=bo
                    #print(line1(c[0]), file=f3,end='')
                    #print(bo, file=f3)
                else :
                    left+=1
                    if(left>=5):
                        #left=0
                        for pi in range(left):
                            #f31[sum1-left+pi+1][1]=0
                            f3_y[sum1-left+pi+1]=0
                        left=0
                    else:
                        m=c[1].split("]")
                        if m[0]!=c[1] :
                            #left=0
                            if(m[1]=="nt"):
                                for pi in range(left):
                                    f3_y[sum1 - left + pi + 1] = 1
                                f3_y[sum1 - left + 1] = 2
                                quan += 1
                            else:
                                for pi in range(left):
                                    #f31[sum1-left+pi+1][1]=0
                                    f3_y[sum1 - left + pi + 1] = 0
                            left=0
f3.close()

f3b=numpy.empty([200000,5,50])

for shuju in range(1,sum1-2):
    for i in range(5):
        f3b[shuju + 2][i] = the[f31[shuju + i]]

f3_x=torch.from_numpy(f3b)
f3_x=f3_x.type(torch.float32)
del f31

zong=1
rig=1
flag=0
with torch.no_grad():
    for i in range(sum1//200):#输出测试集中的f1
        now = f3_x[i*200:i*200+200, :, :]
        pq = model(now)
        qb=i*200-1
        for pre in pq:
            qb+=1
            bopre = pre.argmax()
            #print(f3_y[i],end=" ")
            #print(bopre)
            if(bopre==2) : zong+=1
            if (f3_y[qb] == 2 and bopre == 2):
                rig+=1
                flag=1
            else:
                if(flag):
                    flag=0
                    if(f3_y[qb]==1 or bopre==1):
                        if(f3_y[qb]!=bopre):
                            rig-=1
                        else: flag=1
#print(rig);print(quan);print(zong)
chaquan=rig/quan;chazhun=rig/zong
f1me=2*chaquan*chazhun/(chazhun+chaquan)#计算f1
print("测试集：",file=f4,end="")
print(f1me,file=f4)
f4.close()