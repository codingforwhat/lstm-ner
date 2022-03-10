# lstm-ner

1 使用LSTM长短期记忆神经网络实现机器学习命名实体识别

定义类LSTMTagger，包含lstm神经网络层，linear线性顺序模型

初始化内容为lstm输入最后一维大小，隐藏层参数数量，模型输出维数

Lstm输入的最后一维大小为50，表示词向量长度

Batch_size为200，表示每一次随机梯度下降使用200个词窗口

使用relu激活函数，优化模型



2使用交叉熵求损失函数，设置各分类权重，增加正例在模型中的权重。

使用优化器自动梯度下降

通过对比使用Adam顺序优化器比SGD随机梯度优化效果更好

 

3每次训练分别进行梯度清零，正向传播，计算损失函数，反向传播，梯度下降（更新参数）



4最终f1达到0.87，训练的模型效果很好

5现代汉语切分链接：https://klcl.pku.edu.cn/zygx/zyxz/index.htm

词向量文件链接：https://github.com/jiesutd/LatticeLSTM

