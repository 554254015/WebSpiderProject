    本情景案例是想要获取客户价值，识别客户价值应用的最广泛的模型是RFM模型，三个字母分别代表Recency（最近消费时间间隔）、Frequency（消费频率）、
Monetary（消费金额）这三个指标。结合具体情景，最终选取客户消费时间间隔R、消费频率F、消费金额M这三个指标作为航空公司识别客户价值的指标。
    本案例简单选择三个指标进行K-Means聚类分析来识别出最优价值的客户。航空公司在真实的判断客户类别是，选取的观测维度要大得多。
    本情景案例的主要步骤包括：
    （1）对数据进行清洗处理，包括数据缺失与异常处理、数据属性的规约、数据清洗和变换，把数据处理成可使用的数据（Data）；
    （2）利用已预处理的数据（Data），基于RFM模型进行客户分群，对各个客户群进行特征分析，对客户进行分类；
    （3）针对不同类型的客户指定不同的销售政策，实行个性化服务。

