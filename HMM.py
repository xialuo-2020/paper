import re
import codecs
from pandas import DataFrame as DF
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
#################################################################################################
#                                                                                               #
#                                  数据预处理                                                   #
#################################################################################################
fin = codecs.open('人民日报199801.txt', "r", "utf-8")
text = fin.read()
#去掉二级标记，地名ns、机构名称nt、其他专有名词nz
text = re.sub("\[", "", text)#将“[”替换为空格从而删除“[”
text = re.sub("]nt", "", text)
text = re.sub("]ns", "", text)
text = re.sub("]nz", "", text)
#去掉多个连续空格以及空行
text = re.sub("\n", "@", text)#先将换行符替换为@，方便去除空行个多个连续空格
text = re.sub("\s+", " ", text) #\s是指空白,包括空格、换行、tab缩进等所有的空白
text = re.sub("@", "\n",text)
text = re.sub(" \n","\n",text)
text = re.sub(" ", "@", text)
text = re.sub("\s+", "\n", text)
text = re.sub("@", " ", text)
s = text.encode("utf-8")
#################################################################################################
#                              表                                                          #
#################################################################################################
fout=open("HMM语料.txt", "wb") #将处理好的语料库放入HMM语料.txt
fout.write(s)
fout.close()
fin.close()
print("语料处理完成")

#################################################################################################
#                               切分出所有的词性储存在characters中                              #
#                        characters ：词性序列，记录语料库中的一个个不同的词性                  #
#   用正则表达式切割得到词性标记（状态）序列，开头的日期或者空格+非字母作为分隔符               #
#################################################################################################
fcontent = codecs.open('HMM语料.txt', "r", "utf-8")
content = fcontent.read()
fcontent.close()
characters =re.split("[0-9|-]*/|\\s{1,}[^a-zA-Z]*", content)
del characters[-1]
del characters[0]
##################################################################################################
#                               词性频数字典  characters_p                                      #
#                             characters_rmv：语料库中出现过的词性种类                          #
#                             phrase_rmv :语料库中出现过的词组种类                              #
#                             characters_p：词性频数，记录每种词性的出现频数                    #
###############################################################################################
characters_rmv = list(set(characters))
phrase_rmv = list(set(phrase))
characters_p = {}
for i in characters_rmv:
    characters_p[i] = 0
for j in characters:
    if (j in characters_rmv):
        characters_p[j]+=1
    else:
        print("不存在的词性")
##################################################################################################
#                  导出每种词性的出现频数（characters_p）的txt文件                               #
##################################################################################################
fcharacters_p = open('characters_p.txt', 'w')
for k, v in characters_p.items():
    fcharacters_p.write(str(k) + ' ' + str(v) + '\n')
fcharacters_p.close()

def estimate_parameter(data_train):
    # 转移概率Aij,t时刻由状态i变为状态J的频率
    # 观测概率Bj(k)，由状态J观测为K的概率
    # PAI i 初始状态q出现的频率
    # 先验概率矩阵
    pi = {}
    a = {}
    b = {}
    # 所有的词语
    ww = {}
    # 所有的词性
    pos = []
    # 每个词性出现的频率
    frep = {}
    # 每个词出现的频率
    frew = {}
    ####################################################################################################
    #                                  初始化pos,ww                                                  #
    ###################################################################################################
    for line in range(0, len(data_train)):
        temp = data_train[line].strip().split(" ")#将每行句子切割成（"单词/词性","单词/词性",...,"单词/词性")
        for i in range(1, len(temp)):
            word = temp[i].split("/")
            if (len(word) == 2):
                if (word[0] not in ww):
                    ww[word[0]] = 1

                if (word[1] not in pos):
                    pos.append(word[1])
    ww = ww.keys()
    ####################################################################################################
    #                                  初始化参数                                                  #
    ###################################################################################################
    for i in pos:
        pi[i] = 0
        frep[i] = 0
        a[i] = {}
        b[i] = {}
        for j in pos:
            a[i][j] = 0
        for j in ww:
            b[i][j] = 0
    for w in ww:
        frew[w] = 0
    # 行数
    line_num = 0
    ###################################################################################################
    #                             参数估计                                                            #
    #                   先验概率pi、转移概率矩阵A、发射概率矩阵B                                      #
    ###################################################################################################
    for line in range(0, len(data_train)):
        tmp = data_train[line].strip().split(" ")#将每行句子切割成（"单词/词性","单词/词性",...,"单词/词性")
        n = len(tmp)
        line_num += 1
        for i in range(1, n):
            word = tmp[i].split("/")#i位置上，（"词组","词性"）
            pre = tmp[i - 1].split("/")#i-1位置上，（"词组","词性"）
            # 计算词组频率
            frew[word[0]] += 1
            #计算词性频率
            frep[word[1]] += 1
            if i == 1:
                #统计每行句子第二个词性出现的频数
                pi[word[1]] += 1
            else:
                #[i-1位置的词性]转移到[i位置的词性]的转移概率,i = 2...n-1
                #(第2个词性转移到第3个词性,...,第n-1个词性到第n个词性)
                a[pre[1]][word[1]] += 1
            # [i位置的词组]发射到[i位置的词性]的转移概率,i = 1...n-1
            #(第2个词组发射到第2个词性,...,第n个词组到第n个词性)
            b[word[1]][word[0]] += 1
    ###################################################################################################
    #                             数据平滑 （Add-delta  smoothing）                                                         #
    #                                                                                   #
    ###################################################################################################
    for i in pos:
        pi[i] = float(pi[i]) / line_num
        for j in pos:
            if (a[i][j] == 0):
                a[i][j] = 0.5#防止出现概率矩阵稀疏的问题，δ=0.5
        for j in ww:
            if (b[i][j] == 0):
                b[i][j] = 0.5#防止出现概率矩阵稀疏的问题，δ=0.5
    for i in pos:
        for j in pos:
            a[i][j] = float(a[i][j]) / (frep[i])
            # frep为所有文本汇总后每个词性出现的频数
        for j in ww:
            b[i][j] = float(b[i][j]) / (frew[j])
            # frew为所有文本汇总后每个词性出现的频数
    # 导出矩阵A和矩阵B
    df = DF(a)
    df.to_csv("HMM模型A.csv")
    df = DF(b)
    df.to_csv("HMM模型B.csv", encoding='utf_8_sig')
    return a, b, pi, pos, frew, frep




def viterbi(a, b, pi, str_token, pos):
    # 计算文本长度
    num = len(str_token)
    # 概率转移路径
    dp = [{} for i in range(0, num)]
    # 状态转移路径
    pre = [{} for i in range(0, num)]
    for k in pos:
        for j in range(num):
            dp[j][k] = 0
            #dp表示处理到第j个单词，该单词词性为k的序列出现的概率 ，行为单词，列为词性
            pre[j][k] = ''
            # pre表示处理到第j个单词，该单词词性为k时，第j-1个词组的词性
    # 句子初始化状态概率分布（首个词在所有词性的概率分布）
    for p in pos:
        if (b[p].__contains__(str_token[0])):
        #str_token第一个词组在已出现的词语库中
            dp[0][p] = pi[p] * b[p][str_token[0]] * 1000
            # 0位置上的词组，或第1个词组，该词组词性为p的序列出现的概率=初始概率*词性p对应该单词的观测概率*1000，1000是为了保证精度够用
        else:
            # str_token第一个词组不在已出现的词语库中
            dp[0][p] = pi[p] * 0.5 * 1000
            #防止出现概率矩阵稀疏的问题，δ=0.5
    for i in range(0, num):
        for j in pos:
            if (b[j].__contains__(str_token[i])):
            #str_token第一个词组在已出现的词语库中
                sep = b[j][str_token[i]] * 1000
                #则发射概率已知，在B矩阵中
            else:
            #str_token第一个词组不在已出现的词语库中
                #发射概率未知,这个词不存在，应该置0.5/frew[str_token[i]]，这里默认为1
                sep = 0.5 * 1000
            for k in pos:
                # 计算本step i 的状态是j的最佳概率和step i-1的最佳状态k(计算结果为step i 所有可能状态的最佳概率与其对应step i-1的最优状态)
                if (dp[i][j] < dp[i - 1][k] * a[k][j] * sep):
                    dp[i][j] = dp[i - 1][k] * a[k][j] * sep
                    # 各个step最优状态转移路径
                    pre[i][j] = k
                    #记录下上一个词组的词性k

    resp = {}
    max_state = ""
    #首先找到最后输出的最大观测值的状态设置为max_state
    for j in pos:
        if (max_state == "" or dp[num - 1][j] > dp[num - 1][max_state]):
            max_state = j
    i = num - 1
    # 根据最大观测值max_state和前面求的pre找到概率最大的一条。
    while (i >= 0):
        resp[i] = max_state
        max_state = pre[i][max_state]
        i -= 1
    # for i in range(0, num):
    #     print(str_token[i] + "\\" + resp[i])
    return resp
# #######################################################################################################
# #                                 导入数据                                                            #
# #######################################################################################################
fhm = codecs.open('HMM语料.txt', "r", "utf-8")
data = fhm.readlines()
fhm.close()


#######################################################################################################
#                                 K折交叉验证    K=4时                                                #
#######################################################################################################
if __name__ == "__main__":
    #######################################################################################################
    #                                 划分训练集测试集                                                            #
    #######################################################################################################
    kf = KFold(n_splits=4, shuffle=True, random_state=100)
    data_train_index = [{} for i in range(0, 4)]
    data_test_index = [{} for i in range(0, 4)]
    data_train_k = [{} for i in range(0, 4)]
    data_test_k = [{} for i in range(0, 4)]
    data1 = np.array(data)
    kfd = 0
    m = {}
    n = {}
    for i in range(0, 4):
        n[i] = 0
        m[i] = 0
    for train_index, test_index in kf.split(data1):
        data_train_index[kfd] = train_index
        data_test_index[kfd] = test_index
        data_train_k[kfd] = data1[train_index].tolist()
        data_test_k[kfd] = data1[test_index].tolist()
        kfd += 1
    print("训练集、测试集划分完成")
    for kfold in range(0, len(data_train_k)):
        data_train = data_train_k[kfold]
        pline_train = [{} for i in range(0, len(data_train))]
        cline_train = [{} for i in range(0, len(data_train))]
        for j in range(0, len(data_train)):
            pline_train[j] = (re.split("(/[a-zA-Z]*\\s{0,})", data_train[j]))[:-1:2]
            cline_train[j] = (re.split("[0-9|-]*/|\\s{1,}[^a-zA-Z]*", data_train[j]))[1:-1]
        #######################################################################################################
        #                                 参数训练                                                            #
        #######################################################################################################
        a, b, pi, pos, frew, frep = estimate_parameter(data_train)
        print(pi)
        print("第"+str(kfold)+"次"+"参数π、A、B训练完成")
        #######################################################################################################
        #                                 维特比算法                                                            #
        #######################################################################################################
        for i in range(0, len(pline_train)):
            str_token = pline_train[i]
            tag = cline_train[i]
            resp = {}
            resp = viterbi(a, b, pi, str_token, pos)
            for j in range(0, len(str_token)):
                m[kfold] += 1
                if (resp[j] == tag[j]):
                    n[kfold] += 1
        print("第" + str(kfold) + "次"+"测试集上测试效果为：")
        print("准确率 = "+str(n[kfold]/m[kfold]))







