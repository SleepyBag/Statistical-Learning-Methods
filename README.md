# Statistical-Learning-Method
I want to implement all the algorithms introduced by 统计学习方法, 李航
中文文档请往下翻。

## Feature

- **Complete**
All the algorithms introduced by this book are implemented, including
  - kNN powered by kd-tree.
  - max entropy model. I cannot find any other repo that implements this algorithm.
  - linear chain conditional random field. I cannot find this model in any other similar repo.
  - HMM powered by baum-welch. Most repos only provide with HMM trained by counting.
- **Detailed**.
All the algorithms are implemented thoroughly. I try my best not to skip any detail. For example,
  - about how to select the best one of trimmed CART by cross-validation, I asked Dr. Li Hang by e-mail and got detailed answer. Thanks a lot to Dr. Li Hang for his patiance and kindness.
- **Matrix calculation**.
Strip off `for` loops. Implement most of the algorithms with matrix calculation supported by `numpy`.
- **Extensible**.
It is easy to fit the codes with new datasets because all the algorithms are controllable through parameters, to a large extent.
- **Examples**
Each algorithm comes with some examples. Just run the model file and you will see the examples. If you have better examples for others to understand the model, please feel free to start a PR.

## Dependencies

- Python3
- numpy
- matplotlib
- [rich](https://github.com/willmcgugan/rich)

## Usage

Just run any single file located in each chapter. You will see examples of the algorithm.

---

# 统计学习方法

## 项目特色

GitHub 上有许多实现《统计学习方法》的仓库。本仓库与它们的不同之处在于：

- **完整**
实现了**所有**模型。包括
  - KD 树支持的 KNN 模型。
  - **最大熵模型**。我没有找到其他任何一个仓库实现了该算法。
  - **线性链条件随机场**。我同样没有找到其他任何一个仓库实现了该算法。这个模型花费了我一个月的时间去理解和实现。
  - Baum-Welch 算法支持的 HMM 算法。大多数仓库实现的 HMM 算法都是简单的计数模型。
- **细节**
所有的算法我都在尽力**完全**实现。比如说
  - 有关如何用交叉验证法选取剪枝的 CART 树，我特意邮件询问了李航博士并得到了耐心的解答。在此非常感谢李航博士的支持！
- **矩阵运算**
我不喜欢用循环。你可以看到本仓库中的算法使用了大量的矩阵运算来避免使用循环。
- **可扩展性**
其他仓库的算法可能会在可扩展性上偷懒。比如 GMM 模型可能只实现了两个聚类的简单版本用于演示。而本仓库中的算法尽量将所有可调节部分作为模型参数，以供自由修改使用。
- **示例**
每个算法都加上了我认为会增强读者对算法理解的例子。当然我认为这部分目前还是不太完善的。如果你对如何举例有更好的间接，欢迎提 PR。

## 项目依赖

- Python3
- numpy
- matplotlib
- [rich](https://github.com/willmcgugan/rich)

## 如何使用

直接使用 Python 运行任意一个文件夹内的模型文件，你就可以看到算法示例了。

## 目录

- [第 2 章 - 感知机](2.Perceptron)
  - [感知机](2.Perceptron/perceptron.py)
- [第 3 章 - k 近邻法](3.KNN)
  - [k 近邻模型](3.KNN/knn.py)
  - [k 近邻模型 - 使用 KD 树实现](3.KNN/knn_kdtree.py)
- [第 4 章 - 朴素贝叶斯法](4.NaiveBayes)
  - [使用极大似然估计的朴素贝叶斯模型](4.NaiveBayes/NaiveBayesMLE.py)
  - [使用贝叶斯估计的朴素贝叶斯模型](4.NaiveBayes/NaiveBayesMAP.py)
- [第 5 章 - 决策树](5.DecisionTree)
  - [ID3 决策树](5.DecisionTree/ID3.py)
  - [C4.5 决策树](5.DecisionTree/C4.5.py)
  - [决策树剪枝算法](5.DecisionTree/prune.py)
  - [分类 CART 决策树](5.DecisionTree/ClassificationCART.py)
  - [分类 CART 决策树剪枝算法](5.DecisionTree/pruneClassificationCART.py)
  - [回归 CART 决策树](5.DecisionTree/RegressionCART.py)
- [第 6 章 - 逻辑斯谛回归与最大熵模型](6.LogisticRegression-MaxEntropy)
  - [逻辑斯谛回归模型](6.LogisticRegression-MaxEntropy/BinaryLogisticRegression.py)
  - [最大熵模型](6.LogisticRegression-MaxEntropy/MaxEntropy.py)
- [第 7 章 - 支持向量机](7.SVM)
  - [支持向量机](7.SVM/SVM.py)
- [第 8 章 - 提升方法](8.Boosting)
  - [AdaBoost](8.Boosting/AdaBoost.py)
  - [梯度提升树](8.Boosting/GBDT.py)
- [第 9 章 - EM 算法及其推广](9.EM)
  - [高斯混合模型](9.EM/GMM.py)
- [第 10 章 - 隐马尔科夫模型](10.HMM)
  - [前向算法](10.HMM/Forward.py)
  - [后向算法](10.HMM/Backward.py)
  - [维特比算法](10.HMM/Viterbi.py)
  - [Baum-Welch 算法](10.HMM/BaumWelch.py)
  - [使用 Baum-Welch 算法训练的隐马尔可夫模型](10.HMM/HMM.py)
- [第 11 章 - 条件随机场](11.ConditionalRandomField)
  - [线性链条件随机场](11.ConditionalRandomField/LinearChainConditionalRandomField.py)
- [第 14 章 - 聚类方法](14.Cluster)
  - [层次聚类](14.Cluster/Agglomerative.py)
  - [k 均值聚类](14.Cluster/KMeans.py)
- [第 15 章 - 奇异值分解](15.SVD)
  - [奇异值分解](15.SVD/SVD.py)
- [第 16 章 - 主成分分析](16.PCA)
  - [主成分分析](16.PCA/PCA.py)
- [第 17 章 - 潜在语义分析](17.LSA)
  - [潜在语义分析模型](17.LSA/LSA.py)
- [第 18 章 - 概率潜在语义分析](18.PLSA)
  - [概率潜在语义分析模型](18.PLSA/PLSA.py)
- [第 19 章 - 马尔可夫蒙特卡罗法](19.MCMC)
  - [Metropolis-Hasting 算法](19.MCMC/MetropolisHasting.py)
  - [单分量的 Metropolis-Hasting 算法](19.MCMC/SingleComponentMetropolisHasting.py)
  - [吉布斯采样](19.MCMC/GibbsSampling.py)
- [第 20 章 - 潜在狄利克雷分配](20.LDA)
  - [潜在狄利克雷分配模型](20.LDA/LDA.py)
- [第 21 章 - PageRank 算法](21.PageRank)
  - [PageRank 算法](21.PageRank/PageRank.py)
