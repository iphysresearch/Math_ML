# 机器学习之概率统计基础

[TOC]

## 引言

### 究竟什么是机器学习？

> - A branch of artificial intelligence, concerns the construction and study of systems that can learn from data.
>
>   Ref：https://en.wikipedia.org/wiki/Machine_learning
>
> - 机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。

### 概率 vs. 统计

- 概率：研究随机事件出现的可能性的数学分支，描述非确定性（Uncertainty）的正式语言，是统计推断的基础
  - 概率：一个事件或事件集合出现的可能性
  - <u>基本问题：给定以一个数据产生过程，则输出的性质是什么</u>。
- 统计推断：处理数据分析和概率理论的数学分支，与数据挖掘和机器学习是近亲
  - 统计量：一个用以描述样本或总体性质的数值，如均值或方差
  - <u>基本问题：给定输出数据，我们可以得到该数据的产生过程的哪些信息</u>（相当于是“概率”的一个逆过程）

![概率与统计推断](./images/概率与统计推断.png)

### 概率统计基础的重要性

- 研究数据分析必须大好概率和统计基础
  - Using fancy tools like neural nets, boosting and support vector machines without understanding basic statistics like **doing brain surgery before knowing how to use a band-aid.**
  - 做脑外科手术，却不知道怎么使用绷带！

### 参考书推荐

- Kevin P. Murphy, **Machine Learning: A Probabilistic Perspective**, MIT Press, 2012
  - 统计部分涵盖Larry了的内容，本笔记所用的符号体系来自此书。
- Larry Wasserman, **All of Statistics: A Concise Course in Statistical Inference**
  - 中译本：《**统计学完全教程**》
  - 内容很全，但有些部分篇幅略少，更偏向于从统计的角度讲述，为计算机系所写
- 盛聚／谢式千／潘承毅，**概率论与数理统计**，高等教育出版社
  - 浙大的这本很不错，国产书算是不错的。




## 概率

### 一、 概率公理及推论（基石）

#### 概率

- **频率学派**：多次重复试验（多次抛掷硬币），则我们期望正面向上的次数占总实验次数的一半。
- **Bayesian 学派**：我们相信下一次试验中，硬币正面向上的可能性为0.5.
  - 概率是我们对事情不确定性的定量描述，与信息有关，而非重复实验。
  - 可以用来对不能重复实验的时间的不确定，如明天天晴的概率为0.8，如某邮件是垃圾邮件的概率。

#### 样本空间和事件

- 将某个事先不知道输出的试验的所有可能结果组成的集合称为该试验的**样本空间**。
- 某试验的样本空间的子集为试验的**随机事件**。

#### 概率公理

- 对某随机试验的每个事件 $A$，赋予一个实数 $\mathbb{P}(A)$，称为 $A$ 的概率，如果集合函数 $P(\cdot)$ 满足下列条件：

  - **非负性**：对于每一个事件 $A$，有 $\mathbb{P}(A)\ge0$

  - **规范性**：对于必然事件 $S$（样本空间），有 $\mathbb{P}(S)=1$

  - **可列可加性**：对两两不相交（互斥）事件 $A_i$，即对于 $A_iA_j\neq\emptyset,i\neq j,i,j=1,2,\dots$，有
    $$
    \mathbb{P}\Big(\bigcup^\infty_{i=1}A_i\Big)=\sum^\infty_{i=1}\mathbb{P}(A_i)
    $$






从上述三个公理，可推导出概率的所有的其他性质。

- 推论：

  - 不可满足命题的概率为0

  $$
  \mathbb{P}(\emptyset)=0 \\
  \mathbb{P}(A\cap\bar{A)}=0
  $$

  - 对任意两个事件 $A$ 和 $B$ 

  $$
  \mathbb{P}(A\cup B)=\mathbb{P}(A)+\mathbb{P}(B)-\mathbb{P}(A\cap B)	
  $$

  - 事件 $A$ 的补事件

  $$
  \mathbb{P}(\bar{A})=1-\mathbb{P}(A)
  $$

  - 对任意事件 $A$

  $$
  0 \geq \mathbb{P}(A) \geq 1
  $$








#### 联合概率 & 条件概率

- **联合概率**表示两个事件共同发生的概率。$A$ 与 $B$ 的联合概率表示为 $\mathbb{P}(A,B)$ 或者 $\mathbb{P}(A\cap B)$。对任意两个事件 $A$ 和 $B$，
  $$
  \mathbb{P}(A,B)=\mathbb{P}(A\cap B)=\mathbb{P}(A|B)\mathbb{P}(B)
  $$
  该公式定义下的两种极端情形会反映出性质：统计独立性[^1]和互斥性[^2]。

- **条件概率**是指事件A在另外一个事件B已经发生条件下的发生概率。条件概率表示为：$\mathbb{P}(A|B)$，读作“在B条件下A的概率”。当 $\mathbb{P}(B)>0$ 时，给定 $B$ 时 $A$ 的条件概率为
  $$
  \mathbb{P}(A|B)=\frac{\mathbb{P}(A,B)}{\mathbb{P}(B)}\Rightarrow\mathbb{P}(A,B)=\mathbb{P}(A|B)\mathbb{P}(B)
  $$
  所谓**积规则(Product Rule)**。有穷多个事件的积事件情况：

  设 $A_1,A_2,…,A_n$ 为任意 $n$ 个事件（$n\ge2$）且 $\mathbb{P}(A_1,A_2,…,A_n)>0$，则
  $$
  \mathbb{P}(A_1,A_2,\cdots,A_n)=\mathbb{P}(A_1)\mathbb{P}(A_2|A_1)\cdots\mathbb{P}(A_n|A_1A_2\cdots A_{n-1})
  $$

  [^1]: **当且仅当**两个随机事件 $A$ 与 $B$ 满足 $\mathbb{P}(A\cap B)=\mathbb{P}(A)\mathbb{P}(B)$ 时，事件 $A$ 与 $B$ 才是**统计独立的**。同样，对于两个独立事件 $A$ 与 $B$ 有 $\mathbb{P}(A|B)=\mathbb{P}(A)$ 以及 $\mathbb{P}(B|A)=\mathbb{P}(B)$。
  [^2]: **当且仅当**两个随机事件 $A$ 与 $B$ 满足 $\mathbb{P}(A|B)=0$ 且 $\mathbb{P}(A)\neq0,\mathbb{P}(B)\neq0$ 时，事件 $A$ 与 $B$ 是**互斥的**。因此，对于两个独立事件 $A$ 与 $B$ 有 $\mathbb{P}(A|B)=0$ 以及 $\mathbb{P}(B|A)=0$。

- 给定任意 $B$，若 $\mathbb{P}(B)>0$，则 $ \mathbb{P}(\cdot|B)$ 也是一个概率，即满足概率的三个概率公理。
  $$
  \mathbb{P}(A|B)\geqslant0 \,\,\text{非负性}\\
  \mathbb{P}(\Omega|B) = 1\,\,\,\,\text{规范性}\\
  \mathbb{P}\Big(\bigcup^\infty_{i=1}A_i|B\Big)=\sum^\infty_{i=1}\mathbb{P}\big(A_i|B\big)\,\,\,\,\text{可列可加性}
  $$






#### 贝叶斯公式

- 设 $B_1,B_2,…,B_n$ 是一组事件，若

  1.  $i\neq j\in\{1,2,…,n\},B_iB_j=\emptyset$

  2. $B_1\cup B_2\cup…\cup B_n=\Omega$

    则称 $B_1,B_2,…,B_n$ 样本空间 $\Omega$ 的一个**划分**，或称为样本空间 $\Omega$ 的一个**完备事件组**。


- **全概率公式**：令 $A_1,…,A_K$ 为 $A$ 的一个划分，则对任意事件 $B$，有
  $$
  \mathbb{P}(B)=\sum_j\mathbb{P}(B|A_j)\mathbb{P}(A_j)
  $$







- **贝叶斯公式**：令 $A_1,…,A_k$ 为 $A$ 的一个划分且对每个 $k,k=1,2,…,K$。若 $\mathbb{P}(B)>0$，则对每个 $\mathbb{P}(A_k)>0$ 有（[由条件概率推导贝叶斯公式][https://en.wikipedia.org/wiki/Bayes%27_theorem#Derivation]）
  $$
  \mathbb{P}(A_k|B)=\frac{\mathbb{P}(B,A_k)}{\mathbb{P}(B)}=\frac{\mathbb{P}(B|A_k)\mathbb{P}(A_k)}{\sum_j\mathbb{P}(B|A_j)\mathbb{P}(A_j)}
  $$
  注：上述公式中，分子分母形式相同，分母为所有划分之和。

  注：**先验概率** $\mathbb{P}(A_k)$ 是事情还没有发生，要求这件事情发生的可能性的大小。它往往作为“由因求果”问题中的“因”出现；**后验概率** $\mathbb{P}(A_k|B)$ 是事情已经发生，要求这件事情发生的原因是由某个因素引起的可能性的大小。是“执果寻因”问题中的“因”。后验概率的计算要以先验概率为基础。


- 例：贝叶斯公式应用：

  - 已知Macintosh用户、Windows用户和Linux用户感染病毒的概率（先验概率）分别是0.65、0.82和0.5。用户使用Macintosh、Windows和Linux操作系统的比例分别为0.3、0.5和0.2。问现在发现用户感染了病毒，则该用户为Windows用户的概率（后验概率）是多少？

  - 解：令 $B$ 表示被病毒感染这一随机事件，

    ​	$A_1$ 表示Macintosh用户，则 $\mathbb{P}(A_1)=0.3,\mathbb{P}(B|A_1)=0.65$
    ​	$A_2$ 表示Windows用户，则 $\mathbb{P}(A_2)=0.5, \mathbb{P}(B|A_2)=0.82$
    ​	$A_3$ 表示Linux用户，则 $\mathbb{P}(A_3)=0.2,\mathbb{P}(B|A_3)=0.5$

    ​	因为 $\mathbb{P}(A_1)+\mathbb{P}(A_2)+\mathbb{P}(A_3)=1$，所以 $A_1$、$A_2$ 和 $A_3$ 组成一个完备事件组。

    ​	所以根据贝叶斯公式，
    $$
    \mathbb{P}(A_2|B)=\frac{\mathbb{P}(B,A_2)}{\mathbb{P}(B)}=\frac{\mathbb{P}(B|A_2)\mathbb{P}(A_2)}{\sum_i\mathbb{P}(B|A_i)\mathbb{P}(A_i)}=\frac{0.82\times0.5}{0.65\times0.3+0.82\times0.5+0.5\times0.2}=0.58
    $$

  - 作业题：假设在考试的多项选择中，考生知道正确答案的概率为 $p$，猜测答案的概率为 $1-p$，并且假设考生知道正确答案答对题的概率为1，猜中正确答案的概率为 $1/m$，其中$m$为多选项的数目。那么已知考生答对题目，求他知道正确答案的概率。

  - 解：令 $B$ 表示某一考生答对题目这一随机事件，

    ​	$A_1$ 表示知道正确答案的学生，则 $\mathbb{P}(A_1)=p,\mathbb{P}(B|A_1)=1$

    ​	$A_2$ 表示猜答案的学生，则 $\mathbb{P}(A_2)=1-p,\mathbb{P}(B|A_2)=1/m$

    ​	因为 $\mathbb{P}(A_1)+\mathbb{P}(A_2)=1$，所以 $A_1$ 和 $A_2$ 组成一个完备事件组。

    ​	所以根据贝叶斯公式，
    $$
    \mathbb{P}(A_1|B)=\frac{\mathbb{P}(B,A_1)}{\mathbb{P}(B)}=\frac{\mathbb{P}(B|A_1)\mathbb{P}(A_1)}{\sum_i\mathbb{P}(B|A_i)\mathbb{P}(A_i)}=\frac{1\times p}{1\times p+(1-p)\times 1/m}=\frac{mp}{(m-1)p+1}
    $$






### 二、随机变量及其分布：pmf／pdf、CDF、均值、方差...

#### 随机变量

- 机器学习与**数据**相关。随机变量就是讲随机事件与数据之间联系起来的**纽带**。

- **随机变量**是一个映射／函数 $X:\Omega\rightarrow\mathbb{R}$，将一个实数值 $X(\omega)$ 赋值给一个试验的每一个输出 $\omega$.

- 例：抛10次硬币，令 $X(\omega)$ 表示序列 $\omega$ 中正面向上的次数，如当 $\omega=HHTHHTHHTT$ ，则 $X(\omega)=6$。
  - $X$ 只能取离散值的话，称为**离散型**随机变量。
  - 注：<u>随机变量的取值随试验的结果而定，在试验之前不能预知它取什么值，且它的取值有一定的概率</u>。这些性质显示了随机变量与普通函数有着本质的区别。

- 例：令 $\Omega=\{(x,y):x^2+y^2\le1\}$ 表示单位圆盘，输出为该圆盘中的一点 $\omega=(x,y)$，则可以有随机变量：
  $$
  X(\omega)=x,Y(\omega)=y,Z(\omega)=\sqrt{x^2+y^2}
  $$

  - $X$ 取连续值，称为**连续型**随机变量。

#### 数据和统计量

- **数据**是随机变量的具体值
- **统计量**是数据／随机变量的任何函数
- <u>任何随机变量的函数仍然是随机变量</u>
  - 统计量也是随机变量

#### 累计分布函数CDF

- 令 $X$ 为一随机变量，$x$ 为 $X$ 的一具体值（数据），

  则随机变量 $X$ 的**累积分布函数** $F:\mathbb{R}\rightarrow[0,1]$ (cumulative distribution function, CDF) 定义为
  $$
  F(x) = \mathbb{P}(X\le x),-\infty<x<\infty
  $$

  其中 $x$ 是任意实数。对任意 $x_1,x_2 \,\,(x_1<x_2)$，有
  $$
  P\{x_1<X\le x_2\}=P\{X\le x_2\}-P\{X\le x_1\}=F(x_2)-F(x_1)
  $$

- CDF是一个非常有用的函数：<u>包含了随机变量的所有信息，完整地描述了随机变量的统计规律性</u>。

- 基本性质：

  1. $F(x)$ 是一个非降的函数。对任意 $x_1,x_2 \,\,(x_1<x_2)$，有
     $$
     F(x_2)-F(x_1)=P\{x_1<X\le x_2\}\ge0
     $$

  2. $0\le F(x)\le1$，且（规范的）
     $$
     F(-\infty)=\lim_{x\rightarrow-\infty}F(x)=0\\
     F(\infty)=\lim_{x\rightarrow\infty}F(x)=1
     $$






#### 概率（质量）函数 pmf

- 离散型随机变量的**概率函数** (probability function or probability mass function, pmf) 定义为
  $$
  p(x)=\mathbb(X=x)
  $$

- 性质：

  1. 对所有的 $x\in\mathbb{R},p(x)\ge0$
  2. $\sum_ip(x_i)=1$
  3. CDF与pmf之间的关系为：$F(x)=\mathbb{X\le x}=\sum_{x_i\le x}p(x_i)$ 

#### 概率密度函数 pdf

- 对连续性随机变量 $X$，如果存在一个函数 $p$，使得对所有的 $x,p(x)\ge0$，且对任意 $a\ge b$ 有
  $$
  \mathbb{P}(a<X\le b)=\int^b_ap(x)dx
  $$
  注：$p(x)$ 不必 $<1$，且 $p(x)\ge0$

  函数 $p$ 被称为**概率密度函数** (probability density function, pdf)

- 当 $F$ 可微时，

  1. $F(x)=\int^x_{-\infty}p(t)dt$

  2. $\int^\infty_{-\infty}p(x)dx=1$

  3. 对任意 $x_1,x_2 \,\,(x_1<x_2)$，有
     $$
     P\{x_1<X\le x_2\}=F(x_2)-F(x_1)=\int^{x_2}_{x_1}p(x)dx
     $$

  4. 若 $p(x)$ 在点 $x$ 处连续，则有 $F'(x)=p(x)$

![WX20170725-110108](./images/WX20170725-110108.png)

#### 分布的概述

- 除了概率分布函数，有时我们也采用一些单值描述量来刻画某个分布的性质。
  - 位置描述：期望／均值、中值、众数、分位数
  - 散布程度描述：方差、四分位矩形（IQR）

##### 期望

- 期望／均值：随机变量的平均值

  - 概率加权平均

- 如果下列积分有定义的话（$\int|x|dF(x)<\infty$），定义 $X$ 的期望（均值，一阶矩）为：
  $$
  \mathbb{E}(X)=\mu=\int xdF(x)=\int xp(x)dx
  $$
  数学期望 $\mathbb{E}(X)$ 完全由随机变量 $X$ 的概率分布所决定。

- 离散情况下为：$\sum_x xp(x)$

- 期望的性质：

  - 线性运算：$\mathbb{E}(aX+b)=a\mathbb{E}(X)+b$

  - 加法规则：设 $X_1,\dots,X_N$ 是随机变量，$a_1,\dots,a_N$ 是常量，有
    $$
    \mathbb{E}\Big(\sum_ia_iX_i\Big)=\sum_ia_i\mathbb{E}(X_i)
    $$

  - 乘法规则：设 $X_1,\dots,X_N$ 是相互独立的随机变量，有
    $$
    \mathbb{E}\Big(\prod^N_{i=1}X_i\Big)=\prod^N_{i=1}\mathbb{E}(X_i)
    $$

- 例：期望

  - 期望是随机变量的一个很好单值概述
  - ![WX20170725-145737](./images/WX20170725-145737.png)

##### 众数（mode）

- 众数：设随机变量 $X$ 有密度 $p(x)$，且存在 $x_0$ 满足
  $$
  x_0=\arg \max_x p(x)
  $$
  则称 $x_0$ 为 $X$ 的**众数**。

  - 刻画随机变量**出现次数最多**的位置。

- 期望、中位数和众数都称为**位置函数**。

  - 当随机变量的分布为高斯分布时，三者相等。
  - <u>贝叶斯估计中最大后验估计（MAP）就是后验分布的最大值／众数</u>。

##### 中值（Median）

-   分布的中值可视为分布的中间，即在其上下的概率均为 0.5:

  $$
  Median(X):=x^*:P(X\ge x^*)=0.5
  $$

  - 中值是分布另一个很好的单值概述（对噪声并不敏感）
      - 若我们用 $L_2$ 距离度量一个随检变量 $X$ 与一个常数 $b$ 的距离，则当 $b$ 为中值时，随机变量 $X$ 与 $b$ 的距离最小。
      - Recall：$L_2$ 距离度量下，随机变量 $X$ 与其均值的距离最小。


##### 分位函数（quantile）

-   令随机变量 $X$ 的CDF为 $F$，CDF的反函数或分位函数 (quantile function) 定义为
    $$
    F^{-1}_X(\alpha) = \inf\{x: F_x(x)\ge\alpha\}
    $$
    其中，$\alpha\in[0,1],inf: \text{下界}$ 。若 $F$ 严格递增并且连续，则 $F^{-1}_X(x)$ 为一个唯一确定的实数 $x$，使得 $F^{-1}_X(x)=\alpha$ 。

- 例：

    ![WX20170725-153543](./images/WX20170725-153543.png)

    上图中，**中值**（median）：$F_X^{-1}(0.5)$；上下1/4分位数：$F_X^{-1}(0.25),F_X^{-1}(0.75)$


##### 分位数（cont.）

-   对正态分布 $Z\sim\mathcal{N}(0,1)$，其CDF的反函数记为 $\Phi^{-1}$

  - 当 $\alpha=0.05$ 时，随机变量 95% 即 $(1-\alpha)$ 的概率会落在区间：
    $$
    \Big(\Phi^{-1}\Big(\frac{\alpha}{2}\Big),\Phi^{-1}\Big(1-\frac{\alpha}{2}\Big)\Big)=(-1.96,1.96)
    $$







![WX20170725-154803](./images/WX20170725-154803.png)

##### 方差

-   $X$ 的 $k$ 阶矩定义为 $\mathbb{E}(X^k)$，假设 $\mathbb{E}(X^k)<\infty$

  - 若 $X$ 有均值 $\mu$，则其**方差**（二阶中心距）
    $$
    \sigma^2=\mathbb{V}(X)=\mathbb{E}(X-\mu)^2=\int(x-\mu)^2dF(x)
    $$
    且标准差 $\sigma=sd=\sqrt{\mathbb{V}(X)}$
    $$
    \mathbb{V}(X)=\mathbb{E}\Big[(X-\mu)^2\Big]=\mathbb{E}(X^2)-\mu^2
    $$

  - 方差：刻画随机变量围绕均值的**散布程度**

    方差越大，$X$ 变化越大；方差越小，$X$ 与均值越接近。

  - 方差的性质

    - 设方差有定义，则有以下性质：


1.   $\mathbb{V}(a)=0$，$a$ 为常数
  2. $\mathbb{V}(X)=\mathbb{E}(X)^2-\mu^2$
  3. 当 $a,b$ 是常数时，$\mathbb{V}(aX+b)=a^2\mathbb{V}(X)$
  4. 如果 $X_1,\dots,X_N$ 独立，且 $a_1,\dots,a_N$ 为常数，则

2. $$
     \mathbb{V}\Big(\sum^N_{i=1}a_iX_i\Big)=\sum^N_{i=1}a^2_i\mathbb{V}(X_i)
     $$
     ​     注意：

     ​     I. 期望的加法规则无需独立条件；

     ​     II. 不独立随机变量和的方差计算需考虑变量之间的协方差（下节课）

##### IQR（Interquartile Range）

- 中值是比均值更鲁棒的分布的中心度量

- 比方差更鲁棒的分布的散布范围的度量是**四分位矩** (Interquartile Range, IQR)：25%分位数到75%分位数之间的区间

  - IQR 在 boxplot（seaboard.boxplot）中用到：分布的图像概述

  - 长方形为IQR

  - 中间的线为中值

  - 两头的虚线：1.5IQR

  - 超过上下限的数据为噪声点，用 '*' 或 '+' 等符号表示

      ![WX20170725-161643](./images/WX20170725-161643.png)




### 三、常见随机变量概率分布

#### 离散型随机变量

##### 贝努里（Bernoulli）分布

- Bernoulli 分布又名亮两点分布或者 0-1 分布。若 Bernoulli 试验成功，则 Bernoulli 随机变量 $X$ 取值为1，否则 $X$ 为0。记试验成功概率为 $\theta$，即
  $$
  \mathbb{P}(X=1)=\theta,\mathbb{X=0}=1-\theta,\theta\in[0,1]
  $$

- 我们称 $X$ 服从**参数**为 $\theta$ 的 **Bernoulli 分布**，记为 $x\sim Ber(\theta)$

$$
\begin{aligned}
p(x|\theta) =
& \begin{cases}
\theta&\text{if }x=1,\\
1-\theta&
\text{if }x=0.
\end{cases} \\
 =& \, \theta^x(1-\theta)^{1-x},\text{for } x\in[0,1]
\end{aligned}
$$

- Bernoulli 分布的均值： $\mu=\theta$
- 方差：$\sigma^2=\theta(1-\theta)$
- 两类分类问题：$y|x$ 服从 bernoulli 分布，即类别标签 $y$ 取值为0或1的离散随机变量

##### 二项（binomial）分布

- 二项(Binomial)分布：在抛掷硬币试验中，若只进行一次试验，则为bernoulli试验。若进行n次试验，则硬币正面向上的数目$X$满足两项分布，记为 $x\sim Bin(n,\theta)$ （两个参数$n,\theta$）
  $$
  p(x|n,\theta)=\begin{pmatrix} n  \\ x  \end{pmatrix} \theta^x(1-\theta)^{n-x}
  $$
  其中 $\begin{pmatrix} n  \\ x  \end{pmatrix} \triangleq \frac{n!}{(n-x)!x!},x\in\{0,\dots,n\}$

- 二项分布的均值：$\mu=n\theta$

- 方差：$\theta^2=n\theta(1-\theta)$

##### 多项（multinomial）分布

- 假设抛有$K$个面的骰子，其中抛掷到第j面的概率为$\theta_j$，令 $\boldsymbol{\theta}=(\theta_1,\dots,\theta_K)$ <向量>
- 若一共抛掷n次，$\boldsymbol{x}=(x_1,\dots,x_K)$为随机向量，其中$x_k$表示抛掷到第k面的次数，则x的分布为多项分布，即 $x\sim Mu(n,\theta)$

$$
p(\boldsymbol{x}|n,\boldsymbol{\theta})=\begin{pmatrix} n  \\ x_1,\dots,x_K  \end{pmatrix} \prod^K_k\theta^{x_k}_k,\,\,\begin{pmatrix} n  \\ x_1,\dots,x_K   \end{pmatrix} \triangleq \frac{n!}{x_1!\dots x_K!}
$$

- 当 $n=1$ 时为 $Mu(\boldsymbol{x}|1,\boldsymbol{\theta}),\,p(\boldsymbol{x}|1,\boldsymbol{\theta})=\prod^K_k\theta_k^{x_k}$

###### 分类分布

Multinomial分布的特例：

- 当$n=1$时，记为类别型分布（**Categorical 分布**）
  $$
  Cat(\boldsymbol{x}|\boldsymbol{\theta})\triangleq Mu(\boldsymbol{x}|1,\boldsymbol{\theta})
  $$

- 由于 $\boldsymbol{x}$ 中 $K$ 维数据中只有一个为1，其余均为0，我们将其写成另一种形式

$$
x\sim Cat(\boldsymbol{\theta}), \,\,\text{则 }p(x=k|\boldsymbol{\theta})=\theta_k
$$

- 多类分类问题：$y|\boldsymbol{x}$ 服从 Categorical 分布，即类别标签 $y$ 取值为0到$K$之间整数的离散随机变量。

#### 连续型随机变量

##### 均匀分布

- $X\sim \text{Unif}(a,b)$ 
  $$
  \begin{aligned}
  p(x) =
  & \begin{cases}
  \frac{1}{b-a}& x\in[a,b]\\
  0&
  otherwise.
  \end{cases} \\
  \end{aligned}
  $$






![WX20170726-165538](./images/WX20170726-165538.png)

##### 高斯分布（最常用）

- 高斯分布／正态分布：$X\sim \mathcal{N}(\mu,\sigma^2)$ （两个参数）
  $$
  p(x|\mu,\sigma^2)=\mathcal{N}(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}
  \exp\Big\{\frac{1}{2\sigma^2}(x-\mu)^2\Big\},x\in\mathbb{R},\sigma>0
  $$
  其中 $\mu,\sigma^2$ 分别为高斯分布的均值和方差。

  ![WX20170726-170621](./images/WX20170726-170621.png)

- **高斯分布**的CDF为
  $$
  \Phi(x|\mu,\sigma^2)=\int^\infty_{-\infty}\mathcal{N}(z|\mu,\sigma^2)dz
  $$
  ![WX20170726-171356](./images/WX20170726-171356.png)

- 高斯分布是一种很重要的分布：

  - 参数容易解释，也描述了分布最基本的性质
  - 中心极限定理（样本均值的极限分布为高斯分布）
  - 对模型残差或噪声能很好建模（高斯分布的由来）
  - 具有相同方差的所有可能的概率分布中，正态分布具有最大的不确定性（极大熵）

- 标准正态分布

  - 当 $\mu=0,\sigma=1$ 时，称为**标准正态分布**，通常用 $Z$ 表示服从标准正态分布的变量，记为 $Z\sim\mathcal{N}(0,1)$
  - pdf和CDF分别记为 $\phi(z),\Phi(z)$
  - 标准化：
    - 若 $X\sim\mathcal{N}(\mu,\sigma^2)$，则 $Z=(X-\mu)/\sigma\sim\mathcal{N}(0,1)$
    - 若 $Z\sim\mathcal{N}(0,1)$，则 $X=\mu+\sigma Z\sim\mathcal{N}(\mu,\sigma^2)$

- 退化的高斯分布

  - 当 $\sigma^2\rightarrow0$ 时，高斯分布退化为无限高无限窄、中心位于 $\mu$ 的“针”状分布：$\lim_{\sigma^2\rightarrow0}\mathcal{N}(x|\mu,\sigma^2)=\delta(x-\mu)$

    其中 Dirac delta 函数:
    $$
    \begin{aligned}
    \delta(x) =
    & \begin{cases}
    \infty& x=0\\
    0&
    x\neq 0
    \end{cases} \\
    \end{aligned} \,\,\,\,\text{使得} \int^\infty_{-\infty}\delta(x)dx=1
    $$

  - 有用的性质：将某个信号从求和／积分中筛选出来
    $$
    \int^\infty_{-\infty}f(x)\delta(x-u)dx=f(u)
    $$

  - 注意：别讲 Dirac delta 函数与 kronecker delta 函数混淆

- 经验分布

  - Dirac 分布经常作为经验分布（empirical distribution） 的一个组成部分出现：
    $$
    \hat{p}(x)=\frac{1}{N}\sum^N_{i=1}\delta(x-x_i)
    $$

    - 将密度 1/N 赋给每一个数据点
    - 只有当定义连续型随机变量的额经验分布时，Dirac delta 函数才是必要的；对离散型随机变量，经验分布可以被定义成一个 Multinomial 分布，对每一个可能的输入，其概率可简单地设为在训练集上那个输入值的经验频率。
    - 经验分布也是极大似然估计（使训练数据的出现概率最大的那个概率密度函数）

##### Laplace分布

- Laplace分布是一个有长尾的分布，pdf为
  $$
  \text{Lap}(x|\mu,b)=\frac{1}{2b}\exp(-\frac{|x-\mu|}{b})
  $$
























![WX20170726-174427](./images/WX20170726-174427.png)

- 相比高斯分布，Lapace分布在0附近更集中 -> **稀疏性**

- Laplace 分布的均值：$\mu$

- 方差：$2b^2$

- 高斯分布对噪声敏感（$\log(p(x))$为到中心距离的二次函数$\frac{1}{2\sigma^2}(x-\mu)^2$），而Laplace分布更鲁棒的分布

  ![WX20170726-174832](./images/WX20170726-174832.png)

##### Gamma分布

- 对任意正实数随机变量 $x>0$，Gamma分布为 $x\sim Ga(shape=a,rate=b)$
  $$
  p(x|a,b)=\frac{b^\alpha}{\Gamma(a)}x^{\alpha-1}x^{-xb}
  $$
  其中 $\Gamma(a)$ 为Gamma函数，$a$为**形状**参数，$b$为**比率**度参数。

  ![WX20170726-175641](./images/WX20170726-175641.png)

- Gamma分布的另一种表示：用**尺度**参数代替比率参数
  $$
  Ga(x|shape=\alpha,scale=\beta)=\frac{1}{\beta T(\alpha)}x^{\alpha-1}e^{-x/\beta}\\
  =Ga\Big(x|shape=\alpha,rate=\frac{1}{\beta}\Big)
  $$

- **反Gamma分布**：若 $X\sim Ga(a,b)$，则 $\frac{1}{X}\sim IG(a,b)$
  $$
  IG(x|shape=\alpha,scale=\beta)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{-\alpha-1}e^{-x/\beta}
  $$
  <u>反Gamma分布用于正态分布方差的共轭先验。</u>

- Gamma分布的均值：$a/b$，众数：$(a-1)/b$，方差：$a/b^2$

- 反Gamma分布的均值：$b/(a-1)$，众数：$b/(a+1)$，方差：$b^2/(a-1)^2(a-2)$

##### Beta分布

- Beta 分布的支持区间为 $[0,1]$：$Beta(\theta|a,b)=\frac{1}{B(a,b)}\theta^{\alpha-1}(1-\theta)^{b-1}$

  其中 Beta 函数 $B(a,b):=\frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$

- Beta 分布的均值、众数和方差：$\mathbb{E}(\theta)=\frac{a}{a+b},mode[\theta]=\frac{a-1}{a+b-2},\mathbb{V}=\frac{ab}{(a+b)^2(a+b+1)}$

- 当 $0<a<1,0<b<1$ 时，在0和1处有两个峰值。

- 当 $a>1,b>1$ 时，有单个峰值。

- 当 $a=b=1$ 时，为均匀分布。

- <u>Beta 分布可作为二项分布的参数的共轭先验分布。</u>

![WX20170726-181154](./images/WX20170726-181154.png)

##### Dirichlet 分布

- 将 Beta 分布扩展到多维，即得到 Dirichlet 分布。其pdf为
  $$
  \text{Dir}(\boldsymbol{\theta}|\boldsymbol{\alpha})=\frac{1}{B(\boldsymbol{\alpha})}\prod^K_{k=1}\theta^{\alpha_k-1}_k
  $$
  其中$B$函数为
  $$
  B(\boldsymbol{\alpha}):=\frac{\prod^K_{k=1}\Gamma(\alpha_k)}{\Gamma(\alpha_0)},\alpha_0=\sum^K_{k=1}\alpha_k
  $$
  <u>Dirichlet 分布在文档分析中的主题模型 LDA (Latent Dirichlet Allocation) 用到</u>。

- Dirichlet 分布的均值、众数和方差：$\mathbb{E}(\theta_k)=\frac{\alpha_k}{\alpha_0},mode[\theta_k]=\frac{\alpha_k-1}{\alpha_0-K},\mathbb{V}=\frac{\alpha_k(\alpha_0-\alpha_k)}{(\alpha_0)^2(\alpha_0+1)}$

- 参数 $\alpha_0$ 控制分布的强度（分布有多尖），$\alpha_k$ 控制峰值出现的地方。

![WX20170726-182625](./images/WX20170726-182625.png)

![WX20170726-182635](./images/WX20170726-182635.png)



#### 分布的混合

- 通过组合一些简单的概率分布来定义新的概率分布也很常见。

- 一种通用的组合方法是构造混合分布（mixture distribution）：

  混合分布由一些组建（component）分布构由哪个组建分布产生的取决于从一个 Multinoulli 分布中采样的结果成。每次实验，样本是：
  $$
  p(x)=\sum_k p(c=k)p(x|c=k)
  $$
  其中，$p(x)$ 是对各组件的一个 Multinomial 分布

  例：经验分布就是以 Dirac 分布为组建的混合分布。

- 混合高斯模型

  - 一个非常强大且常见的混合模型是**高斯混合模型**（Gaussian Mixture Model，GMM）

    - 组建 $p(x|c=k)$ 是高斯分布
    - 每个组件用自己的参数：均值、方差-协方差矩阵
    - 组件也可以共享参数：每个组件的方差-协方差矩阵相等。

  - GMM是概率密度的万能近似器（universal approximator）：

    任何平滑的概率密度都可以用具有足够多组件的高斯混合模型以任意精度逼近。

- 一些有意思的分布可以表示为一组无限个高斯的加权和，其中每个高斯的**方差不同**
  $$
  p(x)=\int \mathcal{N}(x|\mu,\tau^2)\pi(r^2)d\tau^2
  $$

  - 如Student分布：
    $$
    \mathcal{T}(x|\mu,\sigma^2,\nu)=\int^\infty_0\mathcal{N}(x|,\sigma^2/\lambda)\text{Ga}\Big(\lambda|\frac{\nu}{2},\frac{\nu}{2}\Big)d\lambda
    $$

    - Student 分布和高斯分布很像，但尾巴更长

    - 当自由度 $\nu\rightarrow\infty$ 时，极限分布为高斯分布
      $$
      \mathcal{T}(x|\mu,\sigma^2,\infty)=\lim_{\nu\rightarrow\infty}\int^\infty_0\mathcal{N}(x|,\sigma^2/\lambda)\text{Ga}\Big(\lambda|\frac{\nu}{2},\frac{\nu}{2}\Big)d\lambda=\mathcal{N}(x|\mu,\sigma^2)
      $$






















#### 各分布之间的关系

![094313lvdvmjlodwuu6vwv](./images/094313lvdvmjlodwuu6vwv.jpg)

​			实线：某种关系；虚线：近似



### 四、抽样分布

#### IID（Independent Identically Distribution）样本

- 当 $X_1,\dots,X_N$ 互相独立且有相同的边缘分布 $F$ 时，记为 $X_1,\dots,X_N\sim F$

  我们称 $X_1,\dots,X_N$ 为独立同分布（Independent Identically Distribution，IID）样本，表示 $X_1,\dots,X_N$ 是从相同分布独立抽样／采样，我们也称 $X_1,\dots,X_N$ 是分布 $F$ 的随机样本。若 $F$ 有密度 $p$，也可记为 $X_1,\dots,X_N\sim p$

#### 抽样分布

- 令 $X_1,\dots,X_N$ 为独立同分布样本（IID），其均值和方差分别为 $\mu$ 和 $\sigma^2$，则样本均值 $\bar{X}_N=\frac{1}{N}\sum^N_{i=1}X_i$ 为一统计量，也是随机变量，因此也可对其进行分布描述，该分布称为统计量的抽样分布。
  - 请不要将 $ X_i$ 的分布与 $\bar{X}_N$ 的分布混淆：如 $X_i$ 的分布是均匀分布，当 $N$ 足够大时，$\bar{X}_N$ 的分布为正态分布（中心极限定理）

#### 样本均值和样本方差

- 最简单的数据分析问题：如何知道产生数据的分布的期望和方差

- 令 $X_1,\dots,X_N$ 为IID，**样本均值**定义为：$\bar{X}_N=\frac{1}{N}\sum^N_{i=1}X_i$

  **样本方差**定义为：$S^2_N=\frac{1}{N-1}\sum^N_{i=1}(X_i-\bar{X}_N)^2=\frac{1}{N-1}(\sum^N_{i=1}X^2_i-N\bar{X}^2)$

- 问题：样本均值和样本方差会是分布 $F$ 真正期望和方差的很好估计？

- 假设 $X_1,\dots,X_N$ 为IID，$\mu=\mathbb{E}(X_i),\sigma^2=\mathbb{V}(X_i)$，那么
  $$
  \mathbb{E}(\bar{X}_N)=\mu,\mathbb{V}(\bar{X}_N)=\frac{\sigma^2}{N},\mathbb{E}(S^2_N)=\sigma^2
  $$
  即 $\bar{X}_N$ 和 $S^2_N$ 分别为 $\mu$ 和 $\sigma^2$ 的很好估计（无偏估计）。样本数 $N$ 越大，$\mathbb{V}(\bar{X}_N)$ 越小，$\bar{X}_N$ 越接近 $\mu$

![WX20170727-103026](./images/WX20170727-103026.png)

#### 两种收敛的定义

- 令 $X_1,\dots,X_N$ 为随机变量序列，$X$ 为另一随机变量，用 $F_N$ 表示 $X_N$ 的CDF，用 $F$ 表示 $X$ 的CDF

  1. 如果对每个 $\epsilon>0$，当 $N\rightarrow\infty$ 时，
     $$
     \mathbb{P}(|X_N-X|>\epsilon)\rightarrow0
     $$
     则 $X_N$ **依概率收敛**于 $X$，记为 $X_N \xrightarrow{P} X$

  2. 如果对所有 $F$ 的连续点 $t$，有
     $$
     \lim_{N\rightarrow\infty}F_N(t)=F(t)
     $$
     则 $X_N$ **依分布收敛**于 $X$，记为 $X_n\sim X$

#### 弱大数定律（WLLN）

- 独立同分布（IID）的随机变量序列 $X_1,\dots,X_N$，期望 $\mathbb{E}(X_i)=\mu$，方差 $\mathbb{V}(X_i)=\sigma^2<\infty$，则随机均值 $\bar{X}_N=\frac{1}{N}\sum^N_{i=1}X_i$ **依概率收敛**于期望 $\mu$，即对任意 $\epsilon>0$
  $$
  \lim_{N\rightarrow\infty}\mathbb{P}(|\bar{X}_N-\mu|>\epsilon)=0
  $$
  称 $\bar{X}_N$ 为 $\mu$ 的一致估计（一致性）

- 在定理条件下，当样本数目 $N$ 无限增加时，随机样本均值将几乎变成一个常量。

- 样本方差也依概率收敛于方差 $\sigma^2$


#### 中心极限定理（Central Limit Theorem，CLT）

- 独立同分布（IID）的随机变量序列 $X_1,\dots,X_N$，期望 $\mathbb{E}(X_i)=\mu$，方差 $\mathbb{V}(X_i)=\sigma^2<\infty$，则样本均值 $\bar{X}_N=\frac{1}{N}\sum^N_{i=1}X_i$ 近似服从期望为 $\mu$，方差为 $\sigma^2/N$ 的正态分布，即
  $$
  Z_N\equiv\frac{\sqrt{N}(\bar{X}_N)-\mu}{\sigma}\approx Z
  $$
  其中 $Z$ 为标准正态分布，也记为 $\bar{X}_N\approx\mathcal{N}(\mu,\sigma^2/N)$

- 无论随机变量 $X$ 为何种类型的分布，只要满足定理条件，其样本均值就近似服从正态分布。正态分布很重要

  - 但近似的程度与原分布有关
  - 大样本统计推理的理论基础


![0b7b02087bf40ad1ae18a4c7552c11dfa9ecce12](./images/0b7b02087bf40ad1ae18a4c7552c11dfa9ecce12.jpg)



- 标准差 $\sigma^2$ 通常不知道，可用样本标准差代替，中心极限定理仍成立，即
  $$
  \frac{\sqrt{N}(\bar{X}_N-\mu)}{S_N}\approx Z
  $$
  其中
  $$
  S^2_N=\frac{1}{N-1}\sum^N_{i=1}(X_i-\bar{X}_N)^2
  $$



















### 五、分布的估计

#### 分布估计

- 已知分布的类型，但参数未知：参数估计
  - 第3/4节课内容
- 分布类型未知：非参数估计
  - 直方图、核密度估计
  - 根据有限个统计量估计分布：极大熵原理

#### 非参数概率模型

##### 直方图估计

- 一种非参数的概率估计方式是**直方图**估计

  - 将输入空间划分为 $M$ 个箱子（bin），箱子的宽度为 $h=1/M$，则这些箱子为：

  - $$
    B_1=[0,\frac{1}{M}),B_2=[\frac{1}{M},\frac{2}{M}),\dots,B_M=[\frac{M-1}{M},1)
    $$

  - 计算落入箱子 $b$ 中的样本的数目 $\nu_b$,落入箱子 $b$ 的比率为 $\hat{p}_b=\frac{\nu_b}{N}$

    则直方图估计为
    $$
    \hat{p}(x)=\sum^M_{b=1}\frac{\hat{p}_b}{h}\mathbb{I}(x\in B_b)=\frac{1}{N}\sum^M_{b=1}\frac{1}{h}\mathbb{I}(x\in B_b)
    $$
    ![WX20170727-112849](./images/WX20170727-112849.png)

  - 当箱子数目 $M$ 为固定值时，该估计为参数模型

    - 参数个数固定／有限

  - 通常 $M$ 与样本数 $N$ 有某种关系

    - 非参数模型，如何选择M？
      - 交叉验证

##### 核密度估计

- 直方图不连续
  - 箱中每个样本的权重相等

- **核密度估计**：更平滑，比直方图收敛更快

  - 基本思想：每个样本的权重随其到目标点的距离平滑衰减

- 核密度估计定义为
  $$
  \hat{p}(x)=\frac{1}{N}\sum^N_{i=1}\frac{1}{h}K\Big(\frac{x-x_i}{h}\Big)
  $$
  其中参数 $h$ 称为**带宽**（bandwidth），核函数可为人意平滑的函数 $K$，满足
  $$
  K(u)\ge0,\int K(u)du=1,\int uK(u)du=0,\sigma^2_K=\int u^2K(u)du>0
  $$

- 实质：

  - 对样本点施以不同的权，用加权来代替通畅的计数

- 例子：

  - Epanechnikov 核：$K(u)=\frac{3}{4}(1-u^2)\mathbb{I}(|u|\le1)$

    - 使风险最小的核函数
    - 亦被称为抛物面核或者叫做二次核函数

    ![WX20170727-112928](./images/WX20170727-112928.png)

  - 高斯核：$K(u)=\frac{1}{\sqrt{2\pi}}e^{u^2/2}$

- 核密度估计-带宽

  - 为了构造一个核密度估计，我们需要选择核函数 $K$ 和 带宽 $h$
  - 带宽的选择比核函数的选择更重要
  - 关键是**平滑参数**—— $h$
    - 小的平滑参数 $h$：估计结果受噪音影响较大，当 $h\rightarrow0$ 得到针状分布
    - 大的平滑参数 $h$：估计结果过分平滑，当 $h\rightarrow\infty$ 趋向于均匀分布

#### Python Seaborn 数据集分布的可视化

