.. Math_ML documentation master file, created by
   sphinx-quickstart on Tue Aug  8 11:47:51 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================
机器学习之概率统计基础
===================================

.. Contents:

.. toctree::
   :maxdepth: 2

---------
引言
---------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
究竟什么是机器学习？
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	| A branch of artificial intelligence, concerns the construction and study of systems that can learn from data.  (`REF`_)
	
	| 机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。


.. _REF: https://en.wikipedia.org/wiki/Machine_learning

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
概率 vs. 统计
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* 概率：研究随机事件出现的可能性的数学分支，描述非确定性（Uncertainty）的正式语言，是统计推断的基础
  
	* 概率：一个事件或事件集合出现的可能性
  	* **基本问题：给定以一个数据产生过程，则输出的性质是什么。**

* 统计推断：处理数据分析和概率理论的数学分支，与数据挖掘和机器学习是近亲

 	* 统计量：一个用以描述样本或总体性质的数值，如均值或方差
  	* **基本问题：给定输出数据，我们可以得到该数据的产生过程的哪些信息（相当于是“概率”的一个逆过程）**


.. image:: ./images/WechatIMG6.png
	:align: center



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
概率统计基础的重要性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* 研究数据分析必须大好概率和统计基础

 	* Using fancy tools like neural nets, boosting and support vector machines without understanding basic statistics like **doing brain surgery before knowing how to use a band-aid.**
	* 做脑外科手术，却不知道怎么使用绷带！


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
参考书推荐
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Kevin P. Murphy, **Machine Learning: A Probabilistic Perspective**, MIT Press, 2012
	* 统计部分涵盖Larry了的内容，本笔记所用的符号体系来自此书。
* Larry Wasserman, **All of Statistics: A Concise Course in Statistical Inference**
	* 中译本：《**统计学完全教程**》
	* 内容很全，但有些部分篇幅略少，更偏向于从统计的角度讲述，为计算机系所写
* 盛聚／谢式千／潘承毅，**概率论与数理统计**，高等教育出版社
	* 浙大的这本很不错，国产书算是不错的。




---------
概率
---------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
一、概率公理及推论（基石）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



""""""""""""""
概率
""""""""""""""

- **频率学派**：多次重复试验（多次抛掷硬币），则我们期望正面向上的次数占总实验次数的一半。
- **Bayesian 学派**：我们相信下一次试验中，硬币正面向上的可能性为0.5.
	- 概率是我们对事情不确定性的定量描述，与信息有关，而非重复实验。
	- 可以用来对不能重复实验的时间的不确定，如明天天晴的概率为0.8，如某邮件是垃圾邮件的概率。


""""""""""""""
样本空间和事件
""""""""""""""





===================================
机器学习之矩阵论
===================================


--------------------
Indices and tables
--------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Install $project by running:

.. math:: e^{i\pi} + 1 = 0
   :label: euler

Euler's identity, equation :eq:`euler`, was elected one of the most
beautiful mathematical formulas.

Since Pythagoras, we know that :math:`a^2 + b^2 = c^2`.

.. math::

   (a + b)^2 = a^2 + 2ab + b^2

   (a - b)^2 = a^2 - 2ab + b^2

way2

 .. math::

   (a + b)^2  &=  (a + b)(a + b) \\
              &=  a^2 + 2ab + b^2

way3

 .. math:: (a + b)^2 = a^2 + 2ab + b^2


Look how easy it is to use $a_a$:

    import project 
	.. math:: (a + b)^2 = a^2 + 2ab + b^2   


    # Get your stuff done
    project.do_stuff()

.. math::
   :nowrap:

   \begin{eqnarray}
      y    & = & ax^2 + bx + c \\
      f(x) & = & x^2 + 2xy + y^2
   \end{eqnarray}


----------
Contribute
----------

- Issue Tracker: https://github.com/iphysresearch/Math_ML/issues
- Source Code: https://github.com/iphysresearch/Math_ML

-------
Support
-------

If you are having issues, please let us know.
We have a mailing list located at: hewang@mail.bnu.edu.cn

-------
License
-------

The project is licensed under the MIT license.