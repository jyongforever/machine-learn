# -*- coding:utf-8 -*-
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建了一张图
g = tf.Graph()
with g.as_default():
    c = tf.constant(10.0)

# print('g图的地址为:',g)
# print('c在的图:',c.graph)


# 相当于分配这个程序的内容
# print(tf.get_default_graph())

# s = tf.Session()

# 用python定义加法

# 不可以运行
# add1 = 0
# add2 = 1
# add_sum = add1 + add2

# 可以运行
# 有一半的类型是op,重载加法运算符
# add1 = 0
# add_sum = add1 + b
# print(add_sum)

# tensorflow实现一个加法操作
a = tf.constant(3.0)
b = tf.constant(4.0)

# 定义加法操作
sum = tf.add(a, b)

# 实现一个函数的作用,自定义传入内容返回结果
# 相当于定义函数的参数
plt1 = tf.placeholder(tf.float32,[2,3])
plt2 = tf.placeholder(tf.float32,[2,3])
print(plt1,plt2)
# sum1 = tf.add(plt1,plt2)

# 开启会话
# 会话只能运行它所在的图的资源
# 如果安装的是GPU 版本tensorflow,那么久可以去使用设备指定计算某些功能
# tf.device()
with tf.Session() as sess:
    # 运行的参数必须是op或者tensor
    # print(sess.run([a, b, sum]))
    print(sess.run(plt1,feed_dict={plt1:[[1,2,3],[4,5,6]],plt2:[[2,3,4],[3,2,1]]}))
    # print(a.graph)
    # print(sum.graph)
    # print(sess.graph)

# tensor打印出来的形状
# 0维 ()
# 1维 (2,)
# 2维 (2,3)
# 3维 (10,2,3)

# 张量的静态形状,和动态形状
# 形状不固定(?,2)
# 静态形状:对于没有固定的形状,可以使用set_shape,如果张量形状全部固定,不能去修改任意形状
plt = tf.placeholder(tf.float32,[None,2])
# 通过静态形状修改plt的形状,plt本身的形状已经修改了
plt.set_shape([1,2])
print(plt)

# 再次修改本身plt的形状
# plt.set_shape([3,2])

with tf.Session() as sess:
    print(sess.run(plt,feed_dict={plt:[[1,2]]}))



