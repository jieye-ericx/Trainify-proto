# Trainify-proto

本工具是CAV2022中[Trainify: A CEGAR-Driven Training and Verification Framework for Safe Deep Reinforcement Learning](https://faculty.ecnu.edu.cn/_upload/article/files/39/62/197880be44aba90d9d44ac6de8bb/b7ef9fd1-51e0-4284-8af0-5d7a2f9f1869.pdf)的原型工具，Trainify-proto在论文代码的基础上进行了重构改进，增强了易用性、可拓展性和通用性，实现了自定义抽象训练流程和自选形式化验证等功能。

## 安装

1. pypi （完整功能）

   `pip install Trainify-proto`

   > 如果想体验最新功能，也可安装测试版本
   >
   > pip install -i https://test.pypi.org/simple/ Trainify-proto

2. git clone（完整功能）

   通过`git clone`项目到本地，集成进自己的项目中使用

3. web在线使用（部分功能）

## 核心模块

`abstract` 存放抽象相关工具

`agents` 存放自带agent/policy

`core` 核心组件

`data` 训练数据保存

`env` 存放自带env

`evaluate` 评估模块

`test` 测试代码

`utils` 工具函数

### 开发内容

1. 环境与状态抽象
   需要输入的环境信息：

   ```
   Dim:2
   State_var: x1, x2
   Range: [0,1] [0,4]
   dqn or ddpg
   Env class obj
   dynamics:  x1' = x1 + t *x2^2
                       x2' = x2+ t* x1
   ```

2. 训练Policy与Agent
3. 验证规则
    1. AG(safe) AF(taget)
    2. 返回['safe']的函数
4. 统一输出
    1. 命令行中的统一格式规范的输出
    2. tensorboard 作为图表形式

## 发布

切换到`publish`分支

`python setup.py sdist bdist_wheel`

`python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*`

