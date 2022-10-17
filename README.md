# Trainify-proto

### 目录介绍

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
