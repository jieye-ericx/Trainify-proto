import time

import re
import gym
import numpy as np
from core.abstract import initiate_divide_tool_rtree, initiate_divide_tool
from core.agent import DDPGAgent
from core.data import Recorder


class Trainify:
    def __init__(self,
                 env_config={},
                 env_class=None,
                 agent_config={},
                 agent_class={},
                 verify_config={},
                 verify=False,
                 log_dir='',
                 experiment_name="default_name"):

        self.env_config = env_config
        self.env_class = env_class
        self.agent_config = agent_config
        self.agent_class = agent_class
        self.verify_config = verify_config
        self.verify = verify
        self.experiment_name = experiment_name
        # self.experiment_name = experiment_name + time.strftime("_%Y%m%d_%H_%M_%S", time.localtime())
        if log_dir == '':
            log_dir = self.experiment_name
        self.recorder = Recorder(experiment_name=self.experiment_name, data_dir_name=log_dir)

        self._np_state_space = np.array(self.env_config['state_space'])
        # Dict(x1:Box([-1.5], [1.5], (1,), float32), x2:Box([-1.5], [1.5], (1,), float32), x3:Box([-10.], [10.], (1,), float32))
        self._gym_dict_state_space = gym.spaces.Dict(self._handle_dict_env_state())
        # Box([ -1.5  -1.5 -10. ], [ 1.5  1.5 10. ], (3,), float32)
        self._gym_box_state_space = gym.spaces.Box(low=np.float32(self._np_state_space[0]),
                                                   high=np.float32(self._np_state_space[1]), dtype=np.float32)

        self.env = self.env_class()
        self.agent = self.agent_class(self.env, config=self.agent_config)

        if self.verify:
            self.divide_tool = initiate_divide_tool_rtree(self.env_config['state_space'],
                                                          self.env_config['abs_initial_intervals'],
                                                          self.env_config['state_key_dim'],
                                                          'rtree_' + experiment_name)
        else:
            self.divide_tool = initiate_divide_tool(self.env_config['state_space'],
                                                    self.env_config['abs_initial_intervals'])

        # print(self.agent.__dict__['actor'])
        self.save_model = self.recorder.create_save_model(self.agent)
        self.load_model = self.recorder.create_load_model(self.agent)
        self.recorder.save_model = self.save_model
        self.recorder.load_model = self.load_model
        self._create_cal_state_func(self.env)

    def _create_cal_state_func(self, Env):
        labels = [
            ['sin', 'math.sin'],
            ['cos', 'math.cos'],
            ['tan', 'math.tan'],
            ['tanh', 'math.tanh']
        ]

        def from_function(str, max):
            def cal(self, x):
                if max:
                    return eval(str)
                else:
                    return -eval(str)

            return cal

        for i, str in enumerate(self.env_config['dynamics']):
            for a in labels:
                reg = re.compile(re.escape(a[0]), re.IGNORECASE)
                self.env_config['dynamics'][i] = reg.sub(a[1], self.env_config['dynamics'][i])

            setattr(Env, self.env_config['states_name'][i] + '_maximum',
                    from_function(self.env_config['dynamics'][i], True))
            setattr(Env, self.env_config['states_name'][i] + '_minimum',
                    from_function(self.env_config['dynamics'][i], False))

        print('Trainify 在Env中加入状态最值计算函数成功')

    def _handle_dict_env_state(self):
        abs_obs = {}
        for i, name in enumerate(self.env_config['states_name']):
            abs_obs.update({
                name: gym.spaces.Box(
                    low=np.float32(self.env_config['state_space'][0][i]),
                    high=np.float32(self.env_config['state_space'][1][i]),
                    shape=(1,),
                    dtype=np.float32
                )
            })
        return abs_obs

    def _gen_dict(self, num):
        return {
            'min': np.full(shape=num, fill_value=float("inf"), dtype=float).tolist(),
            'max': np.full(shape=num, fill_value=-float("inf"), dtype=float).tolist()
        }


if __name__ == '__main__':
    print('main')
    env_c = {
        "dim": 2,
        "states_name": ["x1", "x2", "x3"],
        "state_space": [[-1.5, -1.5, -10], [1.5, 1.5, 10]],
        "abs_initial_intervals": [0.16, 0.16, 0.01],
        "state_key_dim": [0, 1],
        "dynamics": "x1^2+x2"
    }
    a = Trainify(env_config=env_c)
