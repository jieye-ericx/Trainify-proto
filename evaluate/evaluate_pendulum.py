import matplotlib.pyplot as plt

def evaluate_pendulum(agent,env):
    reward_list2 = []
    min_reward = 0
    g_cos_l = 1

    for l in range(10):
        reward = 0
        s0 = env.reset()
        cos_l = 1
        for step in range(10000):
            # envs.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            if s1[0] < cos_l:
                cos_l = s1[0]
            if s1[0] <= 0:
                print('fall down-------', s1)
            reward += r1
            s0 = s1
        if cos_l < g_cos_l:
            g_cos_l = cos_l
        print('evaluate: ', l, ':', reward, cos_l, g_cos_l)

        if reward < min_reward:
            min_reward = reward
        reward_list2.append(reward)
    plt.plot(reward_list2)
    plt.show()
    return min_reward
