# Competition_3v3snakes 代码说明

## 环境配置
执行以下命令配置环境
``` shell
conda create -n snake3v3 python=3.7.5

conda activate snake3v3

pip install -r requirements.txt
```


## 模型训练

Run 
``` shell
python rl_trainer/{algo_name}.py
```
 to train the model. You can refer to below `{algo_name}` and its corresbonding algorithms. 
- `main`: DDPG
- `main_maddpg`: MADDPG
- `main_qmix`: QMIX
- `main_mappo`: MAPPO
- `main_rmappo`: recurrent MAPPO
- `main_sil_mappo`: MAPPO with self imitation learning
- `self_play`: Self-play MAPPO
- `pbt`: Population based learning MAPPO

### 部分参数说明
mappo系列代码运行需要较多cpu进程数和较大的gpu memory，如果想减少这一需求，可以提供`--num_workers`参数（默认为24）和`--batch_size`参数（需设置为200的倍数，默认为24*200）。

在运行`pbt.py`时，可以提供`--num_agents`参数（默认为4）设置agent池的大小。

## 模型评估

Run 
```shell
python evaluation_local.py --my_ai {algo_name} --opponent {algo_name}
```
 to evaluate the model. You can refer to below `{algo_name}` and its corresbonding algorithms.

- `random`: random
- `greedy_old`: origin greedy
- `greedy`: greedy(loop)
- `maddpg`: MADDPG
- `qmix`: QMIX
- `mappo`: MAPPO
- `mappo_sil`: MAPPO with self imitation learning
- `sp`: Self-play MAPPO
- `pbt`: Population based learning MAPPO








