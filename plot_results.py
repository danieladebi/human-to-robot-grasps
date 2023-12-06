import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_mean_and_variance(file_path):
    df = pd.read_csv(file_path)
    df['reward'] = df['reward'].apply(lambda x: float(x[1:-1]))
    mean = df['reward'].mean()
    variance = df['reward'].var()
    data = df['reward'].values
    return data, mean, variance

def plot_boxplots(folders):
    # make # subplots = len(folders)
    fig, axs = plt.subplots(len(folders))
    all_data = []
    all_args = []
    for folder in folders:
        file_path = os.path.join(folder, 'evaluation', 'reward_eval.csv')
        # let's also get hyperparameters
        args_path = os.path.join(folder, 'args.csv')
        data, mean, variance = compute_mean_and_variance(file_path)
        all_data.append(data)
        # parse args and add to all_args
        with open(args_path, 'r') as f:
            args = {}
            for line in f.readlines():
                arg, val = line.split(',')
                val = val.strip()
                # try to parse bool
                if val == 'True':
                    val = True
                elif val == 'False':
                    val = False
                else:
                    try:
                        val = float(val)
                    except:
                        val = val
                args[arg] = val
            all_args.append(args)
        
        
    # make numpy array of all data with shape (len(data), len(folders))
    all_data = np.array(all_data)
    all_data = all_data.reshape(-1, len(folders))
    df = pd.DataFrame(all_data)
    fig, ax = plt.subplots()
    #Note showfliers=False is more readable, but requires a recent version iirc
    box = df.boxplot(ax=ax, labels=folder_names, sym='') 
    ax.margins(y=0)
    ax.set_ylabel('Reward')
    
    # ax.x_label = 'Experiment'
    
    def get_label(args):
        # combine the hyperparameters into a single label
        label = ''
        if args['model'] == 'ppo':
            label += 'PPO\n'
        elif args['model'] == 'a2c':
            label += 'A2C\n'
        if args['use_vip_embedding_obs']:
            label += 'VIP Obs\n'
        elif 'use_r3m_embedding_obs' in args and args['use_r3m_embedding_obs']:
            label += 'R3M Obs\n'
        else:
            label += 'Img Obs\n'
        if args['use_hand_pose_obs']:
            label += 'Hand Obs\n'
        if args['use_vip_reward']:
            label += 'VIP Reward'
        if 'use_r3m_embedding_obs' in args and args['use_r3m_reward']:
            label += 'R3M Reward'
        # if args['vip_reward_type'] == 'replace':
        #     label += 'Replace'
        # elif args['vip_reward_type'] == 'add':
        #     label += 'Add'
        return label
            
    ax.set_xticklabels([get_label(args) for args in all_args])
    fig.savefig('boxplot.png', bbox_inches='tight')
        
        
    # max_val = np.max([np.max(data) for data in all_data])
    # min_val = np.min([np.min(data) for data in all_data])
    # for i in range(len(all_data)):
    #     axs[i].boxplot(data, labels=[folder], sym='') 
    #     axs[i].set_xlabel('Folders')
    #     axs[i].set_ylabel('Reward')
    #     axs[i].set_ylim([min_val, max_val])
    # # save the figure
    # fig.savefig('boxplot.png', bbox_inches='tight')

base_folder = '/scratch/cluster/mmunje/projects/vip/human-to-robot-grasps/trained_models/lift/'
folder_names = ['62', '63', '64', '65', '66']#, '67']
folders = [os.path.join(base_folder, folder_name) for folder_name in folder_names]
plot_boxplots(folders)
