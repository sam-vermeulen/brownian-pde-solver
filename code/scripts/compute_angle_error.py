import torch
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def angle_from_topright(row):
    return (np.arctan2(row.old_point[1] - 1, row.old_point[0] - 1) + np.pi) % (2 * np.pi)

def error_by_angle(df, bin_width=0.035, which='new'):
    df['angle'] = df.apply(angle_from_topright, axis=1)
    
    df['x'] = df['old_point'].apply(lambda p: p[0])
    df['y'] = df['old_point'].apply(lambda p: p[1])
    
    tr_df = df[(df.x > 0.8) & (df.y > 0.8)]
    bin_edges = np.arange(0, np.pi/2, bin_width)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    binned_data = pd.cut(tr_df['angle'], bins=bin_edges, labels=bin_centres)
    tr_df['diff_old'] = (tr_df.prob_x_first - tr_df.prob_x_first_old).abs() 
    tr_df['diff_new'] = (tr_df.prob_x_first - tr_df.prob_x_first_new).abs()
    angle_error = tr_df.groupby(binned_data)[f'diff_{which}'].agg(['mean', 'std', 'count'])
    
    return angle_error, bin_centres

def error_by_angle_plot(angle_error_old, angle_error_new, bin_centers, bin_width=0.035):
    fig, ax = plt.subplots(figsize=(12,6))
    
    ax.plot(bin_centers, angle_error_old['mean'], alpha=0.7, label='old')
    ax.plot(bin_centers, angle_error_new['mean'], alpha=0.7, label='new')

    if 'std' in angle_error_old:
        ax.fill_between(
            bin_centers, 
            angle_error_old['mean'] - angle_error_old['std'] / (angle_error_old['count'])**(1/2), 
            angle_error_old['mean'] + angle_error_old['std'] / (angle_error_old['count'])**(1/2), 
            alpha=0.2
        )

    if 'std' in angle_error_new:
        ax.fill_between(
            bin_centers, 
            angle_error_new['mean'] - angle_error_new['std'] / (angle_error_new['count'])**(1/2), 
            angle_error_new['mean'] + angle_error_new['std'] / (angle_error_new['count'])**(1/2), 
            alpha=0.2
        )

    ax.set_xlabel('angle from top right corner (radians)')
    ax.set_ylabel('average error')
    ax.set_title('average error by angle bin (+-{:.3f} radians)'.format(bin_width/2)) 
    pi_ticks = np.array([0, np.pi/4, np.pi/2])
    pi_labels = ['0', 'π/4', 'π/2']
    ax.set_xticks(pi_ticks)
    ax.set_xticklabels(pi_labels)
     
    ax.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig("test.png")

if __name__ == "__main__":
    seeds = range(0, 20) 
    step_sizes = 0.006

    device = torch.device('cpu')
    
    errs_new = []
    errs_old = []
    for s in seeds:
        path = f'./data/exit_probabilities-{step_sizes}-{s}.parquet'
        df = pd.read_parquet(path)
        df = df[(df.prob_x_first <= 1) & (df.prob_x_first >= 0)]


        err_new, bins = error_by_angle(df, which="new")
        err_old, _= error_by_angle(df, which="old")

        errs_new.append(err_new['mean'])
        errs_old.append(err_old['mean'])

    errs_new = np.stack(errs_new)
    errs_old = np.stack(errs_old)


    errs_new_stats = pd.DataFrame({
        'mean': errs_new.mean(axis=0),
        'std': errs_new.std(axis=0),
        'count': len(errs_new[0])
    })

    errs_old_stats = pd.DataFrame({
        'mean': errs_old.mean(axis=0),
        'std': errs_old.std(axis=0),
        'count': len(errs_old[0])
    })

    error_by_angle_plot(errs_old_stats, errs_new_stats, bins)
