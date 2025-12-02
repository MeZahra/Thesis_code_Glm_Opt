import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import os

# Create output directory if it doesn't exist
# output_dir = r'D:\UBC\510\Figures\Subject_individual_violin'
# os.makedirs(output_dir, exist_ok=True)

# Load data files
sham = loadmat('/mnt/TeamShare/Old servers (Juana working on organizing this)/MedData/data/EEG/GVSEEG_preprocessed/neurodata_dataset/task_gvssham.mat')
gvs7 = loadmat('/mnt/TeamShare/Old servers (Juana working on organizing this)/MedData/data/EEG/GVSEEG_preprocessed/neurodata_dataset/task_gvsstim7.mat')
gvs8 = loadmat('/mnt/TeamShare/Old servers (Juana working on organizing this)/MedData/data/EEG/GVSEEG_preprocessed/neurodata_dataset/task_gvsstim8.mat')

# Define metric indices (Python uses 0-based indexing)
metrics = {
    'Grip Strength': 0,
    'Strength Velocity': 4,
    'Movement Time': 6,
    'Squeeze Time': 7,
    'Peak Time': 11,
    'Reaction Time': 12
}

# Define groups and their data
groups = {
    'HC': {
        'sham': sham['hcoffmed'][0],
        'low_gvs': gvs7['hcoffmed'][0],
        'high_gvs': gvs8['hcoffmed_gvs8'][0],
        'n_subjects': 22
    },
    'PD_ON': {
        'sham': sham['pdonmed'][0],
        'low_gvs': gvs7['pdonmed'][0],
        'high_gvs': gvs8['pdonmed_gvs8'][0],
        'n_subjects': 20
    },
    'PD_OFF': {
        'sham': sham['pdoffmed'][0],
        'low_gvs': gvs7['pdoffmed'][0],
        'high_gvs': gvs8['pdoffmed_gvs8'][0],
        'n_subjects': 20
    }
}
print(type(groups))
print(type(groups['HC']))
print(groups['HC'].keys())
print(type(groups['HC'].values()))

def normalize_metric_data(sham_data, gvs7_data, gvs8_data, metric_idx):
    """Normalize metric data across all conditions using z-score"""
    all_data = []
    for data in [sham_data, gvs7_data, gvs8_data]:
        all_data.extend(data[:, metric_idx].flatten())
    
    scaler = StandardScaler()
    all_data_array = np.array(all_data).reshape(-1, 1)
    scaler.fit(all_data_array)
    
    norm_sham = scaler.transform(sham_data[:, metric_idx].reshape(-1, 1)).flatten()
    norm_gvs7 = scaler.transform(gvs7_data[:, metric_idx].reshape(-1, 1)).flatten()
    norm_gvs8 = scaler.transform(gvs8_data[:, metric_idx].reshape(-1, 1)).flatten()
    
    return norm_sham, norm_gvs7, norm_gvs8

def create_violin_plot(subject_data, subject_id, group_name):
    """Create violin plots for one subject showing all 6 metrics across 3 conditions"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{group_name} - Subject {subject_id + 1}', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (metric_name, metric_idx) in enumerate(metrics.items()):
        ax = axes[idx]
        
        # Extract and normalize data for this metric
        sham_data = subject_data['sham'][:, metric_idx]
        gvs7_data = subject_data['low_gvs'][:, metric_idx]
        gvs8_data = subject_data['high_gvs'][:, metric_idx]
        
        norm_sham, norm_gvs7, norm_gvs8 = normalize_metric_data(
            sham_data.reshape(-1, 1), 
            gvs7_data.reshape(-1, 1), 
            gvs8_data.reshape(-1, 1), 
            0
        )
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'Value': np.concatenate([norm_sham, norm_gvs7, norm_gvs8]),
            'Condition': ['Sham']*10 + ['Low GVS']*10 + ['High GVS']*10,
            'Trial': list(range(1, 11)) * 3
        })
        
        # Create violin plot
        sns.violinplot(data=df, x='Condition', y='Value', ax=ax, 
                      palette=['#E8E8E8', '#A8D5E2', '#548CA8'],
                      inner=None, cut=0)
        
        # Overlay individual trial points
        sns.stripplot(data=df, x='Condition', y='Value', ax=ax,
                     color='black', alpha=0.6, size=5, jitter=0.2)
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('GVS Condition', fontsize=10)
        ax.set_ylabel('Normalized Value (z-score)', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'{group_name}_Subject_{subject_id + 1:02d}.png'
    # filepath = os.path.join(output_dir, filename)
    # plt.savefig(filepath, dpi=300, bbox_inches='tight')
    # plt.close()
    
    return 

# Generate plots for all subjects
print("Generating violin plots for all subjects...")
# print(f"Saving to: {output_dir}\n")

for group_name in groups.keys():
    print(f"Processing {group_name} group...")
    n_subjects = groups[group_name]['n_subjects']
    
    for subj_idx in range(n_subjects):
        subject_data = {
            'sham': groups[group_name]['sham'][subj_idx],
            'low_gvs': groups[group_name]['low_gvs'][subj_idx],
            'high_gvs': groups[group_name]['high_gvs'][subj_idx]
        }
        
        create_violin_plot(subject_data, subj_idx, group_name)
        # print(f"  Created: {os.path.basename(filepath)}")
    
    print(f"Completed {group_name}: {n_subjects} subjects\n")

print("All plots generated successfully!")
print(f"Total plots created: {22 + 20 + 20} = 62 plots")
# print(f"Location: {output_dir}")