import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('validation_LEO_10.csv')

# Extract detailed error data
detail_start = data.index[data.iloc[:,0] == 'sample_id'].tolist()[0] + 1
details = data.iloc[detail_start:].reset_index(drop=True)
details.columns = ['sample_id', 'raw_error', 'corrected_error']

# Create plots
plt.figure(figsize=(12, 8))

# Plot error distributions
plt.subplot(2, 2, 1)
plt.hist(details['raw_error'], alpha=0.5, bins=30, label='Before TMR')
plt.hist(details['corrected_error'], alpha=0.5, bins=30, label='With TMR')
plt.xlabel('Error Magnitude')
plt.ylabel('Frequency')
plt.title('Error Distribution With/Without TMR')
plt.legend()

# Plot error reduction
plt.subplot(2, 2, 2)
plt.scatter(details['raw_error'], details['corrected_error'], alpha=0.5)
plt.xlabel('Error Before Correction')
plt.ylabel('Error After Correction')
plt.title('Error Reduction Effectiveness')
plt.plot([0, details['raw_error'].max()], [0, details['raw_error'].max()], 'r--')

# Plot error over samples
plt.subplot(2, 1, 2)
plt.plot(details['sample_id'], details['raw_error'], 'r-', alpha=0.5, label='Before TMR')
plt.plot(details['sample_id'], details['corrected_error'], 'g-', alpha=0.5, label='With TMR')
plt.xlabel('Sample ID')
plt.ylabel('Error Magnitude')
plt.title('Error Reduction Over Samples')
plt.legend()

# Get summary data
summary = data.iloc[0]
environment = summary['environment']
intensity = summary['radiation_intensity']

# Add summary text
plt.figtext(0.5, 0.01, f'Environment: {environment}, Radiation Intensity: {intensity}\n'
           f'Success Rate: {summary["success_rate"]*100:.2f}%, '
           f'Error Reduction: {(1-summary["mean_error_after"]/summary["mean_error_before"])*100:.2f}%\n'
           f'Theoretical vs Measured Error Rate: {summary["theoretical_error_rate"]:.6f} vs {summary["measured_error_rate"]:.6f}',
           ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Save figure
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('validation_LEO_10_plot.png', dpi=300)
plt.close()
print('Plot saved successfully')
