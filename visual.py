import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')


# Load the 1D data from the text file
data = np.loadtxt('first1x1conv.txt')

# Create a 2D histogram
hist, edges = np.histogram(data, bins=1000)

# Plot the histogram
plt.bar(edges[:-1], hist, width=np.diff(edges), align='edge')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Count')
plt.title('2D Histogram')

# Save the histogram as an image
plt.savefig('histogram.png')


