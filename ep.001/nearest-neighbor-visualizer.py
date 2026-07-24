"""
Interactive 1-Nearest-Neighbor Visual Explorer
----------------------------------------------
Run this locally to click on the plot and see live nearest-neighbor
classification and distance measurements.

Requirements: pip install matplotlib numpy
"""

import math
import matplotlib.pyplot as plt
import numpy as np

# 1. Defined Training Dataset: ((x, y), label)
training_examples = [
    ((0.1, 0.1), 1),
    ((0.9, 0.9), 0),
    ((0.2, -0.1), 1),
    ((1.2, 0.3), 0),
    ((-0.3, 0.2), 1),
]

def euclidean_distance(p, q):
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

def nearest_neighbor_classify(point, examples):
    # Find closest training example and return (label, distance, closest_point)
    closest_example = min(examples, key=lambda ex: euclidean_distance(point, ex[0]))
    dist = euclidean_distance(point, closest_example[0])
    return closest_example[1], dist, closest_example[0]

# --- Interactive Matplotlib UI ---
fig, ax = plt.subplots(figsize=(8, 8))

def update_plot(test_point=None):
    ax.clear()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    
    # Draw Ground Truth Unit Circle (r = 1.0)
    circle = plt.Circle((0, 0), 1.0, color='blue', fill=False, linestyle=':', label='True Boundary (r=1.0)')
    ax.add_patch(circle)

    # Plot Training Points
    for (x, y), label in training_examples:
        color = 'green' if label == 1 else 'red'
        marker = 'o' if label == 1 else 's'
        label_text = 'Inside (1)' if label == 1 else 'Outside (0)'
        ax.scatter(x, y, color=color, s=120, marker=marker, zorder=3)
        ax.annotate(f"({x},{y})", (x + 0.04, y + 0.04), fontsize=9)

    # Plot Test Point if clicked
    if test_point is not None:
        label, dist, closest_pt = nearest_neighbor_classify(test_point, training_examples)
        
        # Draw line from test point to closest neighbor
        ax.plot([test_point[0], closest_pt[0]], [test_point[1], closest_pt[1]], 
                color='purple', linestyle='--', linewidth=1.5, label=f'Distance: {dist:.3f}')
        
        # Draw test point
        pred_color = 'green' if label == 1 else 'red'
        ax.scatter(test_point[0], test_point[1], color=pred_color, s=200, marker='*', 
                   edgecolors='black', linewidth=1.5, zorder=4, label=f'Predicted: {label}')
        
        ax.set_title(f"Test Point {test_point} -> Nearest Neighbor {closest_pt} (Dist: {dist:.3f}) -> Predicted: {label}", fontsize=11)
    else:
        ax.set_title("Click anywhere on the plot to test a new point!", fontsize=12)

    ax.legend(loc='upper right')
    fig.canvas.draw()

def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        click_point = (round(event.xdata, 2), round(event.ydata, 2))
        update_plot(click_point)

fig.canvas.mpl_connect('button_press_event', on_click)
update_plot()
plt.show()