import numpy as np
import matplotlib.pyplot as plt
import sys
import ast
from collections import Counter

points = {'1': [[1, 0, 0], [1, 0, 1], [1, 1, 0]],
          '2': [[0, 0, 1], [0, 1, 0], [0, 1, 1]]}

new_point = [0, 1, 0.5]

def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q))**2))
    
class KNN:
    def __init__(self, k=3) -> None:
        self.k=k
        self.points = None
        
    def fit(self, points):
        self.points = points
        
    def predict(self, new_point):
        distances = []
        for category in self.points:
            for point in self.points[category]:
                distance = euclidean_distance(point, new_point)
                distances.append([distance, category])
        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result

# help page
if len(sys.argv) > 1 and ("--help" in sys.argv or "-h" in sys.argv):
    print("Usage:\n\tpython knn.py\n\tuse -sl or --show-lines to show the distances")
    sys.exit(0)

# actual usage of knn
clf = KNN()
clf.fit(points)
print("the point is predicted to be: " + clf.predict(new_point))

# Plotting
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

for point in points["1"]:
    ax.scatter(point[0], point[1], point[2], color="#104DCA", s=60)

for point in points["2"]:
    ax.scatter(point[0], point[1], point[2], color="#FF0000", s=60)
    
new_class = clf.predict(new_point)
color = "#FF0000" if new_class == "2" else "#104DCA"

# drawing lines from every point to the classified one
if len(sys.argv) > 1 and ("--show-lines" in sys.argv or "-sl" in sys.argv):
    for point in points["1"]:
        ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color="#ADD8E6")
    for point in points["2"]:
        ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color="#ffb3c1")

# scatter plot of predicted point
ax.scatter3D(new_point[0], new_point[1], new_point[2], marker="*", s=200, zorder=100, color=color)
plt.show()