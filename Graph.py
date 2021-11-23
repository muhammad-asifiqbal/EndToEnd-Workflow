from numpy import float64
from DataManipulation import  Datasets
import seaborn as sns
import matplotlib.pyplot as plt

class Plots(Datasets):
  def graph(self,data):
    # Graph of data quality
    sns.histplot(data.quality, kde=False)
    # Boolean expression,store 0,1 in data.quality
    high_quality = (data.quality >= 7).astype(int) 
    data.quality = high_quality
    # boxplot have 3 rows and 4 columns
    dims = (3, 4)

    f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))

    axis_i, axis_j = 0, 0
    for col in data.columns:
     if col == 'is_red' or col == 'quality':
      continue 
     sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
     axis_j += 1
     if axis_j == dims[1]:
      axis_i += 1
      axis_j = 0

    plt.show()
    data=data.astype(float64)
    return data

grapObj = Plots()
graparg1 = grapObj.dataImport()
graparg1 = grapObj.graph(graparg1)


  
