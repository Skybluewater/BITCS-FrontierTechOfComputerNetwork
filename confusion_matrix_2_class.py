import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score

# create confusion matrix
y_true = np.array([-1]*500 + [1]*500)
y_pred = np.array([-1]*476 + [1]*24 +
                  [-1]*9 + [1]*491)
cm = confusion_matrix(y_true, y_pred)
conf_matrix = pd.DataFrame(cm, index=['Defence', 'Undefence'], columns=['Defence', 'Undefence'])

# plot size setting
fig, ax = plt.subplots(figsize = (4.5,3.5))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues", fmt="d")
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('statistic/tm_confusion.png', bbox_inches='tight', format="png")
plt.show()