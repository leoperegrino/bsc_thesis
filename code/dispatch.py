import pandas as pd
import seaborn as sn
import sklearn as sk
import warnings

from datetime import timedelta
from matplotlib import pyplot as plt
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

timestep = timedelta(minutes=10)
start_pred = int(len(df)*0.75)+window_length+1

diesel_timeout = 3
confusion_matrix = []


for start in d[start_pred:-window_length].index:

    window = d.loc[start:start + timestep * 35]

    dispatch = apply_dispatch(
        window,
        d.loc[start]['cc_soc'],
        d.loc[start]['lf_soc'],
        diesel_lag=diesel_timeout
    )
    lolp_cc = lolp(dispatch['cc_net_load'])
    lolp_lf = lolp(dispatch['lf_net_load'])

    if lolp_cc < lolp_lf:
        real = 'cc'
    elif lolp_cc > lolp_lf:
        real = 'lf'

    lolp_cc_predicted, lolp_lf_predicted = best_predicted(
        results,
        start_pred,
        d.loc[start]['cc_soc'],
        d.loc[start]['lf_soc'],
        diesel_timeout=diesel_timeout
    )

    if lolp_cc_predicted < lolp_lf_predicted:
        predicted = 'cc'
    elif lolp_cc_predicted > lolp_lf_predicted:
        predicted = 'lf'

    confusion_matrix.append((real, predicted))


confusion_matrix = pd.DataFrame(confusion_matrix)
confusion_matrix = sk.metrics.confusion_matrix(
    confusion_matrix[0],
    confusion_matrix[1]
)

df_cm = (
    pd.DataFrame(
        confusion_matrix,
        index=['lf', 'cc'],
        columns=['lf', 'cc']
    )
    /
    confusion_matrix.sum()*100
)

ax = sn.heatmap(df_cm, annot=True, fmt='g')
plt.figure(figsize=(10, 7))
ax.set_title('Matrix de Confusao', y=1.08)
ax.set_ylabel('Real')
ax.set_xlabel('Previsto')
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
