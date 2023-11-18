import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay
from datetime import datetime

plt.style.use('seaborn-v0_8-bright')
plt.rcParams['axes.facecolor'] = 'white'
mpl.rcParams.update({"axes.grid" : True, "grid.color": "black"})
mpl.rc('axes',edgecolor='black')
mpl.rcParams.update({'font.size': 12})

def plot_curves_1_dataset(dataname, plotname, data_list):
    """Plot 2 curves: with and without IMU"""
    ax = plt.gca()

    for csv_file, linename, typename, color in data_list:
        df = pd.read_csv(csv_file)
        display = DetCurveDisplay(fpr=df['FAR'].values/100, fnr=df['FRR'].values/100, estimator_name=linename)
        if typename == 'with_imu':
            display.plot(ax, color=color)
        else:
            display.plot(ax, color=color, linestyle='dashed')

    plt.legend(loc='upper right')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dataname}_{plotname}.png"), dpi=400)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    BASE_DIR = '/home/kimngan/storage/code/BehaveFormer/work_dirs'
    results_csv = {'humi': {'SD': {'csv_file': BASE_DIR + '/humi_scroll50down_epoch500_enroll3_b128/20231026_155605/results/20231027_094103/far-frr.csv',
                                   'eer': 25.55},
                            'SD, A, G, M': {'csv_file': BASE_DIR + '/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/results/20231027_094125/far-frr.csv',
                                            'eer': 4.93},
                            'SU': {'csv_file': BASE_DIR + '/humi_scroll50up_epoch500_enroll3_b128/20231026_155901/results/20231027_094146/far-frr.csv',
                                   'eer': 31.48},
                            'SU, A, G, M': {'csv_file': BASE_DIR + '/humi_scroll50up_imu100all_epoch500_enroll3_b128/20231026_155918/results/20231027_094153/far-frr.csv',
                                            'eer': 3.67}},
                   'feta': {'SD': {'csv_file': BASE_DIR + '/feta_scroll50down_epoch500_enroll3_b128/20231018_090238/results/20231027_095112/far-frr.csv',
                                   'eer': 25.88},
                            'SD, A, G': {'csv_file': BASE_DIR + '/feta_scroll50down_imu100all_epoch500_enroll3_b128/20231017_172604/results/20231027_095120/far-frr.csv',
                                         'eer': 22.71},
                            'SU': {'csv_file': BASE_DIR + '/feta_scroll50up_epoch500_enroll3_b128/20231017_173752/results/20231027_095128/far-frr.csv',
                                   'eer': 21.96},
                            'SU, A, G': {'csv_file': BASE_DIR + '/feta_scroll50up_imu100all_epoch500_enroll3_b128/20231017_172352/results/20231027_095135/far-frr.csv',
                                         'eer': 12.17}}}

    OUTPUT_DIR = os.path.dirname(BASE_DIR) + "/figures/" + f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(OUTPUT_DIR)

    # Plot HUMI
    plot_curves_1_dataset('humi', 'scrolldown_scrollup_det',
                          [(results_csv['humi']['SD']['csv_file'], f"SD - EER ({results_csv['humi']['SD']['eer']}%)", 'without_imu', 'r'),
                           (results_csv['humi']['SD, A, G, M']['csv_file'], f"SD, A, G, M - EER ({results_csv['humi']['SD, A, G, M']['eer']}%)", 'with_imu', 'r'),
                           (results_csv['humi']['SU']['csv_file'], f"SU - EER ({results_csv['humi']['SU']['eer']}%)", 'without_imu', 'b'),
                           (results_csv['humi']['SU, A, G, M']['csv_file'], f"SU, A, G, M - EER ({results_csv['humi']['SU, A, G, M']['eer']}%)", 'with_imu', 'b'),
                           ])
    
    # Plot FETA
    plot_curves_1_dataset('feta', 'scrolldown_scrollup_det',
                          [(results_csv['feta']['SD']['csv_file'], f"SD - EER ({results_csv['feta']['SD']['eer']}%)", 'without_imu', 'r'),
                           (results_csv['feta']['SD, A, G']['csv_file'], f"SD, A, G - EER ({results_csv['feta']['SD, A, G']['eer']}%)", 'with_imu', 'r'),
                           (results_csv['feta']['SU']['csv_file'], f"SU - EER ({results_csv['feta']['SU']['eer']}%)", 'without_imu', 'b'),
                           (results_csv['feta']['SU, A, G']['csv_file'], f"SU, A, G - EER ({results_csv['feta']['SU, A, G']['eer']}%)", 'with_imu', 'b'),
                           ])