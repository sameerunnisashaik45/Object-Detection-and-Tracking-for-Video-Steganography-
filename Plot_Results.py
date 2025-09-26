import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import warnings

warnings.filterwarnings('ignore')


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'FANO-DWT', 'LOA-DWT', 'GAO-DWT', 'SFOA-DWT ', 'IRN-ISOA-DWT']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve')
    length = np.arange(Fitness.shape[1])
    plt.plot(length, Fitness[0, :], color='#FF69B4', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label=Algorithm[1])
    plt.plot(length, Fitness[1, :], color='#7D26CD', linewidth=3, marker='*', markerfacecolor='#00FFFF',
             markersize=12, label=Algorithm[2])
    plt.plot(length, Fitness[2, :], color='#FF00FF', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label=Algorithm[3])
    plt.plot(length, Fitness[3, :], color='#43CD80', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label=Algorithm[4])
    plt.plot(length, Fitness[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label=Algorithm[5])
    plt.xlabel('No. of Iteration', fontname="Arial", fontsize=16, fontweight='bold', color='k')
    plt.ylabel('Cost Function', fontname="Arial", fontsize=16, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=16, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=16, fontweight='bold', color='k')
    plt.legend(loc=1, prop={'weight': 'bold', 'size': 14})
    plt.savefig("./Results/Conv.png")
    plt.show()


def plot_Results():
    Eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['SSIM', 'PSNR', 'RMSE', 'MI', 'Entropy', 'MSE', 'NC']
    Algorithm = ['FANO-DWT', 'LOA-DWT', 'GAO-DWT', 'SFOA-DWT ', 'IRN-ISOA-DWT']
    Configurations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for b in range(len(Terms)):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
        fig.canvas.manager.set_window_title(Terms[b] + '- Image Method Comparison')
        length = np.arange(len(Configurations))
        ax.plot(length, Eval[:, 0, b], color='#8338ec', linewidth=5, marker='o',
                markersize=10,
                label=Algorithm[0])

        ax.plot(length, Eval[:, 1, b], color='#fb8b24', linewidth=5, marker='o',
                markersize=10,
                label=Algorithm[1])

        ax.plot(length, Eval[:, 2, b], color='#1a936f', linewidth=5, marker='o',
                markersize=10,
                label=Algorithm[2])

        ax.plot(length, Eval[:, 3, b], color='#f504c9', linewidth=5, marker='o',
                markersize=10,
                label=Algorithm[0])
        ax.plot(length, Eval[:, 4, b], color='#9a031e', linewidth=5, marker='o', markersize=10,
                label=Algorithm[3])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)
        plt.xticks(length, (
            'V1', 'V2', 'V3', 'V4',
            'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11'),fontname="Arial", fontsize=16, fontweight='bold', color='k')

        plt.ylabel(Terms[b], fontname="Arial", fontsize=16, fontweight='bold', color='k')
        plt.xlabel('Videos', fontname="Arial", fontsize=16, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=16, fontweight='bold', color='k')
        plt.grid(axis='y')
        # Custom Legend with Dot Markers, positioned at the top
        color = ['#8338ec', '#fb8b24', '#1a936f', '#f504c9', '#9a031e']
        dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                       in color]
        plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=10,
                   frameon=False, ncol=len(Algorithm), prop={'weight':'bold', 'size':12})
        path = "./Results/%s_Alg_bar.png" % (Terms[b])
        plt.savefig(path)
        plt.show()


def Table():
    Eval_all = np.load('Evaluates_all.npy', allow_pickle=True)
    Terms = ['SSIM', 'PSNR', 'RMSE', 'MI', 'Entropy', 'MSE']
    Methods = ['Terms', 'FANO-DWT', 'LOA-DWT', 'GAO-DWT', 'SFOA-DWT ', 'IRN-ISOA-DWT']
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    value_all = Eval_all[:, :, :]
    stats = np.zeros((value_all.shape[1], 5))
    for i in range(value_all.shape[2]):
        for m in range(value_all.shape[1]):
            stats[m, 0] = np.max(value_all[:, m, i])
            stats[m, 1] = np.min(value_all[:, m, i])
            stats[m, 2] = np.mean(value_all[:, m, i])
            stats[m, 3] = np.median(value_all[:, m, i])
            stats[m, 4] = np.std(value_all[:, m, i])

        Table = PrettyTable()
        Table.add_column(Methods[0], Statistics[2::])
        Table.add_column(Methods[1], stats[0, 2::])
        Table.add_column(Methods[2], stats[1, 2::])
        Table.add_column(Methods[3], stats[2, 2::])
        Table.add_column(Methods[4], stats[3, 2::])
        Table.add_column(Methods[5], stats[4, 2::])
        print('-------------------------------------------------- ', Terms[i],
              'Algorithm Comparison', '--------------------------------------------------')
        print(Table)


def plot_key_sencitivity():
    sencitivity = np.load('key.npy', allow_pickle=True)
    colors = ['#ce4257', '#73d2de', 'y', 'm', 'k']
    Algorithm = ['ECC', 'DES', 'AES', 'FHE', 'ElGamal-AS']
    X = np.arange(5)
    bar_width = 0.15
    x = np.arange(sencitivity.shape[0])

    fig = plt.figure()
    fig.canvas.manager.set_window_title('Key Sensitivity Analysis')
    ax = fig.add_axes([0.12, 0.25, 0.8, 0.65])
    fig.canvas.manager.set_window_title('Algorithm Comparison of Block Size')
    # Plot bars for each category
    for i in range(len(Algorithm)):
        bars = plt.bar(x + i * bar_width, sencitivity[:, i], width=bar_width, label=Algorithm[i], color=colors[i])

    # Customizations
    # Remove axes outline
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.figtext(0.08, 0.08, '01', fontsize=18, ha='center', color='w',
                bbox=dict(boxstyle='circle', facecolor='#ce4257', edgecolor='white'))
    plt.figtext(0.125, 0.08, Algorithm[0], fontsize=10, fontweight='bold', color='k', ha='left')

    plt.figtext(0.25, 0.08, '02', fontsize=18, ha='center', color='w',
                bbox=dict(boxstyle='circle', facecolor='#73d2de', edgecolor='white'))
    plt.figtext(0.288, 0.08, Algorithm[1], fontsize=10, fontweight='bold', color='k', ha='left')

    plt.figtext(0.41, 0.08, '03', fontsize=18, ha='center', color='w',
                bbox=dict(boxstyle='circle', facecolor='m', edgecolor='white'))
    plt.figtext(0.455, 0.08, Algorithm[2], fontsize=10, fontweight='bold', color='k', ha='left')

    plt.figtext(0.62, 0.08, '04', fontsize=18, ha='center', color='w',
                bbox=dict(boxstyle='circle', facecolor='y', edgecolor='white'))
    plt.figtext(0.665, 0.08, Algorithm[3], fontsize=10, fontweight='bold', color='k', ha='left')
    #
    plt.figtext(0.80, 0.08, '05', fontsize=18, ha='center', color='w',
                bbox=dict(boxstyle='circle', facecolor='k', edgecolor='white'))
    plt.figtext(0.835, 0.08, Algorithm[4], fontsize=10, fontweight='bold', color='k', ha='left')

    # Add gridlines for y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.xticks(X + 0.30, ['1', '2', '3', '4', '5'], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
    plt.xlabel('Cases', fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.ylabel('Key Sensitivity Analysis', fontname="Arial", fontsize=12, fontweight='bold', color='k')
    path = "./Results/Key_sen_enc_Bar.png"
    plt.savefig(path)
    plt.show()


def Encryption_Results(file_name, ylabel, title):
    Eval = np.load(file_name, allow_pickle=True)[0]
    Algorithm = ['LSB', 'PVM', 'DCT', 'PCF  ', 'DWT']
    colors = ['#ce4257', '#73d2de', 'y', 'm', 'k']
    Configurations = [1, 2, 3, 4]
    fig = plt.figure()
    ax = fig.add_axes([0.12, 0.12, 0.8, 0.65])
    X = np.arange(len(Configurations))
    # Plotting the bars
    ax.bar(X + 0.00, Eval[:, 0], color=colors[0], width=0.15, label=Algorithm[0])
    ax.bar(X + 0.15, Eval[:, 1], color=colors[1], width=0.15, label=Algorithm[1])
    ax.bar(X + 0.30, Eval[:, 2], color=colors[2], width=0.15, label=Algorithm[2])
    ax.bar(X + 0.45, Eval[:, 3], color=colors[3], width=0.15, label=Algorithm[3])
    ax.bar(X + 0.60, Eval[:, 4], color=colors[4], width=0.15, label=Algorithm[4])

    # Remove axes outline
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=12) for color
                   in colors]
    plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.27), fontsize=10,
               frameon=False, ncol=3, prop={'weight':'bold', 'size':14})
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.xticks(X + 0.30, ['128 X 128', '256 X 256', '512 X 512', '1024 X 1024'], fontname="Arial", fontsize=16,
               fontweight='bold',
               color='#35530a')
    plt.yticks(fontname="Arial", fontsize=16, fontweight='bold', color='k')
    plt.xlabel('Image Size', fontname="Arial", fontsize=16, fontweight='bold', color='k')
    plt.ylabel(ylabel, fontname="Arial", fontsize=16, fontweight='bold', color='k')
    path = f"./Results/{title}.png"
    plt.savefig(path)
    plt.show()


def Encryt_Results():
    Encryption_Results('Execution_Time.npy', 'Execution Time (S)', 'Execution_Time_Embedding')
    Encryption_Results('Embed_Capcity.npy', 'Embedding Capacity (bpp)', 'Encryption time_Embedding')


def Object_Plots_Results():
    eval = np.load('Obj_Evaluation.npy', allow_pickle=True)
    Terms = ['Tracking Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 3, 8]
    bar_width = 0.15
    Video = ['360', '480', '720', '1000']
    Algorithm = ['MVD', 'dMVC', 'DAR-EfficientNetV2', 'EfficientDet', 'RA-EDNet']
    Classifier = ['Ref 2', 'Ref 3', 'ResNet', 'SENet-MSA', 'Proposed']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + 'Algorithm Comparison of Kfold')
            X = np.arange(len(Video))
            plt.bar(X + 0.00, Graph[:, 0], color='darkblue', edgecolor='w', linewidth=2, width=0.15,
                    label=Algorithm[0])
            plt.bar(X + 0.15, Graph[:, 1], color='#9400d3', edgecolor='w', linewidth=2, width=0.15,
                    label=Algorithm[1])
            plt.bar(X + 0.30, Graph[:, 2], color='#a30046', edgecolor='w', linewidth=2, width=0.15,
                    label=Algorithm[2])
            plt.bar(X + 0.45, Graph[:, 3], color='#00bbf9', edgecolor='w', linewidth=2, width=0.15,
                    label=Algorithm[3])
            plt.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', linewidth=2, width=0.15,
                    label=Algorithm[4])
            plt.xticks(X + bar_width * 2, ['360p', '480p', '720p', '1000p'], fontsize=14,
                       fontname="Arial",
                       fontweight='bold', color='k')
            plt.xlabel('Video Resolution', fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=14, fontname="Arial", fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)

            # Custom Legend with Dot Markers, positioned at the top
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                           in ['darkblue', '#9400d3', '#a30046', '#00bbf9', 'k']]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=10,
                       frameon=False, ncol=3, prop={'weight':'bold', 'size':12})
            plt.tight_layout()
            path = "./Results/%s_bar_Object_Detection.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def MAP():
    eval = np.load('MAP.npy', allow_pickle=True)
    bar_width = 0.15
    kfold = [1, 2, 3, 4]
    Algorithm = ['MVD', 'dMVC', 'DAR-EfficientNetV2', 'EfficientDet', 'RA-EDNet']
    Classifier = ['Ref 2', 'Ref 3', 'ResNet', 'SENet-MSA', 'Proposed']

    fig = plt.figure()
    ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
    X = np.arange(len(kfold))
    plt.bar(X + 0.00, eval[:, 0], color='darkblue', edgecolor='w', linewidth=2, width=0.15,
            label=Algorithm[0])
    plt.bar(X + 0.15, eval[:, 1], color='#9400d3', edgecolor='w', linewidth=2, width=0.15,
            label=Algorithm[1])
    plt.bar(X + 0.30, eval[:, 2], color='#a30046', edgecolor='w', linewidth=2, width=0.15,
            label=Algorithm[2])
    plt.bar(X + 0.45, eval[:, 3], color='#00bbf9', edgecolor='w', linewidth=2, width=0.15,
            label=Algorithm[3])
    plt.bar(X + 0.60, eval[:, 4], color='k', edgecolor='w', linewidth=2, width=0.15,
            label=Algorithm[4])
    plt.xticks(X + bar_width * 2, ['1', '2', '3', '4'], fontsize=16,
               fontname="Arial",
               fontweight='bold', color='k')
    plt.xlabel('Kfold', fontname="Arial", fontsize=16, fontweight='bold', color='#14213d')
    plt.ylabel('MAP', fontsize=16, fontname="Arial", fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=16, fontweight='bold', color='#35530a')
    # Remove axes outline
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)

    # Custom Legend with Dot Markers, positioned at the top
    dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                   in ['darkblue', '#9400d3', '#a30046', '#00bbf9', 'k']]
    plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=10,
               frameon=False, ncol=3, prop={'weight':'bold', 'size':12})
    plt.tight_layout()
    path = "./Results/MAP_bar_Object_Detection.png"
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    plotConvResults()
    plot_Results()
    Table()
    Encryt_Results()
    Object_Plots_Results()
    MAP()
