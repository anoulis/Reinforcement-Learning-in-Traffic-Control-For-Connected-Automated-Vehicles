import xml.etree.ElementTree as ElementTree
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl


def main():
    main_path = []
    outputs_path = 'outputs/simulations/'
    # declare model's stored files
    # main_path.append('outputs/simulations/dqn_3trains_pun1.3.zip_20200609-193214/')
    # main_path.append('outputs/simulations/dqn_3trains_pun1.3.zip_20200611-095309/')
    iteration_number = 10

    # we go for every model

    data = []
    for k in range(len(main_path)):
        meanSpeed = []
        TravelTime = []
        avg_cav_dist = None
        duration_list = []
        wt_list = []
        tl_list = []

        for i in range(iteration_number):
            
            # csv files part
            local_path = main_path[k] + "data_run"
            filepath = local_path+str(i+1) + ".csv"
            ms = 0
            tt = 0
            line_count = 0
            with open(filepath) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if line_count == 0:
                        # print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    else:
                        ms += float(row[4])
                        tt += float(row[5])
                        avg_cav_dist = float(row[2])
                        line_count += 1

            meanSpeed.append(ms/line_count)
            TravelTime.append(tt/line_count)

            # xml files part    
            local_path = main_path[k] + "tripinfo_"
            filepath = local_path+str(i+1) + ".xml"
            tree = ElementTree.parse(filepath)
            root = tree.getroot()
            d=0
            wt=0
            tl=0
            counts=0
            for child in root:
                counts =+1
                d+=float(child.attrib['duration'])
                wt+=float(child.attrib['waitingTime'])
                tl+=float(child.attrib['timeLoss'])
            
            duration_list.append(d/counts)
            wt_list.append(wt/counts)
            tl_list.append(tl/counts)

            data.append({'id': int(k+1), 'simulation': int(i+1),  'avg_cav_dist': avg_cav_dist,
                         'meanSpeed':  meanSpeed[-1], 'travelTime': TravelTime[-1], 
                         'duration': duration_list[-1], 'timeLoss': tl_list[-1], 'waitingTime': wt_list[-1]})

    df = pd.DataFrame(data)
    df.to_csv('outputs/simulations/comparisons/full_data.csv', index=False)
    myBoxPlots('outputs/simulations/comparisons/full_data.csv')


def myBoxPlots(path):
    data_path=path
    gapminder = pd.read_csv(data_path)
    df = pd.read_csv(data_path, sep=',', na_values='.')
    bp_ms = df.boxplot(column='meanSpeed', by='id')
    pl.savefig('outputs/simulations/comparisons/meanSpeed.png')
    bp_tt = df.boxplot(column='travelTime', by='id')
    pl.savefig('outputs/simulations/comparisons/travelTime.png')
    bp_d = df.boxplot(column='duration', by='id')
    pl.savefig('outputs/simulations/comparisons/duration.png')
    bp_tl = df.boxplot(column='timeLoss', by='id')    
    pl.savefig('outputs/simulations/comparisons/timeLoss.png')
    bp_wt = df.boxplot(column='waitingTime', by='id')
    pl.savefig('outputs/simulations/comparisons/waitingTime.png')
    bp_cd = df.boxplot(column='avg_cav_dist', by='id')
    pl.savefig('outputs/simulations/comparisons/avg_cav_dist.png')
    # pl.show()


if __name__ == '__main__':
    main()
