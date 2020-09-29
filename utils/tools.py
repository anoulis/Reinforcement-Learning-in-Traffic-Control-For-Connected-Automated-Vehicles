import xml.etree.ElementTree as ElementTree
import csv
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as pl
import os


def main():
    main_path = []
    outputs_path = 'outputs/simulations/'
    if not os.path.exists(outputs_path+'comparisons'):
        os.makedirs(outputs_path+'comparisons')

    # declare model's stored files
    main_path.append('outputs/simulations/runner/')
    main_path.append('final_models/sims/dnq_sample3015.zip_20200613-143151/')
    main_path.append('final_models/sims/dnq_dc_seed_30.zip_20200624-101517/')
    main_path.append('final_models/sims/test4.zip_20200717-114640/')
    # main_path.append('final_models/sims/bad_test_2.zip_20200727-100032/')


    iteration_number = 10

    # we go for every model

    data = []
    for k in range(len(main_path)):
        meanSpeed = []
        TravelTime = []
        avg_cav_dist = None
        duration_list = []
        reward = 0
        wt_list = []
        wc_list = []
        tl_list = []
        dd_list = []

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
                        reward += float(row[3])

            meanSpeed.append(ms/line_count)
            TravelTime.append(tt/line_count)

            # xml files part    
            local_path = main_path[k] + "tripinfo_"
            filepath = local_path+str(i+1) + ".xml"
            tree = ElementTree.parse(filepath)
            root = tree.getroot()
            d=0
            dd = 0
            wt=0
            wc=0
            tl=0
            counts=0
            for child in root:
                counts +=1
                d+=float(child.attrib['duration'])
                wt+=float(child.attrib['waitingTime'])
                wc += int(child.attrib['waitingCount'])
                tl+=float(child.attrib['timeLoss'])
                dd+= float(child.attrib['departDelay'])

            duration_list.append(d/counts)
            wt_list.append(wt)
            tl_list.append(tl/counts)
            dd_list.append(dd/counts)
            wc_list.append(wc)

            data.append({'id': int(k+1), 'simulation': int(i+1),  'avg_cav_dist': avg_cav_dist, 'reward': reward,
                         'meanSpeed':  meanSpeed[-1], 'travelTime': TravelTime[-1], 
                         'duration': duration_list[-1], 'timeLoss': tl_list[-1], 'departDelay': dd_list[-1], 'waitingTime': wt_list[-1], 'waitingCount': wc_list[-1]})

    df = pd.DataFrame(data)
    df.to_csv('outputs/simulations/comparisons/full_data.csv', index=False)
    myBoxPlots('outputs/simulations/comparisons/full_data.csv',iteration_number)


def myBoxPlots(path,iteration_number):
    data_path=path
    dataframe = pd.read_csv(data_path, index_col=0)
    ids = int(len(dataframe['waitingTime'])/iteration_number)
    flag=0
    wt_sum = []
    wc_sum = []
    id = []
    for i in range(ids):
        id.append(i+1)
        wt_sum.append(sum(dataframe['waitingTime'][flag:flag+iteration_number]))
        wc_sum.append(sum(dataframe['waitingCount'][flag:flag+iteration_number]))
        flag+=iteration_number

    # fig, ax = pl.subplots()
    y_pos = np.arange(len(id))
    pl.bar(y_pos, wt_sum,width = 0.2)
    pl.title('Waiting Time Plot')
    pl.xlabel('Models')
    pl.ylabel('Overall WT in secs')
    # Limits for the Y axis
    # pl.ylim(0.0,10.0)
    pl.xticks(y_pos, id)
    pl.savefig('outputs/simulations/comparisons/waitingTime.png')

    pl.bar(y_pos, wc_sum,width = 0.2)
    pl.title('Waiting Count Plot')
    pl.xlabel('Models')
    pl.ylabel('Overall number of WC')
    # Limits for the Y axis
    # pl.ylim(0.0,10.0)
    pl.xticks(y_pos, id)
    pl.savefig('outputs/simulations/comparisons/waitingCount.png')

    df = pd.read_csv(data_path, sep=',', na_values='.')
    bp_ms = df.boxplot(column='meanSpeed', by='id')
    bp_ms.set_title('Mean Speed of both lanes in m/s')
    # pl.title('Mean Speed Plot')
    # pl.xlabel('Models')
    # pl.ylabel('MeanSpeed in m/s')
    pl.savefig('outputs/simulations/comparisons/meanSpeed.png')
    bp_tt = df.boxplot(column='travelTime', by='id')
    bp_tt.set_title('Travel Time of both lanes in s')
    pl.savefig('outputs/simulations/comparisons/travelTime.png')
    bp_d = df.boxplot(column='duration', by='id')
    bp_d.set_title('Average Duration in simulation s ')
    pl.savefig('outputs/simulations/comparisons/duration.png')
    bp_dd = df.boxplot(column='departDelay', by='id')
    bp_dd.set_title('Average Departure Delay in simulation s')
    pl.savefig('outputs/simulations/comparisons/departDelay.png')
    bp_tl = df.boxplot(column='timeLoss', by='id')    
    bp_tl.set_title('Average Time Loss in s')
    pl.savefig('outputs/simulations/comparisons/timeLoss.png')
    bp_r = df.boxplot(column='reward', by='id')
    bp_r.set_title('Total Reward')
    pl.savefig('outputs/simulations/comparisons/reward.png')
    # bp_wt = df.boxplot(column='waitingTime', by='id')
    # df.waitingTime.plot(color='g',lw=1.3)
    # df.id.plot(color='r',lw=1.3)
    # pl.savefig('outputs/simulations/comparisons/waitingTime.png')
    # bp_wc = df.boxplot(column='waitingCount', by='id')
    # pl.savefig('outputs/simulations/comparisons/waitingCount.png')
    bp_cd = df.boxplot(column='avg_cav_dist', by='id')
    bp_cd.set_title('Average covered distance of CAVs in m')
    pl.savefig('outputs/simulations/comparisons/avg_cav_dist.png')
    # pl.show()


if __name__ == '__main__':
    main()
