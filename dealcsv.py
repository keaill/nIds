import csv
import pandas as pd


#打开文件
path='D:\\Program Files (x86)\\program\\UNSW-NB15 - CSV Files\\UNSW-NB15 - CSV Files\\a part of training and testing set\\UNSW_NB15_training-set.csv'
def main():
    try:
        data= pd.read_csv(path,engine = "python")
        print(data)
    except Exception as e:
        print(e, type(e))
        if (isinstance(e, pd.errors.EmptyDataError)):
            print("这里对空行文件进行处理")
    # df = pd.DataFrame(data)
    

        # rows = [row for row in reader]
    num=0
    for i in range(len(data)):


        # print(data["attack_cat"][i])
        if data["attack_cat"][i]!=233:
        
            

            if data["attack_cat"][i]=='Reconnaissance':#Generic,Exploits,Reconnaissance,Fuzzers,DoS,Worms,Backdoor,Analysis
            #Shellcode
                num=num+1
    print(num)
    #             if num<=5000:
    #                 if num%100==0:
    #                     print(num)
    #                 data.drop(i,inplace=True)
    # data.to_csv('D:\\Program Files (x86)\\program\\UNSW-NB15 - CSV Files\\UNSW-NB15 - CSV Files\\a part of training and testing set\\UNSW_NB15_training-set.csv')             



if __name__ == "__main__":
    main()




