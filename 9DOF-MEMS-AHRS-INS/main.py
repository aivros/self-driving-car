# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
import numpy as np
#from . import convolve
# -----------------------------------------------------------------------------
import time 
import pandas as pd
import re
import serial 
import serial.tools.list_ports
from scipy import stats
from scipy.signal import savgol_filter
from scipy.fftpack import fft
from matplotlib import pyplot as plt
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#原型脚手架
class Scanffold(object):
    path = ''    
    port_list = list(serial.tools.list_ports.comports())
    if len(port_list) == 0:
       print('找不到串口')
    else:
        for i in range(0,len(port_list)):
            print(port_list[i])                 
    ser = serial.Serial('COM5', 115200)      
    write = '{\"type\":\"toy\",\"func\":\"search\"};\r\n'
    #print(write)
    search = write.encode('ascii')
    #print(search)
    ser.write(search)    
    headers = ['y','x','z','time']
    df = pd.DataFrame(columns = headers)
    def __init__(self):
        self.serial_data = []
        self.aa = 0
        self.bb = 0
        self.str_list = []
        self.str_list_ = []
        self.search_key = 'd' + 'a' + 't' + 'a' + '"' + ':' + '"'
        self.par= "-?\d+\.*\d*"
        self.status_array = [0] * 60
        self.serial_line = 0
        self.hhh = 0
        self.aa = 0
        self.bb = 0
        self.line_data = 0

    def serial_get(self):
        port_list = list(serial.tools.list_ports.comports())
        if len(port_list) == 0:
           print('找不到串口')
        else:
            for i in range(0,len(port_list)):
                print(port_list[i])      
                
        self.ser = serial.Serial('COM5', 115200)      
        write = '{\"type\":\"toy\",\"func\":\"search\"};\r\n'
        #print(write)
        search = write.encode('ascii')
        #print(search)
        self.ser.write(search)

    def line_to_dataframe(self):

        lenth_line = len(self.serial_data)/4
        if str(lenth_line).isdigit() and lenth_line != 1:
            line_ = []
            for i in self.serial_data:
                line_.append(i)
                if len(line_) == 4:
                    del line_[3]
                    line_.append(int(round(time.time() * 1000)))
                    #print(line_)
                    Scanffold.df = Scanffold.df.append(pd.Series(line_, index=Scanffold.df.columns ), ignore_index=True)
                    line_ = []
                else:
                    None
        elif lenth_line == 1:
            Scanffold.df = Scanffold.df.append(pd.Series(self.serial_data, index=Scanffold.df.columns ), ignore_index=True)   
        return  

    def serial_stream(self):
        self.aa = int(round(time.time() * 1000))
        if self.aa - self.bb > 100:
            #print('time out:',self.aa,self.bb)
            serial_data = 0
            pass
        #print(self.ser.in_waiting)
        ser_in_waiting = self.ser.in_waiting
        if ser_in_waiting: 
            #print(self.ser.in_waiting)               
            self.serial_line = self.ser.read(ser_in_waiting)                
            #print(f'serial_line: {self.serial_line}')
            self.str_line = str(self.serial_line)
            self.str_list.append(self.str_line)                
               
            if '}}' in str(self.str_line) or '"}' in str(self.str_line):
                #self.hhh += 1                
                for i in self.str_list:
                    i = i.lstrip("'b\'")
                    i = "".join(re.findall("[^\']+",i))
                    self.str_list_.append(i)                    
                jsonData = "".join(self.str_list_)  
                jsonData.lstrip("'")
                #print(f'jsonData: {jsonData}')                  
                #serial_data = re.findall(r"\d+\.?\d*",jsonData)

                if (self.search_key in jsonData) and ('"}}' or '"}' in jsonData):
                    #index_one = jsonData.index(self.search_key)
                    
                    #print(f'{len(jsonData)}jsonData: {jsonData}')
                    if len(jsonData) > 150:
                        jsonData = jsonData[-100:]
                    #print(f'{len(jsonData)},jsonData  : {jsonData}') 
                
                    index_one = [i.start() for i in re.finditer(self.search_key, jsonData)]
                    #print(index_one)
                    if len(index_one) == 1:
                        if '"}}' in jsonData:    
                            index_two = jsonData.index('"}}')
                        elif '"}' in jsonData:
                            index_two = jsonData.index('"}')
    
                        serial_data = jsonData[index_one[0] + 7 : index_two]
                        
                        serial_data = re.compile(self.par).findall(serial_data)
                        serial_data = list(map(int, serial_data))
                        #data_dict = json.loads(jsonData)
                        self.serial_data = serial_data
                        current_time = int(round(time.time() * 1000))
                        serial_data.append(current_time)
                        #print(serial_data)
                        self.str_list = []
                        self.str_list_ = []
                        self.bb = int(round(time.time() * 1000))

                        self.line_to_dataframe()
                        
                    elif len(index_one) > 1:
                        self.str_list = []
                        self.str_list_ = []  
                        jsonData = ''
                        serial_data = 0
                    else:
                        serial_data = 0
                else:
                    serial_data = 0
            else:
                serial_data = 0
        else:
            serial_data = 0

        return serial_data   

    def write_csv(self):            
        with open(Scanffold.path + 'sum.txt', 'r') as f:
            xxx = f.read()
        if len(xxx) < 20:
            xxx = xxx + 'x'
        else:
            xxx = 'x'
        Scanffold.df.to_csv(self.path + '//' + xxx + '.csv')
        print(Scanffold.df)
        print(len(xxx))
        with open(Scanffold.path + 'sum.txt', 'w') as f:
            f.write(xxx)
            #print(f.read())    
        return  

    def detected_status(self):
        data = np.sqrt(np.square(Parameter.accelerometer_x) + np.square(Parameter.accelerometer_y) + np.square(Parameter.accelerometer_z))
        data_fft = fft(data)
        savgol_data = savgol_filter(data_fft, 33, 3)
        detected_data = np.abs(savgol_data[int(len(savgol_data)/6):int(len(savgol_data)/2)])
        peak_num = np.argmax(detected_data)
        del self.status_array[0]  
        self.status_array.append(peak_num)   
        high_low_fft_proportion = stats.mode(self.status_array)[0][0]
        if high_low_fft_proportion < 125 and high_low_fft_proportion > 0:
            vibration_level = 3
            airbag_level = 0
        elif high_low_fft_proportion < 175 and high_low_fft_proportion > 125:
            vibration_level = 2
            airbag_level = 0
        elif high_low_fft_proportion <= 200 and high_low_fft_proportion > 175:
            vibration_level = 1
            airbag_level = 0
        else:
            vibration_level = 0
            airbag_level = 0
            
        vibration_level=3
        airbag_level=0
        return vibration_level, airbag_level
    
    
    
    
    
    
    def toEuler(self, quat):
        #from paper: "Adaptive Filter for a Miniature MEMS Based Attitude and
        #Heading Reference System" by Wang et al, IEEE.
        R = np.zeros([5,1])
        R[0] = 2. * np.power(quat[0],2) - 1 + 2. * np.power(quat[1],2)
        R[1] = 2. * (quat[1]*quat[2] - quat[0]*quat[3])
        R[2] = 2. * (quat[1]*quat[3] + quat[0]*quat[2])
        R[3] = 2. * (quat[2]*quat[3] - quat[0]*quat[1])
        R[4] = 2. * np.power(quat[0],2) - 1 + 2. * np.power(quat[3],2)
        phi = np.arctan2(R[3],R[4])
        theta = -np.arctan(R[2] / np.sqrt(1 - np.power(R[2],2)))
        psi = np.arctan2(R[1],R[0])
        phi = np.rad2deg(phi)
        theta = np.rad2deg(theta)
        psi = np.rad2deg(psi)
        return [phi[0],theta[0],psi[0]]

    def toGravity(self, quat):
        x = 2. * (quat[1] * quat[3] - quat[0]*quat[2])
        y = 2. * (quat[0] * quat[1] + quat[2]*quat[3])
        z = quat[0]*quat[0] - quat[1]*quat[1] - quat[2]*quat[2] + quat[3]*quat[3]
        return [x,y,z]
    
    def gravity_compensate(self, q, acc):
        g = [0.0, 0.0, 0.0]

        # get expected direction of gravity
        g[0] = 2 * (q[1] * q[3] - q[0] * q[2])
        g[1] = 2 * (q[0] * q[1] + q[2] * q[3])
        g[2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]

        # compensate accelerometer readings with the expected direction of gravity
        return [acc[0] - g[0], acc[1] - g[1], acc[2] - g[2]]

#------------------------------------------------------------------------------



#剥离主体
#------------------------------------------------------------------------------
#参数空间
class Parameter(object):
    main_param = {
        '00': {'strength': {'data': {'data_length': 300}, 
                            'para': {'data_resection': {'hig_low': 'l', 'threshold' : 100}, 'crosszero' : {'data_length' : 10}}, 
                            'proc': ['E', 'F', 'G'] #  'A', 
                           }
              },
        '10': {'strength': {'data': {'data_length': 300}, 
                            'para': {'data_resection': {'hig_low': 'l', 'threshold' : 500}, 'crosszero' : {'data_length' : 15}}, 
                            'proc': ['B', 'E', 'F', 'G']
                           }
              },
        '20': {'strength': {'data': {'data_length': 300}, 
                            'para': {'data_resection': {'hig_low': 'l', 'threshold' : 500}, 'crosszero' : {'data_length' : 15}}, 
                            'proc': ['B', 'E', 'F', 'G']
                           }
              },
        '30': {'strength': {'data': {'data_length': 300}, 
                            'para': {'data_resection': {'hig_low': 'l', 'threshold' : 500}, 'crosszero' : {'data_length' : 15}}, 
                            'proc': ['B', 'E', 'F', 'G']
                           }
              },
        '40': {'strength': {'data': {'data_length': 300}, 
                            'para': {'data_resection': {'hig_low': 'l', 'threshold' : 500}, 'crosszero' : {'data_length' : 15}}, 
                            'proc': ['B', 'E', 'F', 'G']
                           }
              }
    }#超参数字典,只读 

    stream_data = 0
    vibrate_data = 0
    airbag_data = 0
    time_sequence = [0] * 300
    accelerometer_x = [0] * 300
    accelerometer_y = [0] * 300
    accelerometer_z = [0] * 300
    quaternion = [0] * 4
    
    fifo_data_x = [0] * 100
    
    test_all_sum = 0
    test_all_slight = 0
    test_all_strength = 0
    test_all_slight_fast = 0

    def fifo_main(self):#FIFO
        del Parameter.time_sequence[0]
        del Parameter.accelerometer_x[0]
        del Parameter.accelerometer_y[0]
        del Parameter.accelerometer_z[0]
        Parameter.time_sequence.append(Parameter.stream_data[3])
        Parameter.accelerometer_x.append(Parameter.stream_data[1] - 150)
        Parameter.accelerometer_y.append(Parameter.stream_data[0])
        Parameter.accelerometer_z.append(Parameter.stream_data[2])
        Parameter.fifo_data_x = Parameter.accelerometer_x
        Parameter.fifo_data_y = Parameter.accelerometer_y
        Parameter.fifo_data_z = Parameter.accelerometer_y
        return

#------------------------------------------------------------------------------
#流程模块
class Processor(Parameter):
    def __init__(self):
        self.data = [0] * 60

        self.original_data = [0] * 60#------------------------------------------------------------------优化
        self.crosszero_boolean = False
        self.up_down = 0
        #self.original_up_down = [0] * 15
        #self.smooth_cut = False
        self.cross_rate = 0
        #self.previous_data = [0] * 15
        self.previous_cross_data = [0] * 15
        self.cross_value = 0
        
        
        self.plot_lowpass = 0
        self.plot_interp = 0
        self.plot_resection = 0

    def get_status(self):#状态分类
        if Parameter.vibrate_data == 0:
            set_para = '00'
        elif Parameter.vibrate_data ==1:
            set_para = '10'
        elif Parameter.vibrate_data ==2:
            set_para = '20'
        elif Parameter.vibrate_data ==3:
            set_para = '30'
        elif Parameter.vibrate_data == 4 or 5 or 6 or 7:
            set_para = '40'
        return set_para

    def noise_suppression(self, data_mean, data_rate):#噪声排除
        if data_mean < 4350 and data_mean > 3900 and data_rate < 0.55 and data_rate > 0.45:
            return False
        elif data_mean < 1550 and data_mean > 1450 and data_rate < 0.51 and data_rate > 0.45:
            return False
        elif data_mean < 3100 and data_mean > 3000 and data_rate < 0.55 and data_rate > 0.5:
            return False
        elif data_mean < 4100 and data_mean > 3900 and data_rate < 0.55 and data_rate > 0.45:
            return False
        elif data_mean < 400:
            return False
        else:
            return True
    
    def get_mean(self, data, length=30):#
        #获取平均值
        data = data[-length:]
        data = np.array(data)
        data_mean = np.mean(abs(data))
        #print(data_mean)
        return data_mean



    def noise_check(self, data):
        if data.any():
            return True

        else:
            return False






    def data_resection(self, hig_low, threshold): #A #self.data读取类型:一维数组, 返回类型:一维数组
        #数据切除
        #print(self.data)
        if hig_low == 'h':
            numpy_array = np.array(self.data)
            numpy_array[abs(numpy_array) > threshold] = 0
        else:
            numpy_array = np.array(self.data)
            numpy_array[abs(numpy_array) < threshold] = 0
            
        #print(f'numpy_array: {numpy_array}')
        self.data = numpy_array
        
        self.plot_resection = self.data.copy()
        #print(f'self.plot_resection: {self.plot_resection}')
        return
    
    def data_resection_(self, data, hig_low, threshold): #A #self.data读取类型:一维数组, 返回类型:一维数组
        #数据切除
        #print(self.data)
        if hig_low == 'h':
            numpy_array = np.array(data)
            numpy_array[abs(numpy_array) > threshold] = 0
        else:
            numpy_array = np.array(data)
            numpy_array[abs(numpy_array) < threshold] = 0
            
        #print(f'numpy_array: {numpy_array}')
        data = numpy_array
        
        #self.plot_resection = self.data.copy()
        #print(f'self.plot_resection: {self.plot_resection}')
        return data

    def sinc(self, data):
        pi = 3.141592653589793
        x = np.asanyarray(data)
        y = pi * np.where(x == 0, 1.0e-20, x)
        return np.sin(y)/y



    def low_filter(self):#B   lowpass_blackman_sinc
        #低通虑波
        s = self.data#一维数组
        fs = 60
        low_cut = 4
        fc = low_cut/fs
        b = 0.08 
        N = int(np.ceil((4 / b)))
        if not N % 2: N += 1 
        n = np.arange(N)
        h = np.sinc(2 * fc * (n - (N - 1) / 2))
        w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
            0.08 * np.cos(4 * np.pi * n / (N - 1))
        h = h * w
        h = h / np.sum(h)
        s = np.convolve(s, h)
        self.data = s[26:326] #一维数组s
        
        self.plot_lowpass = self.data.copy()
        
        return 
    
    def low_filter_(self, data):#B   lowpass_blackman_sinc
        #低通虑波
        s = data#一维数组
        fs = 60
        low_cut = 4
        fc = low_cut/fs
        b = 0.08 
        N = int(np.ceil((4 / b)))
        if not N % 2: N += 1 
        n = np.arange(N)
        h = np.sinc(2 * fc * (n - (N - 1) / 2))
        w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
            0.08 * np.cos(4 * np.pi * n / (N - 1))
        h = h * w
        h = h / np.sum(h)
        s = np.convolve(s, h)
        data = s[26:326] #一维数组s
        
        #self.plot_lowpass = self.data.copy()
        
        return data
    

    def zero_scope_interp(self, value_list ,zero_scope_extremum):
        interp_list_all = []
        for i, v in enumerate(value_list):
            interp_list = []
            for x in range(v-1):
                mean_num = (zero_scope_extremum[i][1] - zero_scope_extremum[i][0])/v*(x+1)
                interp_ = zero_scope_extremum[i][0] + mean_num
                interp_list.append(interp_)
            interp_list_all.append(interp_list)
        return interp_list_all
     

    def zero_interp_index(self, zero_scope):
        interp_index_list = []
        for v in zero_scope:
            index_list = []
            for x in range(v[1] - v[0] -1):
                index_ = v[0] + 1 + x 
                index_list.append(index_)
            interp_index_list.append(index_list)
        return interp_index_list

    
    def zero_scope(self, index_list, nonzero_index):
        zero_scope_ = []
        for i in index_list:
            ii = [nonzero_index[i], nonzero_index[i+1]]
            zero_scope_.append(ii)
        return zero_scope_


    def zero_scope_extremum(self, data, zero_scope):
        zero_scope_extremum_ = []
        for i in zero_scope:
            ii = [data[i[0]], data[i[1]]]
            zero_scope_extremum_.append(ii)
        return zero_scope_extremum_


    def interp_1d(self):#M   #self.data读取类型:一维数组, 返回类型:一维数组
        #线性插值
        nonzero_index = self.data.nonzero()[0]
        #data_nonzero = self.data[nonzero_index]
        diff_nonzero = np.diff(nonzero_index)
        index_list = []
        value_list = []
        for index, value in enumerate(diff_nonzero):
            if value != 1:
                index_list.append(index)
                value_list.append(value)

        zero_scope = self.zero_scope(index_list, nonzero_index)#[[nonzero_index[i], nonzero_index[i+1]] for i in index_list]
        zero_scope_extremum = self.zero_scope_extremum(self.data, zero_scope)#[[self.data[i[0]], self.data[i[1]]] for i in zero_scope]

        zero_scope_interp = self.zero_scope_interp(value_list ,zero_scope_extremum)
        #zero_scope_interp = [[zero_scope_extremum[i][0] + (zero_scope_extremum[i][1] - zero_scope_extremum[i][0])/v*(x+1) for x in range(v-1)] for i, v in enumerate(value_list)]
        zero_interp_index = self.zero_interp_index(zero_scope)
        #zero_interp_index = [[v[0] + 1 + x for x in range(v[1] - v[0] -1)] for v in zero_scope]
        for index, value in enumerate(zero_interp_index):
            self.data[value] = zero_scope_interp[index]
            
            
        self.plot_interp = self.data.copy()
        
        return
    
    
    
    
    
    
    
    
    
    

    def crosszero(self, data_length):#E#self.data读取类型:一维数组, 返回类型:无(void)
        #跨零检测
        self.data = self.data[-data_length :]
        self.original_cross = self.original_data[-data_length :]
        cross_zero = np.sign(self.data) 
        self.cross_zero = np.where(np.diff(cross_zero))[0]
        
        self.plot_cross = self.data.copy()
        
        return self.cross_zero

    def crosszero_rate(self, data_length=60):#K
        #跨零率
        data_cross = self.crosszero(data_length)
        self.cross_rate = len(data_cross) / data_length
        return self.cross_rate



    def peak_check(self, data):
        pass


    def sim_data(self):#F#读取类型:一维数组, 返回类型:一维数组
        #检测跨零方向,定义去重检测变量
        if len(self.cross_zero) == 1:
            self.cross_value = self.data[self.cross_zero[0] + 1]
            #print(f'self.cross_value: {self.cross_value}')
            self.crosszero_boolean = True
            if self.data[self.cross_zero[0]] > 0:
                self.up_down = 0
            else:
                self.up_down = 1
            self.data = self.original_cross[-15:]#[self.data[self.cross_zero[0]], self.original_cross[self.cross_zero[0]], self.original_cross[self.cross_zero[0] + 1]]
            #print(len(self.data),self.data)
        return

    def similarity(self):#G#读取类型:一维数组, 返回类型:int
        #两个数组相似度比较
        if self.crosszero_boolean: 
            self.crosszero_boolean = False
            cross_data = self.previous_cross_data
            current_data = self.data
            sim = 0
            for i in cross_data:
                for ii in current_data:
                    if i == ii:
                        sim += 1
            self.data = sim
        return 

#------------------------------------------------------------------------------
#状态分拣
class Sorting(Processor):

    def __init__(self, sli_stren):
        Processor.__init__(self)
        self.sli_stren = sli_stren
        self.set_para = ''
        self.noise_mean = [0] * 10
        self.noise_rate = [0] * 10
        self.vibration_command = False
        self.data_array = [0] * 300
        
    #当前数据只有一个状态
    def pre_process(self):#预处理,处理区间分类
        self.data = Parameter.fifo_data_x
        data_mean = self.get_mean(self.data)
        #print(f'data_mean: {data_mean}')
        data_cross = self.crosszero(data_length=20)
        data_cross_rate = len(data_cross) / 20
        #print(f'data_cross_rate: {data_cross_rate}')
        
        del self.noise_mean[0]
        self.noise_mean.append(data_mean)
        value_noise = self.get_mean(self.noise_mean, length=10)
        #print(f'value_noise: {value_noise}')
        
        del self.noise_rate[0]
        self.noise_rate.append(data_cross_rate)
        #print(self.noise_rate)
        rate_noise = self.get_mean(self.noise_rate, length=10)
        #print(f'rate_noise: {rate_noise}')
        
        noise_vib = self.noise_suppression(value_noise, rate_noise)
        #print(f'noise_vib: {noise_vib}')
        
        self.set_para = self.get_status()
        
        if self.set_para != '00':
            
            lowpass_data = self.low_filter_(self.data)
            lowcut_data = self.data_resection_(lowpass_data, 'l', 1200)
            noise_check = self.noise_check(lowcut_data)
        else:
            noise_check = True
        
        if noise_vib and noise_check:
            if data_mean != 0:
                if self.sli_stren == 'slight':
                    pass
                elif self.sli_stren == 'slight_fast':
                    pass
                elif self.sli_stren == 'strength':
                    if self.set_para == '10':#10-13
                        volume = 900
                        crosszero_ = 0.35
                        if data_mean > volume and data_cross_rate < crosszero_:
                            return True
                        else:
                            return False
                    elif self.set_para == '20':#20-23
                        volume = 1500
                        crosszero_ = 0.4
                        if data_mean > volume and data_cross_rate < crosszero_:
                            return True
                        else:
                            return False
                    elif self.set_para == '30':#30-33
                        volume = 2200
                        crosszero_ = 0.4
                        if data_mean > volume and data_cross_rate < crosszero_:
                            return True
                        else:
                            return False
                    elif self.set_para == '40':#40-70,1-3
                        volume = 2500
                        crosszero_ = 0.4
                        if data_mean > volume and data_cross_rate < crosszero_:
                            return True
                        else:
                            return False
                    else:
                        volume = 400
                        crosszero_rate = 0.1  
                        if data_mean > volume and data_cross_rate < crosszero_rate:
                            return True
                        else:
                            return False 
        return False

    def sorting_func(self):#参数字典获取
        #self.set_para = str(Parameter.vibrate_data) + str(Parameter.airbag_data)
        self.set_status = Parameter.main_param[self.set_para]
        return

    def data_proc(self):#字典获取和数据准备---------------------------------------------------------------------优化
        data_length = self.set_status[self.sli_stren]['data']['data_length']
        self.data_array = Parameter.fifo_data_x[-data_length :]  #循序初次数据
        return

    def param_proc(self):#取出参数
        self.param = self.set_status[self.sli_stren]['para']
        return

    def process_control(self):#处理流程
        proc = self.set_status[self.sli_stren]['proc']
        self.original_data = self.data_array#-----------------------------------------------------------------优化
        #self.original_up_down = self.data_array
        self.data =  self.data_array#-------------------------------------------------------------------------优化
        for i in proc:
            if i == 'A':
                self.data_resection(self.param['data_resection']['hig_low'], self.param['data_resection']['threshold'])#data, hig_low, threshold
            elif i == 'B':
                self.low_filter()
            elif i == 'C':
                self.high_filter()
            elif i == 'D':
                self.smooting()
            elif i == 'E':
                self.crosszero(self.param['crosszero']['data_length'])
            elif i == 'F':
                self.sim_data()
            elif i == 'G':
                self.similarity()
            elif i == 'H':
                self.interp()
            elif i == 'I':
                self.smoo_sim_data()
            elif i == 'J':
                self.smoothing_similarity()
            elif i == 'M':
                self.interp_1d()
            else:
                None
        return

    def send_command(self):#发送震动指令
        if self.up_down:
            up_down = '上'
            style = '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
        else:
            up_down = '下'
            style = '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
        if  type(self.data) == type(1) and self.data == 0 and self.cross_value:
            
            self.cross_value = 0
            self.vibration_command = True
            self.previous_cross_data = self.original_data[-15:]
            print(f'检测到向{up_down}的动作! {style}')
            #---------------------------------------------
            '''
            plot_data(self.original_data)
            print(self.original_data)
            plot_data(self.plot_lowpass)
            print(self.plot_lowpass)
            
            plot_data(self.plot_resection)#self.plot_resection
            print(self.plot_resection)
            
            #plot_data(self.plot_interp)
            #print(self.plot_interp)
            print(f'self.cross_value: {self.cross_value}')'''
            #plot_data(self.plot_lowpass)
            
            #plot_data(self.data)
            #调试部分
            print(self.set_para)
            print(self.sli_stren)
            Parameter.test_all_sum += 1
            if self.sli_stren == 'slight':
                Parameter.test_all_slight += 1
            elif self.sli_stren == 'strength':
                Parameter.test_all_strength += 1
            elif self.sli_stren == 'slight_fast':
                Parameter.test_all_slight_fast += 1
        return
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#一般力度
class Strength(Sorting):
    def __init__(self):
        Sorting.__init__(self, 'strength')

    def strength(self):
        if self.pre_process():
            self.sorting_func()
            self.data_proc()
            self.param_proc()
            self.process_control()
            self.send_command()
#-----------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
            
def plot_data(data):
    #file_name = str(data)
    plt.figure(figsize=(15,8))
    #plt.figure(1)  
    plt.plot(data,'r')
    #plt.plot(eeee,'g')
    #plt.plot(ffff,'b') 
    #plt.title(file_name)
    plt.grid()
    plt.show()
#执行主函数
def main():
    parameter = Parameter()
    scanffold = Scanffold()
    strength = Strength()
    try:
        while True:

            serial_data = scanffold.serial_stream()
            vibration_level, airbag_level = scanffold.detected_status()
            if serial_data == 0:
                continue
            if len(serial_data) != 4:
                continue

            #Parameter.quaternion = serial_data

            Parameter.stream_data = serial_data
            Parameter.vibrate_data = vibration_level
            Parameter.airbag_data = airbag_level
            parameter.fifo_main()
            strength.strength()

    except KeyboardInterrupt:
        print('interrupted!')
    #打印检测结果
    print(f'检测个数:{Parameter.test_all_sum}')
    print(f'slight: {Parameter.test_all_slight}')
    print(f'slight_fast:{Parameter.test_all_slight_fast}')
    print(f'strength: {Parameter.test_all_strength}')
    scanffold.ser.close()
    scanffold.write_csv()
    return
#--------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
