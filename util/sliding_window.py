import numpy as np
import numpy as np


def sliding_window(dataset : list,args):
        
        ws = args.window_size

        if args.datasets == 'Yahoo':
            
            state_set = {} #OUTPUT; consist of timestamp, value, label
            time_state =[]  # sliding_window
            value_state = []
            label_state = []

            for data_i in dataset: # real_i or synthetic_i : set
                ts = np.array(data_i['timestamp']) #list
                val = np.array(data_i['value']) 
                label = np.array(data_i['label'])
                
                time_i = []
                value_i = []
                label_i = []
                num_samples = len(ts) - ws
                for j in range(num_samples):
                    ts_ = ts[j:j+ws]   #list
                    val_ = [val[j+1:j+ws+1]]
                    val_.append(label[j:j+ws])
                    lab_ = label[j+ws]

                    time_i.append(ts_)
                    value_i.append(val_)
                    label_i.append(lab_)

                time_state.append(time_i)
                value_state.append(value_i)
                label_state.append(label_i)

            state_set['timestamp'] = time_state
            state_set['value'] = value_state
            state_set['label'] = label_state      

            return state_set

        elif args.datasets == 'SWaT':
            pass

        elif args.datasets == 'Numenta':
            pass
        
        elif args.datasets == 'KPI':
            pass

# def sliding_window(dataset : list,args):
        
#         ws = args.window_size

#         if args.datasets == 'Yahoo':
            
#             state_set = {} #OUTPUT; consist of timestamp, value, label
#             time_state =[]  # sliding_window
#             value_state = []
#             label_state = []

#             for data_i in dataset: # real_i or synthetic_i : set
#                 ts = np.array(data_i['timestamp']) #list
#                 val = np.array(data_i['value']) 
#                 label = np.array(data_i['label'])
                
#                 time_i = []
#                 value_i = []
#                 label_i = []
#                 num_samples = len(ts) - ws + 1
#                 for j in range(num_samples):
#                     ts_ = ts[j:j+ws]   #list
#                     val_ = val[j:j+ws]
#                     lab_ = label[j+ws-1]

#                     time_i.append(ts_)
#                     value_i.append(val_)
#                     label_i.append(lab_)

#                 time_state.append(time_i)
#                 value_state.append(value_i)
#                 label_state.append(label_i)

#             state_set['timestamp'] = time_state
#             state_set['value'] = value_state
#             state_set['label'] = label_state      

#             return state_set

#         elif args.datasets == 'SWaT':
#             pass

#         elif args.datasets == 'Numenta':
#             pass
        
#         elif args.datasets == 'KPI':
#             pass