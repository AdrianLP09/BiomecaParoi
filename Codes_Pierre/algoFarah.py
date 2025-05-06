#echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
import crappy
import time
import moteur
import ft232R
import os 
import glob


if __name__ == '__main__':

  cam_R = crappy.blocks.Camera(camera="XiAPI", 
                               config=False, 
                               save_images=True,
                               save_folder="./2023_07_06/40d_cd/right/", 
                               img_shape=(2048,2048), 
                               img_dtype='uint8',
                               **{"serial_number": "50480150",
                                  "exposure": 80000,
                                  "trigger": "Hardware"})
                                                                                             
  cam_L = crappy.blocks.Camera(camera="XiAPI", 
                               config=False, 
                               save_images=True,                               
                               save_folder="./2023_07_06/40d_cd/left/", 
                               img_shape=(2048,2048), 
                               img_dtype='uint8', 
                               **{"serial_number": "50480250",
                                  "exposure": 80000,
                                  "trigger": "Hardware"})
  
  ftdi = crappy.blocks.IOBlock('Ft232r', cmd_labels=['cmd'], spam=False, direction=0b0001, URL='ftdi://0x0403:0x6001:A105QJ01/1')
  
  mot = crappy.blocks.Machine([{'type': 'Printer',
                                'mode': 'position',
                                'cmd_label': 'pos',
                                'position_label': 'position',
                                'speed': 100,
                                'port': '/dev/ttyACM0'}],
                              freq=50)

  
  gen_mot = crappy.blocks.Generator([{'type': 'Constant',
                                      'value': 20,
                                      'condition': 'delay=60'}], cmd_label='pos')
  
  gen_ft = crappy.blocks.Generator([{'type': 'Cyclic', 
                                     'value1': 0, 'condition1': 'delay=0.25',
                                     'value2': 1, 'condition2': 'delay=0.25', 'cycles': 120}], cmd_label='cmd')
  
  graph_mot = crappy.blocks.Grapher(('t(s)','position'))
  
  rec_mot = crappy.blocks.Recorder(file_name="./2023_07_06/40d_cd/data_pos_mot", delay=4)
  
  crappy.link(gen_ft, ftdi)
  crappy.link(gen_mot, mot)
  crappy.link(mot, graph_mot, modifier=crappy.modifier.Mean(10)) 
  crappy.link(mot, rec_mot)
  
  crappy.start()

