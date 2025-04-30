#echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
import crappy
import time
import moteur
import ft232R
import os 
from glob import glob
import pycaso as pcs
import numpy as np




if __name__ == '__main__':
  date = "2025_04_28"



  cam_R = crappy.blocks.Camera(camera="XiAPI",
                              config=True,
                              save_images=True,
                              save_folder=f"./{date}/r",
                              timeout=100000,
                              img_shape=(2048,2048),
                              img_dtype='uint8',
                              **{"serial_number": "14482450",
                                  "exposure": 56229,
                                  "trigger": "Hdw after config"})

  cam_L = crappy.blocks.Camera(camera="XiAPI",
                              config=True,
                              save_images=True,
                              save_folder=f"./{date}/l",
                              timeout=100000,
                              img_shape=(2048,2048),
                              img_dtype='uint8',
                              **{"serial_number": "32482550",
                                  "exposure": 75197,
                                  "trigger": "Hdw after config"})



  ftdi = crappy.blocks.IOBlock('Ft232r', cmd_labels=['cmd'], spam=False, direction=0b00000100, URL='ftdi://ftdi:232:FTU7DIHC/1')

  mot = crappy.blocks.Machine([{'type': 'Printer',
                                'mode': 'position',
                                'cmd_label': 'pos',
                                'position_label': 'position',
                                'speed': 100,
                                'port': '/dev/ttyACM0'}],
                              freq=50)


  gen_ft = crappy.blocks.Generator([{'type': 'Cyclic',
                                    'value1': 0, 'condition1': 'delay=4',
                                    'value2': 1, 'condition2': 'delay=1', 'cycles': 50}], cmd_label='cmd')

  path_mot=[]
  pas=-5
  for i in range(125, 25+pas, pas):
    path_mot.append({'type': 'Constant',
                      'value': i,
                      'condition': 'delay=5'})


  gen_mot = crappy.blocks.Generator(path=path_mot, cmd_label='pos')

  graph_mot = crappy.blocks.Grapher(('t(s)','position'))

  record_mot=crappy.blocks.Recorder(file_name=f"./{date}/z_list",
                                    labels=['t(s)','position'],
                                    freq=1,
                                    delay=5)






  crappy.link(gen_ft, ftdi)
  crappy.link(gen_mot, mot)
  crappy.link(mot,graph_mot)
  crappy.link(mot,record_mot)

  crappy.start()
