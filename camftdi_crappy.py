#echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb

import crappy
import time
import ft232R


if __name__ == "__main__":

  date = "2025_05_15"

  cam_R = crappy.blocks.Camera(camera="XiAPI", 
                               config=False,
                               save_images=True,
                               save_folder=f"./{date}/matrix_calibR/",
                               timeout=100000,
                               img_shape=(2048,2048), 
                               img_dtype='uint8',
                               **{"serial_number": "14482450",
                                  "exposure": 56229,
                                  "trigger": "Hardware"})
                                                                                             
  cam_L = crappy.blocks.Camera(camera="XiAPI", 
                               config=False,
                               save_images=True,                               
                               save_folder=f"./{date}/matrix_calibL/",
                               timeout=100000,
                               img_shape=(2048,2048), 
                               img_dtype='uint8', 
                               **{"serial_number": "32482550",
                                  "exposure": 75197,
                                  "trigger": "Hardware"})
                                                                                             
  ftdi = crappy.blocks.IOBlock('Ft232r', cmd_labels=['cmd'], spam=False, direction=0b00000100, URL='ftdi://ftdi:232:FTU7DIHC/1')

  gen_ft = crappy.blocks.Generator([{'type': 'Cyclic', 
                                     'value1': 1, 'condition1': 'delay=0.05',
                                     'value2': 0, 'condition2': 'delay=0.05', 'cycles': 1}], cmd_label='cmd')
                                     
#  gen_ft2 = crappy.blocks.Generator([{'type': 'Constant',
#                                     'value': 1,
#                                     'condition': 'delay=0.5'}], freq=10, spam=True)                                   

  crappy.link(gen_ft, ftdi)
  crappy.start()

