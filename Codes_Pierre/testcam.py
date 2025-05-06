import crappy
import ft232R
import time

date = "2025_03_11"

cam_R = crappy.blocks.Camera(camera="XiAPI", 
                               config=True, 
                               save_images=True,
                               save_folder=f"./{date}/r",
                               img_shape=(2048,2048), 
                               img_dtype='uint8',
                               **{"serial_number": "14482450",
                                  "exposure": 80000,
                                  "trigger": "Software"})

cam_L = crappy.blocks.Camera(camera="XiAPI",
                               config=False,
                               save_images=True,
                               save_folder=f"./{date}/l",
                               img_shape=(2048,2048),
                               img_dtype='uint8',
                               **{"serial_number": "32482550",
                                  "exposure": 80000,
                                  "trigger": "Software"})

#ftdi = crappy.blocks.IOBlock('Ft232r', cmd_labels=['cmd'], spam=False, direction=0b0001, URL='ftdi://ftdi:232:A105QJ01/1')


#gen_ft = crappy.blocks.Generator([{'type': 'Cyclic',
                                     #'value1': 0, 'condition1': 'delay=4',
                                     #'value2': 1, 'condition2': 'delay=1', 'cycles': 50}], cmd_label='cmd')

#crappy.link(gen_ft, ftdi)

crappy.start()




