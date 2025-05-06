#echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb

import crappy
import tridicalddef
import ft232R
import alicat_flow_crappy


if __name__ == "__main__":
  
  ver = crappy.blocks.VideoExtenso(camera='XiAPI',
                                   config=True, 
                                   save_images=False,
                                   labels=['tr(s)', 'meta_r', 'pix_r', 'eyy_r', 'exx_r'],
                                   white_spots=False,
                                   **{"serial_number": "50480150",
                                      "exposure": 50000,
                                      "trigger": "Hdw after config"}) 

  vel = crappy.blocks.VideoExtenso(camera='XiAPI', 
                                   config=True,
                                   save_images=False,
                                   labels=['tl(s)', 'meta_l', 'pix_l', 'eyy_l', 'exx_l'],
                                   white_spots=False,
                                   **{"serial_number": "50480250",
                                      "exposure": 50000,
                                      "trigger": "Hdw after config"})
                                   
  ftdi = crappy.blocks.IOBlock('Ft232r', cmd_labels=['cmd'], spam=False, direction=0b0001, URL='ftdi://0x0403:0x6001:A105QJ01/1')
  
  flow_ali = crappy.blocks.IOBlock('Flow_controller_alicat',
                                   port="/dev/ttyUSB0", 
                                   startbit=1,
                                   databits=8,
                                   parity="N",
                                   stopbits=1,
                                   errorcheck="crc",
                                   baudrate=9600,
                                   method="RTU",
                                   timeout=3,
                                   press='relative',
                                   svp=['Pressure', 'Mass_flow'],
                                   cmd_labels=['flowcmd'],
                                   labels=['t(s)', 'press', 'mass_flow'])
  
  trid = tridicalddef.Tridical(direct_A_file='./2023_08_25/40d_cd/coeff_direct.npy',
                               transfomatrix_file='./2023_08_25/40d_cd/transfomatrix.npy',
                               label = 'exx')

  gen_ft = crappy.blocks.Generator([{'type': 'Cyclic', 
                                     'value1': 0, 'condition1': 'delay=0.25',
                                     'value2': 1, 'condition2': 'delay=0.25', 'cycles': 120}], cmd_label='cmd')

  gen_flow = crappy.blocks.Generator([{'type': 'Constant',
                                       'value': 100,
                                       'condition': 'delay=60'}], cmd_label='flowcmd') 
    
  rec_tridexx = crappy.blocks.Recorder(file_name = './2023_08_25/40d_cd/defexxsvp_f100')
  
  rec_ali = crappy.blocks.Recorder(file_name='./2023_08_25/40d_cd/ali_f100')
  
                                   
  crappy.link(vel, trid)
  crappy.link(ver, trid)
  crappy.link(gen_ft, ftdi)
  crappy.link(gen_flow, flow_ali)
  
  crappy.link(trid, rec_tridexx)
  crappy.link(flow_ali, rec_ali)

  crappy.start()
  
