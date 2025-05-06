#echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb

import crappy
import tridicald_crappy
import ft232R


if __name__ == "__main__":

  date = "2023_12_06"
  
  ver = crappy.blocks.VideoExtenso(camera='XiAPI',
                                   config=True, 
                                   save_images=True,
                                   save_folder=f'./{date}/40d_cd/SC37_40/right/',
                                   labels=['tr(s)', 'meta_r', 'pix_r', 'eyy_r', 'exx_r'],
                                   white_spots=False,
                                   **{"serial_number": "50480150",
                                      "exposure": 50000,
                                      "trigger": "Hdw after config"}) 

  vel = crappy.blocks.VideoExtenso(camera='XiAPI', 
                                   config=True,
                                   save_images=True,
                                   save_folder=f'./{date}/40d_cd/SC37_40/left/',
                                   labels=['tl(s)', 'meta_l', 'pix_l', 'eyy_l', 'exx_l'],
                                   white_spots=False,
                                   **{"serial_number": "50480250",
                                      "exposure": 50000,
                                      "trigger": "Hdw after config"})
                                   
  ftdi = crappy.blocks.IOBlock('Ft232r', cmd_labels=['cmd'], spam=False, direction=0b0001, URL='ftdi://0x0403:0x6001:A105QJ01/1')
  
  trid = tridicald_crappy.Tridical(direct_a_file=f'./{date}/40d_cd/SC37_40/coeff_direct.npy',
                               trans_matrix_file=f'./{date}/40d_cd/SC37_40/transfomatrix.npy',
                               label = 'triexx')

  gen_ft = crappy.blocks.Generator([{'type': 'Cyclic', 
                                     'value1': 1, 'condition1': 'delay=0.05',
                                     'value2': 0, 'condition2': 'delay=0.05', 'cycles': 10}],
                                   cmd_label='cmd')
                            
  rec_tridexx = crappy.blocks.Recorder(file_name = f'./{date}/40d_cd/SC37_40/defexxsvp')

  crappy.link(gen_ft, ftdi)
  
  crappy.link(vel, trid)
  crappy.link(ver, trid)
    
  crappy.link(trid, rec_tridexx)
  
  crappy.start()
  
