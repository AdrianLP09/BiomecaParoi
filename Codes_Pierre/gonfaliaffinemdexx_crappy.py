#echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb

import crappy
import tridicalddef
import ft232R
import trigger
import alicat_flow_crappy


if __name__ == "__main__":

#à peu près carré si pas Mullins pour p=2800 et i = 2000
  p = 1500
  i = 1100
  d = 0
  dexx = 0.001
  o = 0
  O = 100
  
  ver = crappy.blocks.VideoExtenso(camera='XiAPI',
                                   config=True, 
                                   save_images=True,
                                   save_folder=f'./2023_08_25/40d_cd/p{p}i{i}dexx{dexx}o{o}O{O}/right/',
                                   labels=['tr(s)', 'meta_r', 'pix_r', 'eyy_r', 'exx_r'],
                                   white_spots=False,
                                   **{"serial_number": "50480150",
                                      "exposure": 50000,
                                      "trigger": "Hdw after config"}) 

  vel = crappy.blocks.VideoExtenso(camera='XiAPI', 
                                   config=True,
                                   save_images=True,
                                   save_folder=f'./2023_08_25/40d_cd/p{p}i{i}dexx{dexx}o{o}O{O}/left/',
                                   labels=['tl(s)', 'meta_l', 'pix_l', 'eyy_l', 'exx_l'],
                                   white_spots=False,
                                   **{"serial_number": "50480250",
                                      "exposure": 50000,
                                      "trigger": "Hdw after config"})
                                   
  ftdi = crappy.blocks.IOBlock('Ft232r', 
                               cmd_labels=['cmd'], 
                               spam=False, direction=0b0001, 
                               URL='ftdi://0x0403:0x6001:A105QJ01/1')

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
                            
  trig_cam = trigger.Triggercamexx(cmd_labels=['exx', 'exxcmd'])
  
  trig_ali = trigger.Triggerali(Pseuil=6,
                                Miniflow=7,
                                cmd_labels=['exx', 'press', 'dexx'])
  
  gen_exx = crappy.blocks.Generator([{'type': 'Constant',
                                      'value': 0,
                                      'condition': 'Mullins<1'},
                                     {'type': 'Constant',
                                      'value': 0.12,
                                      'condition': 'delay=100'}], cmd_label='exxcmd', spam=True)                                 

  gen_dexx = crappy.blocks.Generator([{'type': 'Constant',
                                      'value': 0,
                                      'condition': 'Mullins<1'},
                                      {'type': 'Constant',
                                       'value': dexx,
                                       'condition':'delay=100'}], cmd_label='dexxcmd', spam=True)
  
  pid = crappy.blocks.PID(kp=p,
                          ki=i,
                          kd=d,
                          out_max=O,
                          out_min=o,
                          setpoint_label='dexxcmd',
                          input_label='dexx',
                          labels=['t(s)', 'flowcmd'],
                          debug=False)
  
  rec_tridexx = crappy.blocks.Recorder(file_name=f'./2023_08_25/40d_cd/p{p}i{i}dexx{dexx}o{o}O{O}/defexxsvp')
  
  rec_pid = crappy.blocks.Recorder(file_name=f'./2023_08_25/40d_cd/p{p}i{i}deex{dexx}o{o}O{O}/pid')  

  rec_ali = crappy.blocks.Recorder(file_name=f'./2023_08_25/40d_cd/p{p}i{i}dexx{dexx}o{o}O{O}/ali')
    
                                   
  crappy.link(gen_exx, trig_cam)
  crappy.link(trid, trig_cam)
  crappy.link(trig_cam, ftdi)
  
  crappy.link(vel, trid)
  crappy.link(ver, trid)

  crappy.link(trid, trig_ali)
  crappy.link(flow_ali, trig_ali)
  crappy.link(trig_ali, flow_ali)
  crappy.link(trig_ali, gen_exx)
  
  crappy.link(gen_dexx, pid)
  crappy.link(trig_ali, pid)
  crappy.link(pid, flow_ali)
  
  crappy.link(trid, rec_tridexx)
  crappy.link(pid, rec_pid)  
  crappy.link(flow_ali, rec_ali)
  
  crappy.start()
  
