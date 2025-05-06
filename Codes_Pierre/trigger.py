import crappy
import time

      
class Triggercamexxlin(crappy.blocks.Block):

  def __init__(self,
               cmd_labels: list) -> None:
               
    crappy.blocks.Block.__init__(self)
    self.cmd_labels = cmd_labels
    self.out = {}

  
  def begin(self):
  
    self.out['cmd'] = 1
    self.send(self.out)

  
  def prepare(self) -> None:
    
    self.values = {}
    for label in self.cmd_labels:
      self.values[label] = None
    self.i = 2 #le temps de résoudre pb inout   
      
     
  def loop(self):

    data = self.recv_all_data()
    for label in self.cmd_labels:
       if label in data.keys():
         self.values[label] = data[label]
    if self.values['exxcmd'] is None or self.values['exx'] is None:
      return  

    else:
      self.out['cmd'] = self.i  #le temps de résoudre pb inout
      self.send(self.out)
      self.i +=1                #le temps de résoudre pb inout      
      
    for label in self.cmd_labels:
      self.values[label] = None 


class Triggercamexx(crappy.blocks.Block):

  def __init__(self,
               cmd_labels: list) -> None:
               
    crappy.blocks.Block.__init__(self)
    self.cmd_labels = cmd_labels
    self.out = {}

  
  def begin(self):
  
    self.out['cmd'] = 1
    self.send(self.out)

  
  def prepare(self) -> None:
    
    self.values = {}
    for label in self.cmd_labels:
      self.values[label] = None
    self.i = 2 #le temps de résoudre pb inout   
      
     
  def loop(self):

    data = self.recv_all_data()
    for label in self.cmd_labels:
       if label in data.keys():
         self.values[label] = data[label]
    if self.values['exxcmd'] is None or self.values['exx'] is None:
      return  

    if (float(self.values['exxcmd'][0]) - float(self.values['exx'][0])) > float(self.values['exxcmd'][0])*5/100 or (float(self.values['exxcmd'][0]) - float(self.values['exx'][0])) < -float(self.values['exxcmd'][0])*5/100:
#      self.out['cmd'] = 1      # le temps de résoudre pb inout
      self.out['cmd'] = self.i  #le temps de résoudre pb inout
      self.send(self.out)
      self.i +=1                #le temps de résoudre pb inout
#      print(self.out)

    else:
      print('okboooooooomCbueeeennnnoooooooo')
#      self.out['cmd'] = 1      #le temps de résoudre pb inout
      self.out['cmd'] = self.i  #le temps de résoudre pb inout
      self.send(self.out)
      self.i +=1                #le temps de résoudre pb inout      
    
    for label in self.cmd_labels:
      self.values[label] = None 
            
      
class Triggerali(crappy.blocks.Block):

  def __init__(self,
               Pseuil: float,
               Miniflow: float,
               cmd_labels: list) -> None:
               
    crappy.blocks.Block.__init__(self)
    self.Pseuil = Pseuil
    self.Miniflow = Miniflow
    self.cmd_labels = cmd_labels
    self.out = {}

  def prepare(self):
    
    self.values = {}
    for label in self.cmd_labels:
      self.values[label] = None
           
  def loop(self):

    data = self.recv_all_data()
    for label in self.cmd_labels:
       if label in data.keys():
         self.values[label] = data[label]
    if self.values['press'] is None or self.values['exx'] is None:
      return  

    if self.values['press'][0] < self.Pseuil:
      self.out['flowcmd'] = self.Miniflow
      self.out['Mullins'] = 1
      self.send(self.out)
      self.out['flowcmd'] = None

    else:
      print('c bueno')
      self.out['Mullins'] = 0
      self.out['exx'] = self.values['exx'][0]  
#      self.out['dexx'] = self.values['dexx'][0]
#      print(self.out)
      self.send(self.out)
      
    for label in self.cmd_labels:
      self.values[label] = None 
            
