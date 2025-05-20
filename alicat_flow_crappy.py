
#requirements: pip install pymodbus==3.0.2

import crappy
import time
import pymodbus
from pymodbus.client.serial import ModbusSerialClient
from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder
from pymodbus.constants import Endian
import struct
import ft232R


if pymodbus.__version__=='3.0.2':

  class Flow_controller_alicat(crappy.inout.InOut):

    def __init__(self,
                port="/dev/ttyUSB0",
                startbit=1,
                databits=8,
                parity="N",
                stopbits=1,
                errorcheck="crc",
                baudrate=115200,
                method="RTU",
                timeout=3,
                press='absolue',
                svp=[]):

      crappy.inout.InOut.__init__(self)
      self.port = port
      self.startbit = startbit
      self.databits = databits
      self.parity = parity
      self.stopbits = stopbits
      self.errorcheck = errorcheck
      self.baudrate = baudrate
      self.method = method
      self.timeout = timeout
      self.press = press
      self.svp = svp
      self.address_list = {#Dict that contains the registers' adress and count associated to the information you want to get
              "Pressure": [1202,2],
              "Temperature": [1204,2],
              "Volumetric_flow": [1206,2],
              "Mass_flow": [1208,2]}

    def open(self):

      self.client = ModbusSerialClient(port=self.port,
                                      startbit=self.startbit,
                                      databits=self.databits,
                                      parity=self.parity,
                                      stopbits=self.stopbits,
                                      errorcheck=self.errorcheck,
                                      baudrate=self.baudrate,
                                      method=self.method,
                                      timeout=self.timeout)
      self.client.connect()
      if self.press == 'relative':
        response = self.client.read_holding_registers(address=self.address_list['Pressure'][0], count=self.address_list['Pressure'][1], unit=1)
        decoder = BinaryPayloadDecoder.fromRegisters(response.registers, byteorder=Endian.Big, wordorder=Endian.Big)
        self.P0 = decoder.decode_32bit_float()

    def get_data(self):

      data = []
      if self.svp:
        for i in self.svp:
          response = self.client.read_holding_registers(address=self.address_list[i][0], count=self.address_list[i][1], unit=1)
          decoder = BinaryPayloadDecoder.fromRegisters(response.registers, byteorder=Endian.Big, wordorder=Endian.Big)
          value = decoder.decode_32bit_float()
          if i == 'Pressure' and self.press == 'relative':
            data.append(value-self.P0)
          else:
  #        print(i, value)
            data.append(value)
        return [time.time()] + data

    def set_cmd(self, cmd):

      if cmd == None:
        pass
      else:
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Big)
        builder.add_32bit_float(cmd)
        payload = builder.build()
        self.client.write_registers(address=1009, values=payload, count=2, unit= 1, skip_encode = True)

    def close(self):

      builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Big)
      builder.add_32bit_float(0)
      payload = builder.build()
      self.client.write_registers(address=1009, values=payload, count=2, unit= 1, skip_encode = True)
      self.client.close()


# requirements: pip install pymodbus==3.8.6

if pymodbus.__version__=='3.8.6':

  class Flow_controller_alicat(crappy.inout.InOut):

      def __init__(self,
                  port="/dev/ttyUSB0",
                  startbit=1,
                  databits=8,
                  parity="N",
                  stopbits=1,
                  errorcheck="crc",
                  baudrate=115200,
                  method="RTU",
                  timeout=3,
                  press='absolue',
                  svp=[]):

          crappy.inout.InOut.__init__(self)
          self.port = port
          self.startbit = startbit
          self.databits = databits
          self.parity = parity
          self.stopbits = stopbits
          self.errorcheck = errorcheck
          self.baudrate = baudrate
          self.timeout = timeout
          self.press = press
          self.svp = svp
          self.address_list = {  # Dict that contains the registers' address and count associated with the information you want to get
              "Pressure": [1202, 2],
              "Temperature": [1204, 2],
              "Volumetric_flow": [1206, 2],
              "Mass_flow": [1208, 2]
          }

      def open(self):
          # Initialize the ModbusSerialClient without the 'method' argument
          self.client = ModbusSerialClient(
              port=self.port,
              baudrate=self.baudrate,
              stopbits=self.stopbits,
              parity=self.parity,
              timeout=self.timeout
          )

          # Connect to the device
          if not self.client.connect():
              raise Exception(f"Unable to connect to the Modbus device at {self.port}")

          if self.press == 'relative':
              response = self.client.read_holding_registers(address=self.address_list['Pressure'][0],
                                                            count=self.address_list['Pressure'][1], unit=1)
              print(response)
              decoder = BinaryPayloadDecoder.fromRegisters(response.registers, byteorder="big", wordorder="big")
              self.P0 = decoder.decode_32bit_float()

      def get_data(self):
          data = []
          if self.svp:
              for i in self.svp:
                  response = self.client.read_holding_registers(address=self.address_list[i][0],
                                                                count=self.address_list[i][1])
                  decoder = BinaryPayloadDecoder.fromRegisters(response.registers, byteorder="big", wordorder="big")
                  value = decoder.decode_32bit_float()

                  if i == 'Pressure' and self.press == 'relative':
                      data.append(value - self.P0)
                  else:
                      data.append(value)

          return [time.time()] + data

      def set_cmd(self, cmd):
          if cmd is None:
              pass
          else:
              builder = BinaryPayloadBuilder(byteorder="big", wordorder="big")
              builder.add_32bit_float(cmd)
              payload = builder.build()
              self.client.write_registers(address=1009, values=payload)

      def close(self):
          builder = BinaryPayloadBuilder(byteorder="big", wordorder="big")
          builder.add_32bit_float(0)
          payload = builder.build()
          self.client.write_registers(address=1009, values=payload)
          self.client.close()


if pymodbus.__version__=='3.9.0':


  class Flow_controller_alicat(crappy.inout.InOut):

    def __init__(self,
                 port="/dev/ttyUSB0",
                 startbit=1,
                 databits=8,
                 parity="N",
                 stopbits=1,
                 errorcheck="crc",
                 baudrate=115200,
                 timeout=3,
                 press='absolue',
                 svp=[]):

        crappy.inout.InOut.__init__(self)
        self.port = port
        self.startbit = startbit
        self.databits = databits
        self.parity = parity
        self.stopbits = stopbits
        self.errorcheck = errorcheck
        self.baudrate = baudrate
        self.timeout = timeout
        self.press = press
        self.svp = svp
        self.address_list = {
            "Pressure": [1202, 2],
            "Temperature": [1204, 2],
            "Volumetric_flow": [1206, 2],
            "Mass_flow": [1208, 2]
        }

    def open(self):
        self.client = ModbusSerialClient(
            port=self.port,
            baudrate=self.baudrate,
            stopbits=self.stopbits,
            parity=self.parity,
            timeout=self.timeout
        )

        if not self.client.connect():
            raise Exception(f"Unable to connect to the Modbus device at {self.port}")

        if self.press == 'relative':
            response = self.client.read_holding_registers(
                address=self.address_list['Pressure'][0],
                count=self.address_list['Pressure'][1],
                unit=1
            )
            if not response.isError():
                self.P0 = self._decode_float(response.registers)
            else:
                raise Exception("Erreur lors de la lecture de la pression initiale")

    def get_data(self):
        data = []
        if self.svp:
            for i in self.svp:
                response = self.client.read_holding_registers(
                    address=self.address_list[i][0],
                    count=self.address_list[i][1])
                if not response.isError():
                    value = self._decode_float(response.registers)
                    if i == 'Pressure' and self.press == 'relative':
                        data.append(value - self.P0)
                    else:
                        data.append(value)
                else:
                    data.append(None)  # En cas d'erreur
        return [time.time()] + data

    def set_cmd(self, cmd):
        if cmd is not None:
            registers = self._encode_float(cmd)
            self.client.write_registers(address=1009, values=registers
                                        )

    def close(self):
        zero_cmd = 0.0
        registers = self._encode_float(zero_cmd)
        self.client.write_registers(address=1009, values=registers
                                    )
        self.client.close()

    def _decode_float(self, registers, byte_order='big', word_order='big'):
      # Changer l'ordre des mots (registres) si nécessaire
      if word_order == 'little':
          registers = registers[::-1]

      # Convertir les registres en bytes
      byte_data = b''.join(struct.pack('>H', reg) for reg in registers)

      # Changer l'ordre des bytes si nécessaire
      if byte_order == 'little':
          byte_data = byte_data[::-1]

      # Décode les bytes en float
      return struct.unpack('>f', byte_data)[0]


    def _encode_float(self, value):
        """Convertit un float en une liste de registres en gérant l'endianness."""
        # Encode le float en 4 octets (format IEEE 754)
        byte_data = struct.pack('>f', value)
        # Divise les octets en 2 registres de 16 bits
        return [struct.unpack('>H', byte_data[i:i+2])[0] for i in range(0, 4, 2)]



if __name__ == "__main__":

    date='2025_05_15'
    sample = 'SC37_40_A1L'

    if os.path.exists(saving_folder) :
        ()
    else :
        P = pathlib.Path(saving_folder)
        pathlib.Path.mkdir(P, parents = True)


    flow_ali = crappy.blocks.IOBlock('Flow_controller_alicat',
                                    port="/dev/ttyUSB1",
                                    startbit=1,
                                    databits=8,
                                    parity="N",
                                    stopbits=1,
                                    errorcheck="crc",
                                    baudrate=9600,
                                    #method="RTU",
                                    timeout=30,
                                    svp=['Pressure', 'Volumetric_flow','Mass_flow'],
                                    cmd_labels=['cmd'],
                                    labels=['t(s)', 'press','volumetric_flow','mass_flow'])

    ver = crappy.blocks.VideoExtenso(camera='XiAPI',
                                    config=True,
                                    save_images=True,
                                    save_folder=f'./{date}/{sample}/video_extenso_right/',
                                    img_shape=(2048,2048),
                                    img_dtype='uint8',
                                    labels=['tr(s)', 'meta_r', 'pix_r', 'eyy_r', 'exx_r'],
                                    white_spots=False,
                                    **{"serial_number": "14482450",
                                        "exposure_time_us": 56229,
                                        "trigger": "Hdw after config",
                                        'timeout':100000})

    vel = crappy.blocks.VideoExtenso(camera='XiAPI',
                                    config=True,
                                    save_images=True,
                                    save_folder=f'./{date}/{sample}/video_extenso_left/',
                                    img_shape=(2048,2048),
                                    img_dtype='uint8',
                                    labels=['tl(s)', 'meta_l', 'pix_l', 'eyy_l', 'exx_l'],
                                    white_spots=False,
                                    **{"serial_number": "32482550",
                                        "exposure_time_us": 50000,
                                        "trigger": "Hdw after config",
                                        'timeout':100000})



    ftdi = crappy.blocks.IOBlock('Ft232r', cmd_labels=['cmd'], spam=False, direction=0b00000100, URL='ftdi://ftdi:232:FTU7DIHC/1')

    gen_ft = crappy.blocks.Generator([{'type': 'Cyclic',
                                    'value1': 0, 'condition1': 'delay=0.1',
                                    'value2': 1, 'condition2': 'delay=1', 'cycles': 150}], cmd_label='cmd')



    gen_flow = crappy.blocks.Generator([{'type': 'Constant',
                                        'value': 0.1,
                                        'condition': 'delay=150'}], cmd_label='cmd')

    rec_ali = crappy.blocks.Recorder(file_name=f'./{date}/{sample}/data_ali.txt')

    rec_ver = crappy.blocks.Recorder(file_name=f'./{date}/{sample}/data_ver.txt', labels=['tr(s)','pix_r','exx_r','eyy_r'])

    rec_vel = crappy.blocks.Recorder(file_name=f'./{date}/{sample}/data_vel.txt', labels=['tl(s)','pix_l','exx_l','eyy_l'])

    crappy.link(gen_ft, ftdi)
    crappy.link(gen_flow, flow_ali)


    crappy.link(flow_ali, rec_ali)
    crappy.link(ver, rec_ver)
    crappy.link(vel, rec_vel)

    crappy.start()
