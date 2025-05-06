import crappy

gen = crappy.blocks.Generator([{'type': 'Constant',
                                'value': 30,
                                'condition' : 'delay=4'}], cmd_label='dccmd')

rasp = crappy.blocks.ClientServer(broker=False, address='localhost', port=1882, cmd_labels=[('dccmd',)])

crappy.link(gen, rasp)
crappy.start()
