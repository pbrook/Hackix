import cwiid
import time



class Wiimote(object):
	
	
	def __init__(self):
		print 'place wiimote in discoverable mode (press 1 and 2)...'
		self._wiimote = cwiid.Wiimote()
		rpt_mode = cwiid.RPT_ACC
		rpt_mode ^= cwiid.RPT_EXT
		rpt_mode ^= cwiid.RPT_BTN
		self._wiimote.rpt_mode = rpt_mode
	
		led = 0
		led ^= cwiid.LED4_ON
		self._wiimote.led = led
	
		self.rotate = {'X': 0, 'Y': 0, 'Z': 0}
		self.translate = {'X': 0, 'Y': 0}
		#self.firstpass = True

	def movement(self):
		state = self._wiimote.state
		buttons = 0
		X = state['acc'][cwiid.X]
		if X <= 115:
			X = -1
		elif X > 135:
			X = 1
		else:
			X = 0
		self.rotate['X'] = X
		Y = state['acc'][cwiid.Y]
		if Y <= 115:
			Y = 1
		elif Y > 135:
			Y = -1
		else:
			Y = 0
		self.rotate['Y'] = Y
		buttons = state['buttons']
		if state['ext_type'] == cwiid.EXT_NUNCHUK:
			if state.has_key('nunchuk'):
				Z = state['nunchuk']['acc'][cwiid.X]
				if Z <= 115:
					Z = 1
				elif Z > 135:
					Z = -1
				else:
					Z = 0
				self.rotate['Z'] = Z
				self.translate['X'] = state['nunchuk']['stick'][0]
				self.translate['Y'] = state['nunchuk']['stick'][1]
				buttons += state['nunchuk']['buttons']
		
		#elif not self.firstpass:
		#	raise Exception('Nunchuk not there. OH NOES!')
		#self.firstpass = False

		#	Returns rotate X,Y,Z each a value between 0-50
		#	Returns translate X, Y each a value between 0-255
		#	If buttons > 0 a button has been pressed

		return self.rotate, self.translate, buttons

def main():
	wiimote = Wiimote()
	while True:
		print '%r %r %r' % wiimote.movement()
		time.sleep(1)
		
if __name__ == '__main__':
	main()
