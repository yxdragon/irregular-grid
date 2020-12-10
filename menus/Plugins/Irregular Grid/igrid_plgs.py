from sciapp.action import Free, Simple
from sciapp.object import Image
from ....igrid.smc_grid_jit import build
from ....igrid.util import render
import numpy as np
import pandas as pd

class Data(Free):
	title = 'Grid Test Data'
	para = {'max':20, 'binary':False}
	view = [(int, 'max', (1, 32), 0, 'max', 'value'),
			(bool, 'binary', 'binary mask')]

	def run(self, para = None):
		x, y = np.ogrid[-2:2:160j, -2:2:160j]
		z = abs(x) * np.exp(-x ** 2 - (y / .75) ** 2)
		z = (para['max'] - z/z.max()*para['max']).astype(np.int32)
		if para['binary']: z[z>0] = para['max']
		self.app.show_img([z], 'grid')

class SMC(Simple):
	title = 'SMC Grid'
	note = ['8-bit', '16-bit', 'int']
	para = {'level':5, 'continue':True, 'table':True, 'render':True}

	view = [(int, 'level', (1, 256), 0, 'level', '2^n'),
			(bool, 'continue', 'can not jump level'),
			(bool, 'render', 'render each cell'),
			(bool, 'table', 'show location and relation table')]

	def run(self, ips, imgs, para = None):
		grid = ips.img.astype(np.int32)
		level = [2**i for i in range(para['level'])]
		grid, locs, rels = build(grid, level, para['continue'])
		self.app.show_img([grid], title=ips.title+'-grid')
		
		if para['render']:
			print(render(grid, locs, rels).shape)
			ips = Image([render(grid, locs, rels)], name=ips.title+'-render')
			ips.rg = [(0, 8)]
			self.app.show_img(ips)
			
		if para['table']:
			locs = pd.DataFrame(locs, columns=['row', 'col', 'layer'])
			self.app.show_table(locs, title=ips.title+'-locs')
			rels = pd.DataFrame(rels, columns=['from', 'to'])
			self.app.show_table(rels, title=ips.title+'-rels')

		'''
		grid = render(grid, locs, rels)
		plot_grid(grid, locs, rels, False)


		grid, locs, rels = build(z, [1,2,4,8,16], True)
		grid = render(grid, locs, rels)
		plot_grid(grid, locs, rels, False)

		if not para['slice']: imgs = [ips.img]
		shift = fftshift if para['shift'] else lambda x:x
		rst = []
		for i in range(len(imgs)):
			rst.append(shift(fft2(imgs[i])))
			self.progress(i, len(imgs))
		ips = Image(rst, '%s-fft'%ips.title)
		ips.log = True
		self.app.show_img(ips)
		'''

plgs = [Data, SMC]