import numpy as np

class Datos(object):
	
	TiposDeAtributos=('Continuo','Nominal')
	numDatos = 0
	nombreAtributos = []
	tipoAtributos = []
	nominalAtributos = []
	diccionarios = []
	datos = None


	def reset(self):
		self.TiposDeAtributos=('Continuo','Nominal')
		self.numDatos = 0
		self.nombreAtributos = []
		self.tipoAtributos = []
		self.nominalAtributos = []
		self.diccionarios = []
		self.datos = None

	# TODO: procesar el fichero para asignar correctamente las variables tipoAtributos, nombreAtributos, nominalAtributos, datos y diccionarios
	def __init__(self, nombreFichero):
		lines = []
		self.reset()
		f = open(nombreFichero, "r")

		#Read number of data lines
		self.numDatos = int(f.readline())
		#Read attributes names
		self.nombreAtributos = f.readline().replace('\n', '').split(',')

		#Read attributes type
		self.tipoAtributos = f.readline().replace('\n', '').split(',')

		#Fill nominalAtributtos
		i = 0
		for atributo in self.tipoAtributos:
			if atributo == 'Nominal':
				self.nominalAtributos.insert(i, True)
			elif atributo == 'Continuo':
				self.nominalAtributos.insert(i, False)
			else:
				raise ValueError("Attribute type non existent")
			i += 1

		#Create dictionaries
		self.diccionarios = [{} for _ in range(len(self.nombreAtributos))]

		#Fill dictionaries
		line = f.readline()
		while line is not '':
			line = line.replace('\n', '').split(',')
			lines.append(line) #lines holds all the data to fill self.datos later
			for i in range(len(self.nombreAtributos)):
				#Check if attribute is Nominal
				if self.nominalAtributos[i]:
					self.diccionarios[i][line[i]] = 0
			line = f.readline()
		#Order dictionaries attributes values A to Z
		for i in range(len(self.nombreAtributos)):
			j = 0
			for key in sorted(self.diccionarios[i].keys()):
					self.diccionarios[i][key] = j
					j += 1

		#Fill datos
		self.datos = np.empty((self.numDatos, len(self.nominalAtributos)))
		for i in range(self.numDatos):
			for j in range(len(self.nombreAtributos)):
				if self.nominalAtributos[j]:
					#If the attribute is Nominal we use its dictionaty
					self.datos[i][j] = self.diccionarios[j][lines[i][j]]
				else:
					#If the atribute is Continuo we use its value
					self.datos[i][j] = lines[i][j]
		
		f.close()
		
	# TODO: implementar en la pr�ctica 1
	def extraeDatos(self,idx):
 
		return self.datos[idx]	



  