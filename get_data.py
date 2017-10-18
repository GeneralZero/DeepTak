import sqlite3, os.path, requests, datetime, re, pickle, traceback
import h5py
import numpy as np
from board import TakBoard
import zipfile


class PlayTak(object):
	"""Utility to get playtak.com data"""
	def __init__(self):
		if not os.path.isdir(os.path.join(os.getcwd(), "ptn")):
			os.mkdir(os.path.join(os.getcwd(), "ptn"))
		if not os.path.isfile(os.path.join(os.getcwd(), "ptn", "games_anon.db")):
			self.download_sqllite()
		self.connection = sqlite3.connect(os.path.join(os.getcwd(), "ptn", "games_anon.db"))
		self.cursor = self.connection.cursor()
		#White Win
		#self.cursor.execute("""SELECT * FROM games WHERE games.result == "R-0" and games.size = 5 """)

		#Black Win
		self.cursor.execute("""SELECT * FROM games WHERE games.result == "0-R" and games.size = 5 """)
		self.notation_array = self.cursor.fetchall() 

		#print(len(self.notation_array))

		#for x in self.notation_array:
		#	self.sql_to_ptn(x)

	def sql_to_ptn(self, sqlentry, transformation):
		return self.server_to_ptn(
			{"id": sqlentry[0], 
			"date": sqlentry[1], 
			"size":sqlentry[2], 
			"player_white": sqlentry[3], 
			"player_black": sqlentry[4], 
			"moves": self.parse_server_to_dict(sqlentry[5]), 
			"result": sqlentry[6]}, transformation)

	def sql_to_numpy(self, sqlentry):
		ret = []
		moves = self.parse_server_to_dict(sqlentry[5])

		game = TakBoard(5)

		##Get Blank Board
		ret.append(game.get_numpy_board())

		#Start Game
		is_white = False

		for index, move in enumerate(moves):
			if index == 2:
				is_white = False

			#print(move)

			#Move Board
			try:
				if move["movetype"] == 'p':
					game.place(move["piece"], move["placement"])
				elif move["movetype"] == 'm':
					game.move(move["start"], move["end"], move["order"])
				else:
					raise ValueError("Invalid Movetype")
			except Exception as e:
				print(e)
				#traceback.print_exc()
				#exit()
				return None

			#Update
			is_white = not is_white

			#Get updated board
			#print(game.get_current_string_board())
			ret.append(game.get_numpy_board())

		#if len(ret) % 2 == 1:
		#	print("Odd game {}".format(sqlentry[0]))
		return ret

	def download_sqllite(self):
		r = requests.get("https://www.playtak.com/games_anon.db", stream=True)
		
		if r.status_code == 200:
			with open(os.path.join(os.getcwd(), "ptn", "games_anon.db"), 'wb') as f:
				f.write(r.content)	
	
	def get_all_games_h5(self):
		for transformation in [7]:
			with h5py.File(os.path.join(os.getcwd(), "ptn", "Black_Win_size_5_rot_{}.h5".format(transformation)), "w") as hf:
				for index, game in enumerate(self.notation_array):
					all_boards = self.sql_to_numpy(game)
					if type(all_boards) == list:
						#print(len(all_boards), type(all_boards))
						print("Write file gamedata_{} for Transformation {}".format(index, transformation))
						hf.create_dataset("gamedata_{}".format(index), data=all_boards, compression="gzip", compression_opts=9)
					else:
						pass
						#print("Error with gamedata_{}.pickle for Transformation {}".format(index, transformation))
						#print("Null Error")

	def get_all_games_ptn(self):
		for transformation in [0,1,2,3,4,5,6,7]:
			with zipfile.ZipFile(os.path.join(os.getcwd(), "ptn", "White_Win_size_5_rot_{}.zip".format(transformation)), "w") as newZip:
				for index, game in enumerate(self.notation_array):
					ptn_board_string = self.sql_to_ptn(game, transformation)
					print("Write file gamedata_{}.ptn for Transformation {}".format(index, transformation))

					newZip.writestr("gamedata_{}.ptn".format(index), ptn_board_string)


	def server_to_ptn(self, game, transformation=0):
		ptn_out = ""

		ptn_out += "[Event \"{}\"]\n".format("PlayTak.com")
		ptn_out += "[Date \"{}\"]\n".format(datetime.datetime.fromtimestamp(game["date"]/1000).strftime('%Y.%m.%d'))
		ptn_out += "[Time \"{}\"]\n".format(datetime.datetime.fromtimestamp(game["date"]/1000).strftime('%H:%M:%S'))
		ptn_out += "[Player1 \"{}\"]\n".format(game["player_white"])
		ptn_out += "[Player2 \"{}\"]\n".format(game["player_black"])
		ptn_out += "[Size \"{}\"]\n".format(game["size"])
		ptn_out += "[Transformation \"{}\"]\n".format(transformation)
		ptn_out += "[Result \"{}\"]\n".format(game["result"])
		ptn_out += "\n"

		i = 0
		index = 1
		while i < len(game["moves"]):
			#print(game["moves"][i])
			current_move = self.transform_move(game["moves"][i], transformation, game["size"])

			out1 = self.output_to_ptn(current_move, game["size"])
			out2 = ""
			i += 1
			if i < len(game["moves"]):
				current_move = self.transform_move(game["moves"][i], transformation, game["size"])
				out2 = self.output_to_ptn(current_move, game["size"])

			ptn_out += "{}. {} {}\n".format(index, out1, out2)
			index += 1 
			i += 1
		return ptn_out

	def parse_server_to_dict(self, moves):
		player_moves = []
		for move in moves.split(","):
			if move == None or move == "":
				return ""
			#print move
			if move[0].lower() == "p":
				# Place a piece
				# Ex P E3 W
				m = re.search("([a-zA-Z][0-9]+)\s?([WSC])?", move[2:])
				placement = m.group(1)
				piece  = m.group(2)
				if type(piece) == str:
					piece = piece.replace("W", "S")
				player_moves.append({"movetype": move[0].lower(), "placement": placement, "piece": piece})
			elif move[0].lower() == "m":
				# Move a piece
				# M FROM TO #PLACE+
				#print move[2:]
				m = re.match("([a-zA-Z][0-9]+)\s([a-zA-Z][0-9]+)\s([0-9\s]+)", move[2:])
				start = m.group(1)
				end = m.group(2)
				#print m.group(3)
				countarray = [int(x) for x in m.group(3).replace(" ", "")]
				player_moves.append({"movetype": move[0].lower(), "start": start, "end": end, "order": countarray})
			else:
				#Parcing Error
				raise ValueError('Error Parcing move: \"' + move + "\"")
		return player_moves	

	def transform_move(self, move, transformation, size):
		#transformation can be 1=90,2=180,3=270
		if move["movetype"].lower() == "p":
			test = self.transform_pos(move["placement"], transformation, size)
			#print("Rotating from {} to {}".format(move["placement"], test))
			move["placement"] = test
			return move
		elif move["movetype"].lower() == "m":
			#print(move)
			move["start"] = self.transform_pos(move["start"], transformation, size)
			move["end"] = self.transform_pos(move["end"], transformation, size)
			return move
		else:
			#Parcing Error
			raise ValueError('Error Parcing move: \"' + move + "\"")

	def transform_pos(self, pos, transformation, size):
		ret =""

		#No Transformation 
		if transformation == 0:
			return pos

		#Flip Vertical
		elif transformation == 1:
			ret =  chr((ord("A") + size -1) - (ord(pos[0]) - ord("A") ))
			ret +=pos[1:]

		#Flip Horozontal and Vertical
		elif transformation == 2:
			#reflect Letter for 90 or 180
			ret =  chr((ord("A") + size -1) - (ord(pos[0]) - ord("A") ))
			ret += str( size - (int(pos[1:]) - 1 ))
		
		#Flip Horozontal
		elif transformation == 3:
			ret = pos[0]
			ret += str( size - (int(pos[1:]) - 1 ))

		#Rotate 90
		elif transformation == 4:
			ret = chr(int(pos[1:]) + ord("A") - 1)
			ret += str(size - (ord(pos[0]) - ord("A")))

		#Rotate 90 and Flip Vertical
		elif transformation == 5:
			ret = chr(int(pos[1:]) + ord("A") - 1)
			ret += str(size - (ord(pos[0]) - ord("A")))

			pos = ret

			ret =  chr((ord("A") + size -1) - (ord(pos[0]) - ord("A") ))
			ret +=pos[1:]

		#Rotate 90 and Flip Horozontal and Vertical
		elif transformation == 6:
			ret = chr(int(pos[1:]) + ord("A") - 1)
			ret += str(size - (ord(pos[0]) - ord("A")))

			pos = ret

			ret =  chr((ord("A") + size -1) - (ord(pos[0]) - ord("A") ))
			ret += str( size - (int(pos[1:]) - 1 ))

		#Rotate 90 and Flip Horozontal
		elif transformation == 7:
			ret = chr(int(pos[1:]) + ord("A") - 1)
			ret += str(size - (ord(pos[0]) - ord("A")))

			pos = ret

			ret = pos[0]
			ret += str( size - (int(pos[1:]) - 1 ))

		else:
			raise ValueError("Error Parsing transformation {}".format(transformation))

		return ret



	def output_to_ptn(self, move, size):
		output = {}

		if move["movetype"] == "p":
			if move["piece"] == None:
				return "{}".format(move["placement"].lower())
			else:
				return "{}{}".format(move["piece"].upper(), move["placement"].lower())

		elif move["movetype"] == "m":
			#order
			output["count"] = sum(move["order"])
			output["order"] = move["order"]
			#count info
			if len(move["order"]) == 1:
				output["order"] = []
				if move["order"] == 1:
					output["count"] = ""

			#start is done

			#direction lowest is bottom left
			start_int = size * int(move["start"][1:]) + (ord('a') - ord(move["start"][0].lower()))
			end_int = size * int(move["end"][1:]) + (ord('a') - ord(move["end"][0].lower())) 

			if start_int > end_int:
				# Move Down or Left
				if end_int > start_int - size:
					#Move Down
					output["direction"] = ">"
				else:
					#Move Left
					output["direction"] = "-"
			else:
				#Move Up or Right
				if end_int >= start_int + size:
					#Move Down
					output["direction"] = "+"
				else:
					#Move Left
					output["direction"] = "<"


			return "{}{}{}{}".format(output["count"], move["start"].lower(), output["direction"], "".join(str(x) for x in output["order"]))

		else:
			raise ValueError("Invalid Move Type")

	def analyseTak(self, move, white_turn, ptn_file):
		exe = "C:\\Users\\generalzero\\Documents\\Go\\bin\\analyzetak.exe"

		process = None
		
		if white_turn:
			process = subprocess.Popen([exe, "-move", move, "-white", ptn_file])
		else:
			process = subprocess.Popen([exe, "-move", move, "-white", ptn_file])

		print(process.communicate())


			
if __name__ == '__main__':
	#download/load file
	b = PlayTak()

	#Save Game data
	all_games = b.get_all_games_h5()

	#Save Game data to PTN
	#all_games = b.get_all_games_ptn()
