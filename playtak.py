import sqlite3, os.path, requests, datetime, re, pickle
import numpy as np
from board import TakBoard

class PlayTak(object):
	"""Utility to get playtak.com data"""
	def __init__(self):
		if not os.path.isfile("ptn\\games_anon.db"):
			self.download_sqllite()
		self.connection = sqlite3.connect("ptn\\games_anon.db")
		self.cursor = self.connection.cursor()

		self.cursor.execute("""SELECT * FROM games WHERE (games.result == "0-R" or games.result == "R-0") and games.size = 5 """)
		self.notation_array = self.cursor.fetchall() 

		#print(len(self.notation_array))

		#for x in self.notation_array:
		#	self.sql_to_ptn(x)

	def sql_to_ptn(self, sqlentry):
		return self.server_to_ptn({"date": sqlentry[1], "size":sqlentry[2], "player_white": sqlentry[3], "player_black": sqlentry[4], "moves": self.parse_server_to_dict(sqlentry[5]), "result": sqlentry[6]})

	def sql_to_numpy(self, sqlentry):
		ret = []
		moves = self.parse_server_to_dict(sqlentry[5])

		game = TakBoard(5)

		#Start Game
		is_white = False

		#Start at move 3 to simplify the placements
		for index, move in enumerate(moves):
			if index == 2:
				is_white = False

			#print(index,is_white)

			#Move Board
			try:
				if move["movetype"] == "p":
					game.place(move["piece"], move["placement"], is_white)
				elif move["movetype"] == "m":
					game.move(move["start"], move["end"], move["order"])
				else:
					raise ValueError("Invalid Movetype")
			except:
				return None

			#Update
			is_white = not is_white

			#Get updated board
			#print(np.array(game.get_current_board()))
			ret.append(np.array(game.get_current_board()))

		return ret

	def download_sqllite(self):
		r = requests.get("https://www.playtak.com/games_anon.db", stream=True)
		
		if r.status_code == 200:
			with open("ptn\\games_anon.db", 'wb') as f:
				f.write(r.content)	
	
	def get_all_games(self):
		for index, game in enumerate(self.notation_array):
			all_boards = self.sql_to_numpy(game)
			all_boards = np.array(all_boards)
			#print(len(all_boards), type(all_boards))
			if type(all_boards) == np.ndarray:
				print("Write file gamedata_{}".format(index))
				with open("ptn\\gamedata_{}".format(index), 'wb') as f:
					pickle.dump(all_boards, f)

		#Rotate Boards 90
		#with h5py.File('ptn\\gamedata_90.h5', 'w') as h5f:
		#	h5f.create_dataset('gamedata_90', data=all_boards)

		#Rotate Boards 180
		#with h5py.File('ptn\\gamedata_180.h5', 'w') as h5f:
		#	h5f.create_dataset('gamedata_180', data=all_boards)

		#Rotate Boards 270
		#with h5py.File('ptn\\gamedata_270.h5', 'w') as h5f:
		#	h5f.create_dataset('gamedata_270', data=all_boards)

	def server_to_ptn(self, game):
		ptn_out = ""

		ptn_out += "[Event \"{}\"]\n".format("PlayTak.com")
		ptn_out += "[Date \"{}\"]\n".format(datetime.datetime.fromtimestamp(game["date"]/1000).strftime('%Y-%m-%d'))
		ptn_out += "[Time \"{}\"]\n".format(datetime.datetime.fromtimestamp(game["date"]/1000).strftime('%H:%M:%S'))
		ptn_out += "[Player1 \"{}\"]\n".format(game["player_white"])
		ptn_out += "[Player2 \"{}\"]\n".format(game["player_black"])
		ptn_out += "[Size \"{}\"]\n".format(game["size"])
		ptn_out += "[Result \"{}\"]\n".format(game["result"])
		ptn_out += "\n"

		i = 0
		index = 1
		while i < len(game["moves"]):

			out1 = self.output_to_ptn(game["moves"][i], game["size"])
			out2 = ""
			i += 1
			if i < len(game["moves"]):
				out2 = self.output_to_ptn(game["moves"][i], game["size"])

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
				m = re.search("([a-zA-Z][0-9]+)\s?([WC])?", move[2:])
				placement = m.group(1)
				piece  = m.group(2)
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

	def rotate_moves(self, moves, angle, size):
		#angle can be 1=90,2=180,3=270
		player_moves = []
		for move in moves.split(","):
			if move == None or move == "":
				return ""
			#print move
			if move[0].lower() == "p":
				# Place a piece
				# Ex P E3 W
				m = re.search("([a-zA-Z][0-9]+)\s?([WC])?", move[2:])
				placement = m.group(1)
				piece  = m.group(2)
				player_moves.append({"movetype": move[0].lower(), "placement": self.rotate_pos(placement, angle, size), "piece": piece})
			elif move[0].lower() == "m":
				# Move a piece
				# M FROM TO #PLACE+
				#print move[2:]
				m = re.match("([a-zA-Z][0-9]+)\s([a-zA-Z][0-9]+)\s([0-9\s]+)", move[2:])
				start = m.group(1)
				end = m.group(2)
				#print m.group(3)
				countarray = [int(x) for x in m.group(3).replace(" ", "")]
				player_moves.append({"movetype": move[0].lower(), "start": self.rotate_pos(start, angle, size), "end": self.rotate_pos(end, angle, size), "order": countarray})
			else:
				#Parcing Error
				raise ValueError('Error Parcing move: \"' + move + "\"")
		return player_moves	

	def rotate_pos(self, pos, angle, size):
		ret =""
		if angle <= 2:
			#reflect Letter for 90 or 180
			ret =  chr((ord("A") + size -1) - (ord(pos[0]) - ord("A") ))
		else:
			ret = pos[0]

		if angle >= 2:
			#reflect number for 180 270
			ret += str( size - (int(pos[1:]) - 1 ))
		else:
			ret +=pos[1:]

		return ret


	def output_to_ptn(self, move, size):
		output = {}

		if move["movetype"] == "p":
			if move["piece"] == None:
				return "{}".format(move["placement"])
			else:
				return "{}{}".format(move["piece"].upper(), move["placement"])

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
			start_int = size * (ord('A') - ord(move["start"][0].upper())) + int(move["start"][1:])
			end_int = size * (ord('A') - ord(move["end"][0].upper())) + int(move["end"][1:])

			if start_int > end_int:
				# Move Down or Left
				if end_int > start_int - size:
					#Move Down
					output["direction"] = "<"
				else:
					#Move Left
					output["direction"] = "-"
			else:
				#Move Up or Right
				if end_int >= start_int + size:
					#Move Down
					output["direction"] = ">"
				else:
					#Move Left
					output["direction"] = "+"


			return "{}{}{}{}".format(output["count"], move["start"], output["direction"], "".join(str(x) for x in output["order"]))

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
	all_games = b.get_all_games()
