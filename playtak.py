import sqlite3, os.path, requests

class PlayTak(object):
	"""Utility to get playtak.com data"""
	def __init(self):
		if !os.path.isfile("games_anon.db"):
			self.download_sqllite()
		self.connection = sqlite3.connect("games_anon.db")
		self.cursor = connection.cursor()

		self.cursor.execute("""SELECT notation FROM games WHERE games.result != "1-0" OR games.result != "0-1" """)
		self.notation_array = self.cursor.fetchall() 

	def download_sqllite(self):
		r = requests.get("https://www.playtak.com/games_anon.db")
		with open('games_anon.db', 'bw') as the_file:
			the_file.write(r.content)

	def parse_server_to_dict(self):
		player_moves = []
		for move in moves.split(","):
			if move[0].lower = "p":
				# Place a piece
				# Ex P E3 W
				placement, piece = re.match("([a-zA-z][0-9]+)\w?([WC])", move[2:])
				player_moves.append({"movetype": move[0], "placement": placement, "piece": piece})
			elif move[0].lower = "m":
				# Move a piece
				# M FROM TO #PLACE+
				start, end, countstring = re.match("([a-zA-z][0-9]+)\w([a-zA-z][0-9]+)\w([0-9\w]+)", move[2:])
				countarray = countstring.split()
				player_moves.append({"movetype": move[0], "start": start, "end": end, "order": countarray})
			else:
				#Parcing Error
				raise ValueError('Error Parcing move: ' + move)
		return player_moves	

	def download_database():
		pass

	def get_complete_games():
		pass

def valid_moves(board, player):
	all_moves = []
	#All of the places
	for index, open_squares in enumerate(board) if len(open_squares) == 0:
		if player.used_capstone():
			for types in ["F","W"]:
				all_moves.append({"movetype": move[0], "placement": placement, "piece": piece})
		else:
			for types in ["F","W", "C"]:
	#All of the types

	#All of the Stacks
	#All of the directions
	#All of the places
	return possoble_moves



