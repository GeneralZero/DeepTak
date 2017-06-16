import sqlite3, os.path, requests, datetime, re

class PlayTak(object):
	"""Utility to get playtak.com data"""
	def __init__(self):
		if not os.path.isfile("games_anon.db"):
			self.download_sqllite()
		self.connection = sqlite3.connect("games_anon.db")
		self.cursor = self.connection.cursor()

		self.cursor.execute("""SELECT * FROM games WHERE games.result != "1-0" OR games.result != "0-1" """)
		self.notation_array = self.cursor.fetchall() 

		for x in self.notation_array:
			self.sql_to_ptn(x)

	def sql_to_ptn(self, sqlentry):
		print self.server_to_ptn({"date": sqlentry[1], "size":sqlentry[2], "player_white": sqlentry[3], "player_black": sqlentry[4], "moves": self.parse_server_to_dict(sqlentry[5]), "result": sqlentry[6]})

	def download_sqllite(self):
		r = requests.get("https://www.playtak.com/games_anon.db", stream=True)
		
		if r.status_code == 200:
			with open('games_anon.db', 'wb') as f:
				f.write(r.content)

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

	def get_complete_games(self):
		pass

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

			
if __name__ == '__main__':
	b = PlayTak()
