Deep Tak: A Tak AI
##################

Deep Tak is a Deep Learning AI that is made through games played on https://www.playtak.com/.
The goal is .... Deep Tak is a hobby project to learn deeplearning with an actual 
generally interacting with HTTP servers.


##Add Picture


Try to beat me on playTak.com 

* [What is Tak](#what-is-tak)
* [How it Works](#idea)
* [Ge](#features)

What is Tak?
=============
Tak is a simple board game simular to other games like Chess, Connect 4, and others. To win a game of TAK, a player must be the first to create a “road” of stones connecting opposite sides of the board.

Stones can be laid flat or stood on end. When played flat, they are called “flat stones.” In this orientation, other stones can be stacked on them. If they are stood on end, they are called “standing stones” or “walls.” Nothing can be stacked atop a standing stone, but these do not count as part of a player’s road.

Depending on the size of the game, players may also have capstones, which can can come in many decorative shapes. Capstones serve as both a flat stone and a wall, and can also flatten standing walls.[1](http://cheapass.com/tak/)

The full rules are available for *FREE* [here](http://cheapass.com/wp-content/uploads/2016/05/TakWebRules.pdf).

Overview of Deep Tak
===================
There are multable parts of this seperated in to diffrent files and sections below.

The first part of this is getting the training data. The data is retreved from [Play Tak](https://playtak.com). The website allows users to play online with other people or other AIs. The database containing the information to these games can be downloaded and used. The data is then split in to two groups the Where Player1 wins, and where Player2 wins.

The origional data is in its own format and needs to be convered to board states to be used in the Deep Learning network. The boards are split in to groups the board before the winner of the game plays and the move that the winner played. This is the input and the intended output of the Deep Learning network.

The final part of the project is to take any board state and to return the "best" move to the board.

Generating the Data
===================

Geting the Games
----------------
Taking the database game data from [Play Tak](https://www.playtak.com/games_anon.db) is easy because the developer gives you a link to the database. It current has over 100,000 Games. The bad part is that the database has its own version of how it stores the data. Time to make a parcer.

Converting the games to data
-------------------
The Move data for each game is not that hard. Below are some example of Moves in the server format.

`P D4`
`P C2 C`
`M A1 A2 1`
`M A1 C1 1 2`

These are pretty self explanatory

Place a Piece at space D4
Place a Capstone Piece at C2
Move 1 peice from A1 to A2
Move 3 peices from A1 to C1 in the order of 1 on B1 and 2 on C2.

Using these moves I generated a internal version of the board. Using this at each move generating a list of all of the board states of the current game is possoble.

Removing the Bad games
---------------------
When you are dealing with any large data there are going to be bad or incomplete data.

Some of the things that I found looking through the database include 
* Incomplete games
* Games with wins with exausting tiles
* Games where the opponent played the last move. (In chess terms the opponent moves in to checkmate.)

These games do not help learning about how to make winning moves and were discarded.

Spliting the Moves
------------------
Since the data needed to be sepperated into before and after the winning player played. Data is sepperated in to the two groups making sure that the for each pre-move there is a post-move.

Training the Model
==================

Converting the models
---------------------

Loading the models
------------------

Final Model
-----------

Predicting the game
==================

Examples
--------


