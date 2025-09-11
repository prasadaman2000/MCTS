"""
Network player will work the following way:

endpoints:
/start?opponent_type=<type>&player_id=<player_number>
 - this will create a session against a particular opponent type
 - will return a session ID associated with the game with player as player_id

/play?session_id=<session_id>&column=<col>
 - this will add the played move to a played move queue for the player
 - will return the next state

/quit?session_id=<session_id>
 - will destroy the session at session_id
"""

import bottle
from bottle import route, request, response
import uuid
import game
import player
import threading
import time
import mcts_players
import random
import json


class Session:
    def __init__(self, session_id: str, game: game.ConnectFour, player_id):
        self.session_id = session_id
        self.pending_move = None
        self.game = game
        self.player_id = player_id
        self.mu = threading.Lock()
        self.cond = threading.Condition()

    def set_action(self, action: int):
        print(f"setting action to {action}")
        with self.cond:
            with self.mu:
                self.pending_move = action
                print(f"set pending move to {self.pending_move}")
                self.cond.notify()

session_id_to_session = dict[str, Session]()

player1_mcts = mcts_players.SmarterMCTSTrainer(2, r=0.1, s=1000)
player2_mcts = mcts_players.SmarterMCTSTrainer(2, r=0.1, s=1000)

class NetworkPlayer(player.Player):
    def __init__(self):
        self.session = None

    def set_session(self, session: Session):
        self.session = session

    def get_pending_move(self):
        with self.session.mu:
            return self.session.pending_move
        
    def reset_pending_move(self):
        with self.session.mu:
            self.session.pending_move = None

    def get_move(self, state: "game.GameState", **kwargs):
        del kwargs
        with self.session.cond:
            print(f"waiting for network player in session {self.session.session_id}")
            self.session.cond.wait(timeout=30)
            if self.get_pending_move() is not None:
                move = self.get_pending_move()
                print(f"got human move {move}")
                self.reset_pending_move()
            else:
                print("Timed out, choosing random action")
                move = random.choice(state.get_valid_actions())
            return move


@route("/")
def hello():
    response.headers['Access-Control-Allow-Origin'] = '*'
    return "hi" 

@route("/getactions")
def get_actions():
    response.headers['Access-Control-Allow-Origin'] = '*'
    session_id = request.query.session
    if session_id not in session_id_to_session.keys():
        return json.dumps({"error": "bad session id"})
    
    session = session_id_to_session[session_id]

    if not session.player_id == session.game.get_player_turn():
        return json.dumps([])
    else:
        game_state = session.game.get_cur_state()
        valid_actions = game_state.get_valid_actions()
        return json.dumps({"actions": valid_actions})
    
@route("/playaction")
def playaction():
    response.headers['Access-Control-Allow-Origin'] = '*'
    session_id = request.query.session
    if session_id not in session_id_to_session.keys():
        return json.dumps({"error": "bad session id"})
    
    session = session_id_to_session[session_id]

    try:
        action = int(request.query.action)
    except:
        response.status = 400
        return json.dumps({"error": "Action should be an integer."})
    
    valid_actions = session.game.get_cur_state().get_valid_actions()
    if action not in valid_actions:
        response.status = 400
        return {"error": f"Action needes to be in {valid_actions}"}
    
    session.set_action(action)
    return json.dumps({"success": True})

@route("/getstate")
def getstate():
    response.headers['Access-Control-Allow-Origin'] = '*'
    session_id = request.query.session
    if session_id not in session_id_to_session.keys():
        return json.dumps({"error": "bad session id"})

    session = session_id_to_session[session_id]
    winner = session.game.get_winner()
    return {"board": str(session.game.get_cur_state().board),
            "turn": session.game.get_player_turn(),
            "winner": str(winner) if winner else -1}

@route("/start")
def start():
    response.headers['Access-Control-Allow-Origin'] = '*'
    # response.set_header("Access-Control-Allow-Methods:", "GET")
    try:
        player_id = int(request.query.id)
    except Exception as e:
        response.status = 400
        return f"Bad player id {request.query.id}. Must be 1 or 2."

    session_id = str(uuid.uuid1()).replace("-","")

    if player_id == 1:
        player1 = NetworkPlayer()
        player2 = player2_mcts
    if player_id == 2:
        player2 = NetworkPlayer()
        player1 = player1_mcts

    g = game.ConnectFour((7,6), player1, player2)

    session = Session(session_id, g, player_id)
    session_id_to_session[session_id] = session

    if player_id == 1:
        player1.set_session(session)
    if player_id == 2:
        player2.set_session(session)

    threading.Thread(target=g.play, kwargs={"eval": False}, daemon=True).start()
    return session_id
    

bottle.run(host="0.0.0.0", port=8080, debug=True, reloader=True)
