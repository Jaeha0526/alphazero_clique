#!/usr/bin/env python
import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import collections
import math
import encoder_decoder as ed
from chess_board import board as c_board
import copy
import torch.multiprocessing as mp
from alpha_net import ChessNet
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random
import time

@dataclass
class UCTNode:
    game: c_board
    move: Optional[int]
    parent: Optional['UCTNode']
    is_expanded: bool = False
    children: Dict[int, 'UCTNode'] = None
    child_priors: np.ndarray = None
    child_total_value: np.ndarray = None
    child_number_visits: np.ndarray = None
    action_idxes: List[int] = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.child_priors is None:
            self.child_priors = np.zeros([4672], dtype=np.float32)
        if self.child_total_value is None:
            self.child_total_value = np.zeros([4672], dtype=np.float32)
        if self.child_number_visits is None:
            self.child_number_visits = np.zeros([4672], dtype=np.float32)
        if self.action_idxes is None:
            self.action_idxes = []
    
    @property
    def number_visits(self) -> float:
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value: float) -> None:
        self.parent.child_number_visits[self.move] = value
    
    @property
    def total_value(self) -> float:
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value: float) -> None:
        self.parent.child_total_value[self.move] = value
    
    def child_Q(self) -> np.ndarray:
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self) -> np.ndarray:
        return math.sqrt(self.number_visits) * (
            abs(self.child_priors) / (1 + self.child_number_visits))
    
    def best_child(self) -> int:
        if self.action_idxes:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove
    
    def select_leaf(self) -> 'UCTNode':
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current
    
    def add_dirichlet_noise(self, action_idxs: List[int], child_priors: np.ndarray) -> np.ndarray:
        valid_child_priors = child_priors[action_idxs]
        valid_child_priors = 0.75*valid_child_priors + 0.25*np.random.dirichlet(
            np.zeros([len(valid_child_priors)], dtype=np.float32)+0.3)
        child_priors[action_idxs] = valid_child_priors
        return child_priors
    
    def expand(self, child_priors: np.ndarray) -> None:
        self.is_expanded = True
        action_idxs = []
        c_p = child_priors.copy()
        
        for action in self.game.actions():
            if action:
                initial_pos, final_pos, underpromote = action
                action_idxs.append(ed.encode_action(self.game, initial_pos, final_pos, underpromote))
                
        if not action_idxs:
            self.is_expanded = False
            return
            
        self.action_idxes = action_idxs
        c_p[~np.isin(np.arange(len(c_p)), action_idxs)] = 0.0
        
        if self.parent.parent is None:
            c_p = self.add_dirichlet_noise(action_idxs, c_p)
            
        self.child_priors = c_p
    
    def decode_n_move_pieces(self, board: c_board, move: int) -> c_board:
        i_pos, f_pos, prom = ed.decode_action(board, move)
        for i, f, p in zip(i_pos, f_pos, prom):
            board.player = self.game.player
            board.move_piece(i, f, p)
            a, b = i
            c, d = f
            
            if board.current_board[c,d] in ["K","k"] and abs(d-b) == 2:
                if a == 7 and d-b > 0:  # castle kingside for white
                    board.player = self.game.player
                    board.move_piece((7,7), (7,5), None)
                if a == 7 and d-b < 0:  # castle queenside for white
                    board.player = self.game.player
                    board.move_piece((7,0), (7,3), None)
                if a == 0 and d-b > 0:  # castle kingside for black
                    board.player = self.game.player
                    board.move_piece((0,7), (0,5), None)
                if a == 0 and d-b < 0:  # castle queenside for black
                    board.player = self.game.player
                    board.move_piece((0,0), (0,3), None)
        return board
            
    def maybe_add_child(self, move: int) -> 'UCTNode':
        if move not in self.children:
            copy_board = copy.deepcopy(self.game)
            copy_board = self.decode_n_move_pieces(copy_board, move)
            self.children[move] = UCTNode(
                copy_board, move, parent=self)
        return self.children[move]
    
    def backup(self, value_estimate: float) -> None:
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player == 1:
                current.total_value += value_estimate
            else:
                current.total_value += -value_estimate
            current = current.parent

class DummyNode:
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)

def UCT_search(game_state: c_board, num_reads: int, net: nn.Module) -> Tuple[int, UCTNode]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = UCTNode(game_state, move=None, parent=DummyNode())
    
    for _ in range(num_reads):
        leaf = root.select_leaf()
        encoded_s = ed.encode_board(leaf.game)
        encoded_s = encoded_s.transpose(2,0,1)
        encoded_s = torch.from_numpy(encoded_s).float().to(device)
        
        with torch.no_grad():
            child_priors, value_estimate = net(encoded_s)
            child_priors = child_priors.cpu().numpy().reshape(-1)
            value_estimate = value_estimate.item()
            
        if leaf.game.check_status() and not leaf.game.in_check_possible_moves():
            leaf.backup(value_estimate)
            continue
            
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
        
    return np.argmax(root.child_number_visits), root

def do_decode_n_move_pieces(board: c_board, move: int) -> c_board:
    i_pos, f_pos, prom = ed.decode_action(board, move)
    for i, f, p in zip(i_pos, f_pos, prom):
        board.move_piece(i, f, p)
        a, b = i
        c, d = f
        
        if board.current_board[c,d] in ["K","k"] and abs(d-b) == 2:
            if a == 7 and d-b > 0:  # castle kingside for white
                board.player = 0
                board.move_piece((7,7), (7,5), None)
            if a == 7 and d-b < 0:  # castle queenside for white
                board.player = 0
                board.move_piece((7,0), (7,3), None)
            if a == 0 and d-b > 0:  # castle kingside for black
                board.player = 1
                board.move_piece((0,7), (0,5), None)
            if a == 0 and d-b < 0:  # castle queenside for black
                board.player = 1
                board.move_piece((0,0), (0,3), None)
    return board

def get_policy(root: UCTNode) -> np.ndarray:
    policy = np.zeros([4672], dtype=np.float32)
    valid_indices = np.where(root.child_number_visits != 0)[0]
    if len(valid_indices) > 0:
        policy[valid_indices] = root.child_number_visits[valid_indices] / root.child_number_visits.sum()
    return policy

def save_as_pickle(filename: str, data: List) -> None:
    complete_name = os.path.join("./datasets/iter2/", filename)
    with open(complete_name, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(filename: str) -> List:
    complete_name = os.path.join("./datasets/", filename)
    with open(complete_name, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def MCTS_self_play(chessnet: nn.Module, num_games: int, cpu: int) -> None:
    for game_idx in range(num_games):
        current_board = c_board()
        checkmate = False
        dataset = []
        states = []
        value = 0
        
        while not checkmate and current_board.move_count <= 100:
            # Check for draw by repetition
            draw_counter = sum(1 for s in states if np.array_equal(current_board.current_board, s))
            if draw_counter >= 3:
                break
                
            states.append(copy.deepcopy(current_board.current_board))
            board_state = copy.deepcopy(ed.encode_board(current_board))
            
            best_move, root = UCT_search(current_board, 777, chessnet)
            current_board = do_decode_n_move_pieces(current_board, best_move)
            
            policy = get_policy(root)
            dataset.append([board_state, policy])
            
            print(f"Move {current_board.move_count}:")
            print(current_board.current_board)
            print()
            
            if current_board.check_status() and not current_board.in_check_possible_moves():
                value = -1 if current_board.player == 0 else 1
                checkmate = True
                
        # Prepare dataset with values
        dataset_with_values = []
        for idx, (s, p) in enumerate(dataset):
            dataset_with_values.append([s, p, 0 if idx == 0 else value])
            
        # Save dataset
        save_as_pickle(
            f"dataset_cpu{cpu}_{game_idx}_{datetime.datetime.today().strftime('%Y-%m-%d')}",
            dataset_with_values
        )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_to_play = "current_net_trained8_iter1.pth.tar"
    
    mp.set_start_method("spawn", force=True)
    net = ChessNet()
    net.to(device)
    net.share_memory()
    net.eval()
    
    current_net_filename = os.path.join("./model_data/", net_to_play)
    checkpoint = torch.load(current_net_filename, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    
    processes = []
    for i in range(6):
        p = mp.Process(target=MCTS_self_play, args=(net, 50, i))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
