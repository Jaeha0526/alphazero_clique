#!/usr/bin/env python3
"""
Web-based Game Trajectory Viewer using Plotly
Generates an interactive HTML file that can be viewed in any browser
"""

import pickle
import numpy as np
from pathlib import Path
import argparse
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx


class WebGameViewer:
    def __init__(self, filepath: str):
        """Initialize the viewer with a game data file."""
        self.filepath = filepath
        self.load_game_data()
    
    def load_game_data(self):
        """Load and parse the game data file."""
        print(f"Loading game data from: {self.filepath}")
        
        with open(self.filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.iteration = data.get('iteration', 'unknown')
        self.n = data['vertices']
        self.k = data['k']
        self.game_mode = data.get('game_mode', 'symmetric')
        self.games = data['sample_games']
        self.num_games = len(self.games)
        
        print(f"Loaded {self.num_games} games from iteration {self.iteration}")
        print(f"Graph: n={self.n}, k={self.k}, mode={self.game_mode}")
        
        # Precompute edge mappings
        self.edge_to_idx = {}
        self.idx_to_edge = {}
        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.edge_to_idx[(i, j)] = idx
                self.edge_to_idx[(j, i)] = idx
                self.idx_to_edge[idx] = (i, j)
                idx += 1
        
        self.num_edges = idx
        
        # Create graph layout
        G = nx.complete_graph(self.n)
        self.pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(self.n))
    
    def create_game_animation(self, game_idx: int):
        """Create an animated visualization of a single game."""
        game = self.games[game_idx]
        num_moves = len(game['moves'])
        
        # Create frames for animation
        frames = []
        
        for move_idx in range(num_moves):
            # Build edge colors up to this move
            edge_colors = {}
            for i in range(self.num_edges):
                edge_colors[i] = 'lightgray'
            
            for m_idx in range(move_idx + 1):
                move = game['moves'][m_idx]
                if move['top_actions']:
                    action = move['top_actions'][0][0]
                    player = move['player']
                    edge_colors[action] = 'red' if player == 0 else 'blue'
            
            # Create edge trace
            edge_trace = []
            for edge_idx, (i, j) in self.idx_to_edge.items():
                x0, y0 = self.pos[i]
                x1, y1 = self.pos[j]
                
                color = edge_colors[edge_idx]
                width = 3 if color != 'lightgray' else 1
                
                edge_trace.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=width, color=color),
                    hoverinfo='none',
                    showlegend=False
                ))
            
            # Highlight current move
            if move_idx < num_moves and game['moves'][move_idx]['top_actions']:
                current_action = game['moves'][move_idx]['top_actions'][0][0]
                i, j = self.idx_to_edge[current_action]
                x0, y0 = self.pos[i]
                x1, y1 = self.pos[j]
                
                edge_trace.append(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines',
                    line=dict(width=5, color='yellow'),
                    hoverinfo='none',
                    showlegend=False
                ))
            
            frames.append(go.Frame(data=edge_trace, name=str(move_idx)))
        
        # Create node trace
        node_trace = go.Scatter(
            x=[self.pos[i][0] for i in range(self.n)],
            y=[self.pos[i][1] for i in range(self.n)],
            mode='markers+text',
            text=[str(i) for i in range(self.n)],
            textposition="middle center",
            marker=dict(size=30, color='lightblue', line=dict(width=2, color='black')),
            hoverinfo='text',
            hovertext=[f'Node {i}' for i in range(self.n)],
            showlegend=False
        )
        
        # Create initial figure
        fig = go.Figure(
            data=frames[0].data + [node_trace],
            frames=frames
        )
        
        # Add slider and buttons
        fig.update_layout(
            title=f"Game {game_idx + 1}/{self.num_games} - n={self.n}, k={self.k}, mode={self.game_mode}",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 1000, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 0}}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}])
                    ],
                    x=0.1,
                    y=0
                )
            ],
            sliders=[dict(
                steps=[dict(args=[[str(k)], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0}}],
                           label=f"Move {k+1}",
                           method="animate")
                      for k in range(num_moves)],
                active=0,
                y=0,
                x=0.1,
                len=0.9
            )]
        )
        
        return fig
    
    def create_full_visualization(self, output_file: str = None):
        """Create a full HTML page with all games."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Game Viewer - Iteration {self.iteration}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .game-selector {{
            margin: 20px 0;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
        }}
        .game-info {{
            margin: 20px 0;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
        }}
        .move-info {{
            margin: 10px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-left: 3px solid #333;
        }}
        button {{
            padding: 10px 20px;
            margin: 5px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background-color: #45a049;
        }}
        #graph-container {{
            background-color: white;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AlphaZero Clique Game - Trajectory Viewer</h1>
        <p>Iteration: {self.iteration} | Graph: n={self.n}, k={self.k} | Mode: {self.game_mode}</p>
        <p>Total Games: {self.num_games}</p>
    </div>
    
    <div class="game-selector">
        <h2>Select Game:</h2>
        <select id="gameSelect" onchange="loadGame(this.value)">
"""
        
        # Add game options
        for i in range(self.num_games):
            game = self.games[i]
            html_content += f'            <option value="{i}">Game {i+1} ({len(game["moves"])} moves)</option>\n'
        
        html_content += """
        </select>
    </div>
    
    <div id="graph-container"></div>
    
    <div id="game-info" class="game-info"></div>
    
    <script>
        // Game data
        const gamesData = """
        
        # Convert games data to JSON
        games_json = []
        for game in self.games:
            game_data = {
                'num_moves': len(game['moves']),
                'moves': []
            }
            for move in game['moves']:
                move_data = {
                    'player': move['player'],
                    'value': move['value'],
                    'top_actions': move['top_actions'][:5] if move['top_actions'] else []
                }
                if 'player_role' in move:
                    move_data['player_role'] = move['player_role']
                game_data['moves'].append(move_data)
            games_json.append(game_data)
        
        html_content += json.dumps(games_json, indent=8)
        
        html_content += f""";
        
        // Edge mappings
        const idxToEdge = {json.dumps({str(k): v for k, v in self.idx_to_edge.items()})};
        const n = {self.n};
        const k = {self.k};
        
        let currentGame = 0;
        let currentMove = 0;
        
        function loadGame(gameIdx) {{
            currentGame = parseInt(gameIdx);
            currentMove = 0;
            updateDisplay();
        }}
        
        function updateDisplay() {{
            const game = gamesData[currentGame];
            
            // Update info panel
            let infoHtml = '<h3>Game ' + (currentGame + 1) + ' Information</h3>';
            infoHtml += '<p>Total Moves: ' + game.num_moves + '</p>';
            
            if (currentMove < game.moves.length) {{
                const move = game.moves[currentMove];
                infoHtml += '<div class="move-info">';
                infoHtml += '<h4>Move ' + (currentMove + 1) + '</h4>';
                infoHtml += '<p>Player: ' + (move.player + 1) + '</p>';
                if (move.player_role !== undefined) {{
                    const role = move.player_role === 0 ? 'Attacker' : 'Defender';
                    infoHtml += '<p>Role: ' + role + '</p>';
                }}
                infoHtml += '<p>Value: ' + move.value.toFixed(2) + '</p>';
                
                if (move.top_actions.length > 0) {{
                    infoHtml += '<p>Top Actions:</p><ul>';
                    for (let i = 0; i < move.top_actions.length; i++) {{
                        const action = move.top_actions[i];
                        const edge = idxToEdge[action[0].toString()];
                        infoHtml += '<li>Edge (' + edge[0] + ',' + edge[1] + '): ' + action[1].toFixed(3) + '</li>';
                    }}
                    infoHtml += '</ul>';
                }}
                infoHtml += '</div>';
            }}
            
            infoHtml += '<div style="margin-top: 20px;">';
            infoHtml += '<button onclick="prevMove()">Previous Move</button>';
            infoHtml += '<button onclick="nextMove()">Next Move</button>';
            infoHtml += '<button onclick="resetGame()">Reset</button>';
            infoHtml += '<button onclick="playGame()">Auto Play</button>';
            infoHtml += '</div>';
            
            document.getElementById('game-info').innerHTML = infoHtml;
            
            // Update graph visualization
            drawGraph();
        }}
        
        function drawGraph() {{
            // This would contain the Plotly graph drawing code
            // For simplicity, showing a placeholder
            const graphDiv = document.getElementById('graph-container');
            graphDiv.innerHTML = '<p>Move ' + (currentMove + 1) + ' of ' + gamesData[currentGame].num_moves + '</p>';
            graphDiv.innerHTML += '<p style="color: gray;">Graph visualization would appear here</p>';
        }}
        
        function nextMove() {{
            const game = gamesData[currentGame];
            if (currentMove < game.num_moves - 1) {{
                currentMove++;
                updateDisplay();
            }}
        }}
        
        function prevMove() {{
            if (currentMove > 0) {{
                currentMove--;
                updateDisplay();
            }}
        }}
        
        function resetGame() {{
            currentMove = 0;
            updateDisplay();
        }}
        
        let playing = false;
        let playInterval;
        
        function playGame() {{
            if (playing) {{
                clearInterval(playInterval);
                playing = false;
            }} else {{
                playing = true;
                playInterval = setInterval(() => {{
                    if (currentMove < gamesData[currentGame].num_moves - 1) {{
                        nextMove();
                    }} else {{
                        clearInterval(playInterval);
                        playing = false;
                    }}
                }}, 1000);
            }}
        }}
        
        // Initialize
        loadGame(0);
    </script>
</body>
</html>
"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)
            print(f"Created HTML visualization: {output_file}")
            print(f"Open this file in your web browser to view the games")
        
        return html_content


def main():
    parser = argparse.ArgumentParser(description='Web-based Game Trajectory Viewer')
    parser.add_argument('filepath', help='Path to game data pickle file')
    parser.add_argument('--output', '-o', default=None, help='Output HTML file (default: game_viewer.html)')
    parser.add_argument('--game', type=int, help='Specific game to visualize')
    
    args = parser.parse_args()
    
    if not Path(args.filepath).exists():
        print(f"Error: File {args.filepath} not found!")
        return
    
    viewer = WebGameViewer(args.filepath)
    
    output_file = args.output or 'game_viewer.html'
    viewer.create_full_visualization(output_file)


if __name__ == "__main__":
    main()