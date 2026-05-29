# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import torch
import numpy as np
from net import PacmanNet
import os
from util import manhattanDistance
from game import Directions
import random, util

from config import PACMAN_SEED

random.seed(PACMAN_SEED)  # For reproducibility
from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, state: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = state.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(state, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, state: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # util.raiseNotDefined()

        def minimax(agentIndex: int, depth: int, gameState: GameState):
            """
            Recursive minimax function

            Args:
            agentIndex, which indicates if it's pacman (0) ir a ghost (>= 1)
            depth, which is the current depth of the game
            gameState, which is the current state of the game

            returns the best evaluation score for this agent (max if pacman, min if ghost)
            """

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                # if it's a win, a lose, or the maximum depth, we can't do anything else
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                # if the agent is pacman, we want to get the maximum value
                return maxValue(agentIndex, depth, gameState)

            else:
                # if the agent is a ghost, it watns to get the minimum value
                return minValue(agentIndex, depth, gameState)

        def maxValue(agentIndex: int, depth: int, gameState: GameState):
            """
            Function that calculates the maximum possible value

            Args:
            agentIndex, in which case should always be "0", indicating Pacman
            depth, it being the current depth in the game tree
            gameState, it being the current state of the game
            """
            best = float(
                "-inf"
            )  # we intially set the worst possible value as best to get the best one
            legal_actions = gameState.getLegalActions(agentIndex)

            if not legal_actions:
                return self.evaluationFunction(gameState)

            else:
                for action in legal_actions:  # we try each of the actions:
                    # we get the next state after said action and we call minimax until no legal_actions are found, or
                    # the game ends
                    successor = gameState.generateSuccessor(agentIndex, action)
                    best = max(
                        best, minimax(1, depth, successor)
                    )  # next it's the ghost's turn
                return best

        def minValue(agentIndex: int, depth: int, gameState: GameState):
            """
            Function that calculates de minimum possible value
            Works just like "maxValue" but with the ghosts' (enemies') perspective
            """
            worst = float("inf")
            legal_actions = gameState.getLegalActions(agentIndex)
            if not legal_actions:
                return self.evaluationFunction(gameState)

            else:
                # here we have to keep in mind the number of agents we have:
                # the next agent will be agentIndex + 1 UNLESS the current agent is the last one before Pacman's turn
                num_agents = gameState.getNumAgents()
                # next_agent, depth = agentIndex + 1, depth if agentIndex +1 < num_agents else 0, depth +1
                if agentIndex + 1 < num_agents:
                    next_agent = agentIndex + 1
                    next_depth = depth
                else:
                    next_agent = 0
                    next_depth = depth + 1

                for action in legal_actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    worst = min(worst, minimax(next_agent, next_depth, successor))
                return worst

        # time to chose the next action:
        best_action = None
        best_score = float("-inf")
        for action in state.getLegalActions(0):  # pacman starts
            successor = state.generateSuccessor(0, action)
            score = minimax(1, 0, successor)

            if score > best_score:
                best_score = score
                best_action = action
            # best_score, best_action = score, action if score > best_score else best_score, best_action

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, state: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        # We define variables
        alpha = float("-inf")
        beta = float("inf")
        best_action = None
        best_score = float("-inf")

        def alphabeta(
            agentIndex: int, depth: int, gameState: GameState, alpha: float, beta: float
        ):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # ============= PACMAN'S MAX ==============
            if agentIndex == 0:
                max_eval = float("-inf")
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    eval_score = alphabeta(1, depth, successor, alpha, beta)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)

                    if beta <= alpha:  # prune, or "poda" as we studied in class
                        break
                return max_eval

            # ============== GHOSTS' MIN ===============
            else:
                min_eval = float("inf")
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    if agentIndex + 1 < gameState.getNumAgents():
                        next_agent = agentIndex + 1
                        next_depth = depth
                    else:
                        next_agent = 0
                        next_depth = depth + 1
                    eval_score = alphabeta(
                        next_agent, next_depth, successor, alpha, beta
                    )
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)

                    if beta <= alpha:  # pruning
                        break
                return min_eval

        # ============ RUNNING ALPHA-BETA FOR ALL ACTIONS ===============
        legal_actions = state.getLegalActions(0)  # we start with pacman
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            score = alphabeta(1, 0, successor, alpha, beta)

            if score > best_score:
                best_score = score
                best_action = action
                alpha = max(alpha, score)

        return best_action


# From the original excercise, not specified in our project
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction


###########################################################################
# Ahmed
###########################################################################


class NeuralAgent(Agent):
    """
    Un agente de Pacman que utiliza una red neuronal para tomar decisiones
    basado en la evaluación del estado del juego.
    """

    def __init__(self, model_path="models/pacman_model.pth"):
        super().__init__()
        self.model = None
        self.input_size = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(model_path)

        # Mapeo de índices a acciones
        self.idx_to_action = {
            0: Directions.STOP,
            1: Directions.NORTH,
            2: Directions.SOUTH,
            3: Directions.EAST,
            4: Directions.WEST,
        }

        # Para evaluar alternativas
        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}

        # Contador de movimientos
        self.move_count = 0

        print(f"NeuralAgent inicializado, usando dispositivo: {self.device}")

    def load_model(self, model_path):
        """Carga el modelo desde el archivo guardado"""
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: No se encontró el modelo en {model_path}")
                return False

            # Cargar el modelo
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_size = checkpoint["input_size"]

            # Crear y cargar el modelo
            self.model = PacmanNet(self.input_size, 128, 5).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()  # Modo evaluación

            print(f"Modelo cargado correctamente desde {model_path}")
            print(f"Tamaño de entrada: {self.input_size}")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False

    def state_to_matrix(self, state):
        """Convierte el estado del juego en una matriz numérica normalizada"""
        # Obtener dimensiones del tablero
        walls = state.getWalls()
        width, height = walls.width, walls.height

        # Crear una matriz numérica
        # 0: pared, 1: espacio vacío, 2: comida, 3: cápsula, 4: fantasma, 5: Pacman
        numeric_map = np.zeros((width, height), dtype=np.float32)

        # Establecer espacios vacíos (todo lo que no es pared comienza como espacio vacío)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1

        # Agregar comida
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2

        # Agregar cápsulas
        for x, y in state.getCapsules():
            numeric_map[x][y] = 3

        # Agregar fantasmas
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(
                ghost_state.getPosition()[1]
            )
            # Si el fantasma está asustado, marcarlo diferente
            if ghost_state.scaredTimer > 0:
                numeric_map[ghost_x][ghost_y] = 6  # Fantasma asustado
            else:
                numeric_map[ghost_x][ghost_y] = 4  # Fantasma normal

        # Agregar Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        numeric_map[int(pacman_x)][int(pacman_y)] = 5

        # Normalizar
        numeric_map = numeric_map / 6.0

        return numeric_map

    def neural_eval(self, state) -> float:
        """
        Provides a valuation of a given pacman state using exclusively the NN's trained
        output.

        Parameters
        ----------
        state
            The board state to evaluate.

        Returns
        -------
        neural_score: float
            NN-based valuation of `state`. Will always be `0` if no model is loaded.
        """
        if self.model is None:
            return 0  # Si no hay modelo, devolver 0

        # Convertir a matriz
        state_matrix = self.state_to_matrix(state)

        # Convertir a tensor
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)

        # Obtener predicciones
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

        # Obtener acciones legales
        legal_actions = state.getLegalActions()

        # puntuación de la red
        neural_score = 0
        for i, action in enumerate(self.idx_to_action.values()):
            if action in legal_actions:
                neural_score += probabilities[i] * 100

        return neural_score

    def heuristic_eval(self, state) -> float:
        """
        Provides a valuation of a given pacman state using exclusively human-written heuristics.
        output.

        Parameters
        ----------
        state
            The board state to evaluate.

        Returns
        -------
        heuristic_score: float
            heuristic-based valuation of `state`.

        Heurísticas adicionales
        -----------------------
        - Prefer going toward power capsules.
        - Avoid going for further power capsules when under the effect of one.
        - Avoid 'undoing' moves unless necessary.
        """
        # Aplicar heurísticas adicionales, similar a betterEvaluationFunction
        heuristic_score = state.getScore()

        # Mejorar la evaluación con conocimiento del dominio
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()
        legal_actions = state.getLegalActions()

        # Factor 1: Distancia a la comida más cercana
        if food:
            min_food_distance = min(
                manhattanDistance(pacman_pos, food_pos) for food_pos in food
            )
            heuristic_score += 1.0 / (min_food_distance + 1)

        # Factor 2: Proximidad a fantasmas
        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = manhattanDistance(pacman_pos, ghost_pos)

            if ghost_state.scaredTimer > 0:
                # Si el fantasma está asustado, acercarse a él
                heuristic_score += 50 / (ghost_distance + 1)
            else:
                # Si no está asustado, evitarlo
                if ghost_distance <= 2:
                    heuristic_score -= (
                        200  # Gran penalización por estar demasiado cerca
                    )

        # Factor 3: Distancia a la cápsula de poder más cercana
        power_capsules = state.getCapsules()

        if power_capsules:
            power_active = any(
                g.scaredTimer > 0 for g in ghost_states
            )  # detect power capsule effect
            min_capsule_distance = min(
                manhattanDistance(pacman_pos, cap_pos) for cap_pos in power_capsules
            )

            if not power_active:
                heuristic_score += 5.0 / (min_capsule_distance + 1)
            else:
                # Strongly discourage consuming capsules while powered
                heuristic_score -= 100.0 / (min_capsule_distance + 1)

        # Factor 4: Discourage "undoing" moves
        opposites = {
            Directions.NORTH: Directions.SOUTH,
            Directions.SOUTH: Directions.NORTH,
            Directions.EAST: Directions.WEST,
            Directions.WEST: Directions.EAST,
        }

        if (
            hasattr(self, "last_action") and self.last_action in opposites
        ):  # (last move might've been STOP)
            undo = opposites[self.last_action]
            if undo in legal_actions:
                heuristic_score -= 10

        return heuristic_score

    def evaluationFunction(self, state) -> float:
        """
        Una función de evaluación basada en la red neuronal y en heurísticas
        adicionales.

        Heurísticas adicionales
        -----------------------
        - Prefer going toward power capsules.
        - Avoid going for further power capsules when under the effect of one.
        - Avoid 'undoing' moves unless necessary.
        """

        heuristic_score: float = self.heuristic_eval(state=state)
        neural_score: float = self.neural_eval(state=state)

        return heuristic_score + neural_score

    def _return_action(self, action):
        self.last_action = action
        return action

    def getAction(self, state):
        """
        Devuelve la mejor acción basada en la evaluación de la red neuronal
        y heurísticas adicionales.
        """
        self.move_count += 1

        # Si no hay modelo, hacer un movimiento aleatorio
        if self.model is None:
            print("ERROR: Modelo no cargado. Haciendo movimiento aleatorio.")
            exit()
            legal_actions = state.getLegalActions()
            return self._return_action(random.choice(legal_actions))

        # Obtener acciones legales
        legal_actions = state.getLegalActions()

        # Evaluación directa con la red neuronal
        state_matrix = self.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

        # Mapear índices del modelo a acciones del juego
        action_probs = []
        for idx, prob in enumerate(probabilities):
            action = self.idx_to_action[idx]
            if action in legal_actions:
                action_probs.append((action, prob))

        # Ordenar por probabilidad (mayor a menor)
        action_probs.sort(key=lambda x: x[1], reverse=True)

        # Exploración: con una probabilidad decreciente, elegir aleatoriamente
        exploration_rate = 0.2 * (0.99**self.move_count)  # Disminuye con el tiempo
        if random.random() < exploration_rate:
            # Excluir STOP si es posible
            if len(legal_actions) > 1 and Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            return self._return_action(random.choice(legal_actions))

        # Evaluación alternativa: generar sucesores y evaluar cada uno
        successors = []
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            eval_score = self.evaluationFunction(successor)
            neural_score = 0
            for a, p in action_probs:
                if a == action:
                    neural_score = p * 100
                    break
            # Combinar evaluación heurística con la predicción de la red
            combined_score = eval_score + neural_score

            # Penalizar STOP a menos que sea la única opción
            if action == Directions.STOP and len(legal_actions) > 1:
                combined_score -= 50

            successors.append((action, combined_score))

        # Ordenar por puntuación combinada
        successors.sort(key=lambda x: x[1], reverse=True)

        # Devolver la mejor acción
        return self._return_action(successors[0][0])


# Definir una función para crear el agente
def createNeuralAgent(model_path="models/pacman_model.pth"):
    """
    Función de fábrica para crear un agente neuronal.
    Útil para integrarse con la estructura de pacman.py.
    """
    return NeuralAgent(model_path)


# AlphaBeta con pesos:
class AlphaBetaNeuralAgent(AlphaBetaAgent, NeuralAgent):
    """
    Implements the Alpha-Beta algorithm with a weighted eval function, giving the result
    final_score = w_heuristic * heuristic_score + w_neural * neural_score
    """

    def __init__(
        self,
        w_heuristic=0.5,
        w_neural=0.5,
        model_path="models/pacman_model.pth",
        depth="2",
    ) -> None:
        """
        Constructor of the class

        Args
        ----
        w_heuristic (float) -> the weight of the heuristic part of the eval function.
        w_neural (float) -> the weight of the neural part of the eval function.
        """

        # Initialize AlphaBetaAgent (search logic)
        AlphaBetaAgent.__init__(self, depth=depth)

        # Initialize NeuralAgent (eval functions)
        NeuralAgent.__init__(self, model_path=model_path)

        self._w_heuristic = w_heuristic
        self._w_neural = w_neural

        if self._w_heuristic + self._w_neural != 1:
            self.normalise()

    def normalise(self) -> None:
        """Normalises the weights to make sure they don't excede 1
        This is a method I decided to include in case any mistake is made when receiving the weight parameters,
        it wasn't asked in the practice :)
        """
        total = self._w_heuristic + self._w_neural
        if total == 0:
            return
        self._w_heuristic /= total
        self._w_neural /= total

    def evaluationFunction(self, state) -> float:

        heuristic_score: float = self.heuristic_eval(state=state)
        neural_score: float = self.neural_eval(state=state)

        return self._w_heuristic * heuristic_score + self._w_neural * neural_score
