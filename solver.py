import numpy as np
from collections import Counter
from tqdm import tqdm
import pickle
import os


class ScrabbleBoard:
    BOARD_SIZE = 15
    VALID_POSITIONS = {(i, j) for i in range(15) for j in range(15)}

    TRIPLE_WORD_SCORE = {(0, 0), (7, 0), (14, 0), (0, 7),
                         (14, 7), (0, 14), (7, 14), (14, 14)}
    DOUBLE_WORD_SCORE = {(1, 1), (2, 2), (3, 3), (4, 4), (1, 13), (2, 12), (3, 11), (4, 10), (
        13, 1), (12, 2), (11, 3), (10, 4), (13, 13), (12, 12), (11, 11), (10, 10), (7, 7)}
    TRIPLE_LETTER_SCORE = {(1, 5), (1, 9), (5, 1), (5, 5), (5, 9),
                           (5, 13), (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)}
    DOUBLE_LETTER_SCORE = {(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14), (6, 2), (6, 6), (6, 8), (6, 12), (
        7, 3), (7, 11), (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14), (12, 6), (12, 8), (14, 3), (14, 11)}

    LETTER_VALUES = {
        "A": 1, "B": 3, "C": 3, "D": 2, "E": 1, "F": 4, "G": 2, "H": 4, "I": 1,
        "J": 8, "K": 5, "L": 1, "M": 3, "N": 1, "O": 1, "P": 3, "Q": 10, "R": 1,
        "S": 1, "T": 1, "U": 1, "V": 4, "W": 4, "X": 8, "Y": 4, "Z": 10,  # "#": 0
    }

    INITIAL_TILE_COUNTS = {
        "A": 9, "B": 2, "C": 2, "D": 4, "E": 12, "F": 2, "G": 3, "H": 2, "I": 9,
        "J": 1, "K": 1, "L": 4, "M": 2, "N": 6, "O": 8, "P": 2, "Q": 1, "R": 6,
        "S": 4, "T": 6, "U": 4, "V": 2, "W": 2, "X": 1, "Y": 2, "Z": 1,  # "#": 2
    }

    CORPUS_FILENAME = 'sowpods.txt'

    if os.path.isfile('corpus.pkl'):
        CORPUS = pickle.load(open('corpus.pkl', 'rb'))
    else:
        CORPUS = set()
        with open(CORPUS_FILENAME) as file:
            for line in file:
                word = line.strip().upper()
                CORPUS.add(word)
        pickle.dump(CORPUS, open('corpus.pkl', 'wb'))

    WORD_SETS = {length: set() for length in range(2, 16)}
    for word in CORPUS:
        WORD_SETS[len(word)].add(word)

    def __init__(self):
        self.squares = np.full(
            (self.BOARD_SIZE, self.BOARD_SIZE), ' ', dtype=np.dtype('<U1'))
        self.tiles = set()
        self.unused_tiles = Counter(self.INITIAL_TILE_COUNTS)
        start_pos = (self.BOARD_SIZE // 2, self.BOARD_SIZE // 2)

        self.halo = {start_pos}  # Initial halo at the center
        self.halo_letters_across = {start_pos: set(self.LETTER_VALUES.keys())}
        self.halo_letters_down = {start_pos: set(self.LETTER_VALUES.keys())}

        self.plays = []
        self.scores = []

    def board_string(self):
        top_border = '       ' + \
            ' '.join([f'{i:<3d}' for i in range(self.BOARD_SIZE)])

        rows = []
        for i in range(self.BOARD_SIZE):
            row = [f'{i:<3d} |']
            for j in range(self.BOARD_SIZE):
                if (i, j) in self.tiles:
                    row.append(f'[{self.squares[i, j]}]')
                # elif (i, j) in self.halo:
                #     row.append(f' @ ')
                #     len_across = len(self.halo_letters_across[(i, j)])
                #     len_down = len(self.halo_letters_down[(i, j)])
                #     str_ac = f'{len_across:X}' if len_across <27 else '-'
                #     str_dn = f'{len_down:X}' if len_down <27 else '-'
                #     row.append(f'{str_ac},{str_dn}')
                elif (i, j) in self.TRIPLE_WORD_SCORE:
                    row.append(f'w*3')
                elif (i, j) in self.DOUBLE_WORD_SCORE:
                    row.append(f'w*2')
                elif (i, j) in self.TRIPLE_LETTER_SCORE:
                    row.append(f'l*3')
                elif (i, j) in self.DOUBLE_LETTER_SCORE:
                    row.append(f'l*2')
                else:
                    row.append(f' _ ')
            rows.append(' '.join(row))

        return top_border + '\n' + '\n'.join(rows) + '\n'

    def __str__(self):

        s = '\n' + self.board_string() + '\n'

        s += f'Total score: {sum(self.scores)}\n'
        s += f'Scores: {self.scores}\n'
        s += f'Words: {[play[2] for play in self.plays]}\n'
        # s += f'Halo squares (@): {len(self.halo)}\n'
        s += f'Tiles placed|remaining: {len(self.tiles)}|{sum(self.unused_tiles.values())}\n'
        for i, letter in enumerate(self.unused_tiles.keys()):
            if i % 9 == 0:
                s += '\n'
            s += f'{letter} :{self.unused_tiles[letter]:2d}    '

        return s + '\n'

    def get_tiles_left(self):
        return sum(self.unused_tiles.values())

    def place_word(self, row: int, col: int, word: str, direction: str) -> None:
        if word not in self.CORPUS:
            raise ValueError(f'{word} is not in the corpus.')

        # Calculate score before placing the word
        score, _ = self.calculate_score(row, col, word, direction)

        print_word = ''
        for i, letter in enumerate(word):
            x,y = row + i if direction == 'down' else row, col + i if direction == 'across' else col
            if (x,y) in self.tiles:
                print_word += f'({letter})'
            else:
                print_word += letter

        print(
            f'Placing {print_word} at ({row}, {col}) {direction} for {score} points.')

        # New tiles are the tiles that will be placed but not the tiles that are already on the board
        new_tiles = set((row + i, col) if direction == 'down' else (row, col + i)
                        for i in range(len(word))).difference(self.tiles)

        for i, letter in enumerate(word):
            # Update the board squares array
            x = row + i if direction == 'down' else row
            y = col + i if direction == 'across' else col
            self.squares[x, y] = letter

            # Update the tile counts if the tile is not already on the board
            if (x, y) not in self.tiles:
                self.unused_tiles[letter] -= 1
                if self.unused_tiles[letter] < 0:
                    raise ValueError(f'Not enough {letter}s in the bag.')

        self.plays.append((row, col, word, direction))
        self.scores.append(score)

        # Update tiles and halo
        self.tiles.update(new_tiles)
        self.update_halo(new_tiles)

    def save_game(self, filename):
        with open(filename, 'a') as file:
            file.write(self.board_string())
            file.write(f'Total score: {sum(self.scores)}\n')
            for play, score in zip(self.plays, self.scores):
                file.write(
                    f'{score:<4} --- row: {play[0]:<2} col: {play[1]:<2} dir: {play[3]:<7} word: {play[2]}\n')
            file.write('\n\n')

    def update_halo(self, new_tiles) -> None:
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, right, down, left
        new_halo_squares = {(x + dx, y + dy)
                            for x, y in new_tiles for dx, dy in neighbors}
        # Only keep squares within the board
        new_halo_squares.intersection_update(self.VALID_POSITIONS)
        # Remove squares that have tiles
        new_halo_squares.difference_update(self.tiles)

        # Update the halo locations
        self.halo.update(new_halo_squares)
        self.halo.difference_update(self.tiles)  # Remove tiles from the halo

        for square in self.halo:
            self.halo_letters_across[square] = self.get_cross_letters(
                square[0], square[1], 'across')
            self.halo_letters_down[square] = self.get_cross_letters(
                square[0], square[1], 'down')

        # Remove halo letter keys which are no longer in the halo
        for square in list(self.halo_letters_across.keys()):
            if square not in self.halo:
                del self.halo_letters_across[square]
        for square in list(self.halo_letters_down.keys()):
            if square not in self.halo:
                del self.halo_letters_down[square]

    def get_cross_letters(self, row: int, col: int, direction: str):
        dx, dy = (0, 1) if direction == 'across' else (1, 0)

        valid_letters = set()
        left_word, right_word = '', ''
        x, y = row - dx, col - dy

        # Get letters before the halo square
        while (x, y) in self.tiles:
            left_word = self.squares[x, y] + left_word
            x, y = x - dx, y - dy

        x, y = row + dx, col + dy

        # Get letters after the halo square
        while (x, y) in self.tiles:
            right_word += self.squares[x, y]
            x, y = x + dx, y + dy

        # If both left and right word exist, then the halo is in the middle of a word.
        if left_word and right_word:
            length = len(left_word) + len(right_word) + 1
            if length > 1 and length < 16:
                # Check each word in the word set of the corresponding length
                for word in self.WORD_SETS[length]:
                    if word.startswith(left_word) and word.endswith(right_word):
                        valid_letters.add(word[len(left_word)])

        elif left_word:  # The halo is at the end of a word
            length = len(left_word) + 1
            for word in self.WORD_SETS[length]:
                if word.startswith(left_word):
                    valid_letters.add(word[len(left_word)])

        elif right_word:  # The halo is at the start of a word
            length = len(right_word) + 1
            for word in self.WORD_SETS[length]:
                if word.endswith(right_word):
                    valid_letters.add(word[-len(right_word)-1])
        else:
            # The halo is not touching a word, so all letters are valid in this direction
            valid_letters = set(self.LETTER_VALUES.keys())

        return valid_letters

    def calculate_score(self, row: int, col: int, word: str, direction: str) -> int:
        avaliable_tiles = self.unused_tiles.copy()
        main_score = 0
        cross_scores = 0
        word_multiplier = 1

        num_new_tiles = 0

        for i, letter in enumerate(word):

            letter_multiplier = 1
            pos = (row + i, col) if direction == "down" else (row, col + i)

            position_score = 0
            position_word_mult = 1

            if pos not in self.tiles:
                num_new_tiles += 1

                # Check if the tile is avaliable
                if avaliable_tiles[letter] <= 0:
                    return -1 , -1

                # Otherwise use the tile
                avaliable_tiles[letter] -= 1

                if pos in self.DOUBLE_LETTER_SCORE:
                    letter_multiplier = 2
                elif pos in self.TRIPLE_LETTER_SCORE:
                    letter_multiplier = 3

                if pos in self.DOUBLE_WORD_SCORE:
                    position_word_mult = 2
                    word_multiplier *= 2
                elif pos in self.TRIPLE_WORD_SCORE:
                    position_word_mult = 3
                    word_multiplier *= 3

            position_score = self.LETTER_VALUES[letter] * letter_multiplier

            main_score += self.LETTER_VALUES[letter] * letter_multiplier

            # Check for a cross word at this position
            cross_score = 0
            if pos not in self.tiles:
                if direction == "down":
                    cross_score = self.get_cross_word_value(
                        pos[0], pos[1], "across")
                else:  # direction == "across"
                    cross_score = self.get_cross_word_value(
                        pos[0], pos[1], "down")

            if cross_score > 0:
                cross_score = (cross_score + position_score) * \
                    position_word_mult

            cross_scores += cross_score

        bingo_bonus = 50 if num_new_tiles == 7 else 0

        play_score = main_score * word_multiplier + cross_scores + bingo_bonus

        norm_score = play_score/num_new_tiles

        return play_score, norm_score

    def get_cross_word_value(self, row: int, col: int, direction: str) -> int:
        # Direction to move
        dx, dy = (0, -1) if direction == 'across' else (-1, 0)
        word = ''  # Word to build
        x, y = row + dx, col + dy

        # Traverse in the negative direction
        while (x, y) in self.tiles:
            word = self.squares[x, y] + word
            x, y = x + dx, y + dy

        # Reverse direction
        dx, dy = -dx, -dy
        x, y = row + dx, col + dy

        # Traverse in the positive direction
        while (x, y) in self.tiles:
            word += self.squares[x, y]
            x, y = x + dx, y + dy

        # Return the score of the word (if there is a word)
        return self.raw_word_score(word) if word else 0

    def raw_word_score(self, word: str) -> int:
        return sum(self.LETTER_VALUES[letter] for letter in word)

    # Get constraint satisfaction problems as a list
    def get_csps(self, min_new_letters=1):
        assert min_new_letters >= 1 and min_new_letters <= 7

        csps = []
        for direction in ['across', 'down']:
            dx, dy = (0, 1) if direction == 'across' else (1, 0)
            for row, col in self.VALID_POSITIONS:

                # Ignore this csp if there is a tile letter behind it
                if (row-dx, col-dy) in self.tiles:
                    continue

                # Create a list starting at that spot until 7 non tile locations are met or the edge of the board
                csp = [row, col, direction]
                halo_letters = self.halo_letters_across if direction == 'down' else self.halo_letters_down
                new_letter_count = 0
                letter_count = 0
                x, y = row, col
                constraints = dict()
                halo_hit = False
                i = 0
                while (x, y) in self.VALID_POSITIONS:

                    # If a board tile is found, add it to the constraints
                    if (x, y) in self.tiles:
                        constraints[i] = set(self.squares[x, y])
                    # If a halo is found, add it to the constraints
                    elif (x, y) in self.halo:
                        halo_hit = True
                        constraints[i] = halo_letters[(x, y)]

                    letter_count += 1
                    # If a new tile is found, increment the new tile count
                    if (x, y) not in self.tiles:
                        new_letter_count += 1

                    x, y = x+dx, y+dy
                    i += 1

                    # If the next position is not a tile, we can add the current csp to the list of csps
                    if (x, y) not in self.tiles and halo_hit and (min_new_letters <= new_letter_count <= 7) and (letter_count > 1):
                        partial_csp = csp.copy()
                        partial_csp.append(letter_count)
                        partial_csp.append(constraints.copy())
                        csps.append(partial_csp)

        return csps

    def solve_csp(csp):
        _, _, _, length, constraints = csp

        # Get all words of the correct length
        words = ScrabbleBoard.WORD_SETS[length]

        # Reduce the words to only those that match the constraints
        return {word for word in words if all(word[index] in constraint for index, constraint in constraints.items())}

    def sample_play_by_score(self, min_new_letters=1, temp=1):
        csps = self.get_csps(min_new_letters=min_new_letters)

        plays = []
        scores = []
        for row, col, direction, length, constraints in tqdm(csps, desc='Solving...'):
            words = ScrabbleBoard.solve_csp(
                (row, col, direction, length, constraints))
            for word in words:
                score, score_norm = self.calculate_score(row, col, word, direction)
                if score_norm > 0:
                    plays.append((row, col, word, direction))
                    scores.append(score_norm)

        # Normalize and sample
        scores = np.array(scores)
        scores = np.exp(scores/(temp+0.001))
        scores = scores / np.sum(scores)
        index = np.random.choice(len(scores), p=scores)
        return plays[index]


if __name__ == '__main__':
    scrabble = ScrabbleBoard()
    print(scrabble)

    while scrabble.get_tiles_left() > 0:
        temp = 3 * scrabble.get_tiles_left() / 98
        print(f'Temperature: {temp:.2f}')

        min_new_letters = 1

        row, col, word, direction = scrabble.sample_play_by_score(
            temp=temp, min_new_letters=min_new_letters)
        scrabble.place_word(row, col, word, direction)
        print(scrabble)

    scrabble.save_game('games.txt')
