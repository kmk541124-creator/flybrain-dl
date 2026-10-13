import numpy as np

FACE_AXIS = {"U": (1, 1), "D": (1, -1), "R": (0, 1), "L": (0, -1), "F": (2, 1), "B": (2, -1)}
COLOR_NAMES = ["white", "yellow", "red", "orange", "green", "blue"]
MOVES = ["U", "U'", "D", "D'", "R", "R'", "L", "L'", "F", "F'", "B", "B'"]

def solved_cube():
    position, normals, colors = [], [], []
    for color, (axis, sign) in enumerate(FACE_AXIS.values()):
        for a in (-1 , 0, 1):
            for b in (-1, 0, 1):
                pos = [a,b]
                pos.insert(axis, sign)

                normal = [0,0,0]
                normal[axis] = sign
                position.append(pos)
                normals.append(normal)
                colors.append(color)
    return np.array(position), np.array(normals), np.array(colors)

def rotation_martix(axis, s):
    R = np.eye(3, dtype=int)
    i,j = (axis + 1) % 3, (axis + 2) % 3

    R[i, i] = 0
    R[j, j] = 0

    R[i, j] = -s
    R[j, i] = s

    return R

def apply_move(postions , normals, move):
    axis, sign = FACE_AXIS[move[0]]

    turn = -1 if "'" in move else 1

    s = -sign * turn

    R = rotation_martix(axis, s)

    layer = postions[:, axis] == sign

    new_pos = postions.copy()
    new_nor = normals.copy()

    new_pos[layer] = postions[layer] @ R.T
    new_nor[layer] = normals[layer] @ R.T

    return new_pos, new_nor

SOLVED_POS, SOLVED_NOR, SOLVED_COL = solved_cube()

SLOT_KEY = {tuple(p * 2 + n): i for i, (p, n) in enumerate(zip(SOLVED_POS, SOLVED_NOR))}

def slot_colors(postions, normals, colors):
    borads = np.zeros(54, dtype=int)

    for p, n, c in zip(postions, normals, colors):
        borads[SLOT_KEY[tuple(p * 2 + n)]] = c

    return borads


def is_solved(postions, normals, colors):
    return np.array_equal(slot_colors(postions, normals, colors), SOLVED_COL)


def scramble(defth, rang):
    pos, nor, col = solved_cube()
    before = None
    history = []

    for _ in range(defth):
        move = MOVES[rang.integers(len(MOVES))]
        while move[0] == before:
            move = MOVES[rang.integers(len(MOVES))]

        pos, nor = apply_move(pos,nor, move)
        history.append(move)
        before = move[0]

    return pos, nor, col, history

def observe(positions, normals , color):
    board = slot_colors(positions, normals, color)

    return np.eye(6)[board].flatten()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pos, nor, col, hist = scramble(3, rng)
    print(hist, is_solved(pos, nor, col))
    x = observe(pos, nor, col)
    print(x.shape, x.sum(), x[:12])