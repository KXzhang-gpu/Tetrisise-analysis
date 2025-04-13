# -*- coding: UTF-8 -*-
import time

import cv2
import pygame
import random
import os

from camera_surface import CameraSurface
from chart_surface import ChartSurface
from analysis import get_joint_angles, calculate_energy

"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

# Initialize the game engine
pygame.init()

# Sizes and locations
s_width = 1400
s_height = 750
grid_width = 300  # meaning 300 // 10 = 30 width per block
grid_height = 600  # meaning 600 // 20 = 20 height per block
block_size = 30

grid_pos_x = s_width - grid_width - 540
grid_pos_y = s_height - grid_height - 20

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BEIGE = (247, 243, 239)

# Initialize the screen
screen = pygame.display.set_mode((s_width, s_height))
pygame.display.set_caption('Tetris')

# Initialize the surface components
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

camera = CameraSurface(camera_index=0, camera_size=(200, 300))
angle_charts = []
joint_names = ['Left Arm', 'Right Arm', 'Knee', 'Hip']
for i in range(4):
    chart = ChartSurface(
        size=(camera.capture.get(cv2.CAP_PROP_FRAME_WIDTH), 100),
        ylim=(0, 180),
        history=100,
        title=f"{joint_names[i] }Angle",
        save_file=os.path.join(output_dir ,f"{joint_names[i].replace(' ', '_')}_Angle.csv")
    )
    angle_charts.append(chart)

# SHAPE FORMATS
S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....'],
     ['.....',
      '.....',
      '.....',
      '.0...',
      '.....'],
     ['.....',
      '.000.',
      '.000.',
      '.000.',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]

backgrounds = {
    pygame.K_o: pygame.image.load("./game/Crunch.png").convert(),
    pygame.K_j: pygame.image.load("./game/Rightpushup.png").convert(),
    pygame.K_l: pygame.image.load("./game/Leftpushup.png").convert(),
    pygame.K_s: pygame.image.load("./game/Rightsquat.png").convert(),
    pygame.K_z: pygame.image.load("./game/Leftsquat.png").convert(),
    pygame.K_t: pygame.image.load("./game/Overheadclap.png").convert(),
    pygame.K_i: pygame.image.load("./game/Background.png").convert(),
}

exercise_counting = {
    pygame.K_o: 0,
    pygame.K_j: 0,
    pygame.K_l: 0,
    pygame.K_s: 0,
    pygame.K_z: 0,
    pygame.K_t: 0,
    pygame.K_i: 0,
}

def update_exercise_counting(key):
    global exercise_counting
    if key in exercise_counting:
        exercise_counting[key] += 1

class Piece(object):
    rows = 20  # y
    columns = 10  # x

    def __init__(self, column, row, shape):
        self.x = column
        self.y = row
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0  # number from 0-3


def create_grid(locked_positions={}):
    grid = [[(0, 0, 0) for x in range(10)] for x in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_positions:
                c = locked_positions[(j, i)]
                grid[i][j] = c
    return grid


def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    accepted_positions = [[(j, i) for j in range(10) if grid[i][j] == (0, 0, 0)] for i in range(20)]
    accepted_positions = [j for sub in accepted_positions for j in sub]
    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_positions:
            if pos[1] > -1:
                return False

    return True


def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False

shape_key = None
get_times = 0
def get_shape():
    global shapes, shape_colors, shape_key, get_times
    key = None
    get_times += 1
    mp = {pygame.K_s: S, pygame.K_z: Z, pygame.K_i: I, pygame.K_o: O, pygame.K_j: J, pygame.K_l: L, pygame.K_t: T}
    if get_times % 5 != 0:
        receive_key = False
        while not receive_key:
            # insert camera controller
            update_camera_and_chart()
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    update_exercise_counting(event.key)

                    if (event.key in [pygame.K_s, pygame.K_z, pygame.K_i,
                                     pygame.K_o, pygame.K_j, pygame.K_l, pygame.K_t]
                            and event.key != shape_key):
                        key = event.key
                        receive_key = True
                        break
    else:
        key = random.choice([pygame.K_s, pygame.K_z, pygame.K_i, pygame.K_o, pygame.K_j, pygame.K_l, pygame.K_t])
    shape_key = key
    # return Piece(5, 0, random.choice(shapes))
    return Piece(5, 0, mp[key])



def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont('comicsans', size, bold=True)
    text = font.render(text, 1, color)

    text_rect = text.get_rect()
    text_rect.centerx = surface.get_width() // 2

    surface.blit(text, text_rect)


def draw_grid(surface, row, col):
    sx = grid_pos_x
    sy = grid_pos_y
    for i in range(row):
        pygame.draw.line(surface, (128, 128, 128), (sx, sy + i * 30),
                         (sx + grid_width, sy + i * 30))  # horizontal lines
        for j in range(col):
            pygame.draw.line(surface, (128, 128, 128), (sx + j * 30, sy),
                             (sx + j * 30, sy + grid_height))  # vertical lines


def clear_rows(grid, locked):
    # need to see if row is clear the shift every other row above down one

    inc = 0
    for i in range(len(grid) - 1, -1, -1):
        row = grid[i]
        if (0, 0, 0) not in row:
            inc += 1
            # add positions to remove from locked
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j, i)]
                except:
                    continue
    if inc > 0:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + inc)
                locked[newKey] = locked.pop(key)


def draw_next_shape(shape, surface):
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Next Shape', 1, (255, 255, 255))

    sx = grid_pos_x + grid_width + 50
    sy = grid_pos_y + grid_height / 2 - 100
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color, (sx + j * 30, sy + i * 30, 30, 30), 0)

    surface.blit(label, (sx + 10, sy - 30))


def draw_window(surface, image_bg=None):
    surface.fill(BEIGE)
    if image_bg is None:
        image_bg = backgrounds[pygame.K_i]
    screen.blit(image_bg, (s_width - image_bg.get_width(), 0))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j], (grid_pos_x + j * 30, grid_pos_y + i * 30, 30, 30), 0)

    # draw grid and border
    draw_grid(surface, 20, 10)
    pygame.draw.rect(surface, (255, 0, 0), (grid_pos_x, grid_pos_y, grid_width, grid_height), 5)


    # pygame.display.update()

def update_camera_and_chart():
    cam_surface = camera.update()
    if cam_surface:
        screen.blit(cam_surface, (20, 20))
    if camera.data is not None:
        joint_angles = get_joint_angles(camera.data)
        for i, chart in enumerate(angle_charts):
            chart.update(joint_angles[i])
            screen.blit(chart.surface, (20, cam_surface.get_height() + 40 + i * chart.height))
            chart.draw()


def main():
    global grid

    locked_positions = {}  # (x,y):(255,0,0)
    grid = create_grid(locked_positions)
    change_piece = False
    run = True
    current_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    level_time = 0
    fall_speed = 1
    score = 0
    energy_cost = 0
    prev_time = time.time()

    while run:

        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        level_time += clock.get_rawtime()
        clock.tick()

        if level_time / 1000 > 4:
            level_time = 0
            # if fall_speed > 0.15:
            #    fall_speed -= 0.005

        # PIECE FALLING CODE
        if fall_time / 1000 >= fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not (valid_space(current_piece, grid)) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                update_exercise_counting(event.key)

                if event.key == pygame.K_a:
                    current_piece.x -= 1
                    if not valid_space(current_piece, grid):
                        current_piece.x += 1

                elif event.key == pygame.K_d:
                    current_piece.x += 1
                    if not valid_space(current_piece, grid):
                        current_piece.x -= 1
                elif event.key == pygame.K_w or event.key == shape_key:
                    # rotate shape
                    current_piece.rotation = current_piece.rotation + 1 % len(current_piece.shape)
                    if not valid_space(current_piece, grid):
                        current_piece.rotation = current_piece.rotation - 1 % len(current_piece.shape)

                elif (shape_key == pygame.K_i and
                      event.key in [pygame.K_s, pygame.K_z, pygame.K_o, pygame.K_j, pygame.K_l, pygame.K_t]):

                    # rotate shape
                    current_piece.rotation = current_piece.rotation + 1 % len(current_piece.shape)
                    if not valid_space(current_piece, grid):
                        current_piece.rotation = current_piece.rotation - 1 % len(current_piece.shape)

        shape_pos = convert_shape_format(current_piece)

        # add piece to the grid for drawing
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = current_piece.color

        # IF PIECE HIT GROUND
        if change_piece:
            for pos in shape_pos:
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color
            current_piece = get_shape()
            change_piece = False

            # call four times to check for multiple clear rows
            if clear_rows(grid, locked_positions):
                score += 10

        # update the game window
        draw_window(screen, image_bg=backgrounds[shape_key])

        # draw score and counting of exercise
        energy_cost += calculate_energy(camera.controller, time.time() - prev_time)
        prev_time = time.time()

        font = pygame.font.SysFont('Calibri', 25, True, False)
        score_label = font.render("Score: " + str(score), 1, BLACK)
        pushup = font.render("Pushups: " + str(exercise_counting[pygame.K_l] + exercise_counting[pygame.K_j]),
                             True, BLACK)
        squat = font.render("Squats: " + str(exercise_counting[pygame.K_s] + exercise_counting[pygame.K_z]),
                            True, BLACK)
        overhead = font.render("Claps: " + str(exercise_counting[pygame.K_t]), True, BLACK)
        crunch = font.render("Crunches: " + str(exercise_counting[pygame.K_o]), True, BLACK)
        energy = font.render("Energy: " + f'{energy_cost: .1f}' + ' cal', True, GRAY)

        screen.blit(score_label, (grid_pos_x - 150, grid_pos_y))
        screen.blit(pushup, (grid_pos_x - 150, grid_pos_y + 30))
        screen.blit(squat, (grid_pos_x - 150, grid_pos_y + 60))
        screen.blit(overhead, (grid_pos_x - 150, grid_pos_y + 90))
        screen.blit(crunch, (grid_pos_x - 150, grid_pos_y + 120))
        screen.blit(energy, (grid_pos_x - 180, grid_pos_y + 150))


        # update the charts and camera
        update_camera_and_chart()

        pygame.display.update()

        # Check if user lost
        if check_lost(locked_positions):
            run = False

    draw_text_middle("You Lost", 40, (255, 255, 255), screen)
    pygame.display.update()
    pygame.time.delay(2000)


def main_menu():
    run = True
    image = pygame.image.load("./game/Exercises.png").convert()
    while run:
        screen.fill(BEIGE)
        draw_text_middle('Do a exercise!', 60, GRAY, screen)
        screen.blit(image, image.get_rect(center = screen.get_rect().center))

        pygame.display.update()
        main()

    pygame.quit()
    camera.release()


main_menu()  # start game
