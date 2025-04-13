# -*- coding: UTF-8 -*-
import os
import csv
import time
import pygame
from collections import deque

from typing import Tuple

class ChartSurface:
    def __init__(
            self,
            size: Tuple[float, float] = (400, 150),
            ylim: Tuple[float, float] = (0, 100),
            history=100,
            title: str = "",
            save_file=None,
    ):
        # Initializing
        self.surface = pygame.Surface(size)
        # self.rect = self.surface.get_rect()

        # Deque setting
        self.history = history
        self.data = deque([0]*history, maxlen=history)
        self.ylim = ylim

        # Chart appearance
        self.bg_color = (30, 30, 30)
        self.axis_color = (200, 200, 200)
        self.line_color = (0, 255, 0)
        self.font = pygame.font.SysFont(None, 16)
        self.title = title

        # Smoothing data using Exponential Moving Average (EMA)
        self.alpha = 0.2
        self.smooth_data = None

        # recording data
        self.save_file = save_file
        if self.save_file:
            with open(self.save_file, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["time", "raw", "smoothed"])
                writer.writeheader()
        self.start_time = None

    def update(self, value):
        # Apply EMA smoothing
        if self.smooth_data is None:
            self.smooth_data = value
        else:
            self.smooth_data = (1 - self.alpha) * self.smooth_data + self.alpha * value
        # Recoding the new value to plot
        self.data.append(self.smooth_data)

        if self.save_file:
            if self.start_time is None:
                self.start_time = time.time()

            with open(self.save_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["time", "raw", "smoothed"])
                writer.writerow({
                    "time": time.time() - self.start_time,
                    "raw": value,
                    "smoothed": self.smooth_data
                })

    def draw(self):
        self.surface.fill(self.bg_color)
        w, h = self.surface.get_size()

        # Draw title
        title_surf = self.font.render(self.title, True, (255, 255, 255))
        self.surface.blit(title_surf, ((w - title_surf.get_width()) // 2, 5))
        top_margin = 20

        ox, oy = 40, h - 30
        chart_w = w - ox - 10
        chart_h = h - 40 - top_margin

        # Draw axes
        pygame.draw.line(self.surface, self.axis_color, (ox, oy), (ox, oy - chart_h), 1)
        pygame.draw.line(self.surface, self.axis_color, (ox, oy), (ox + chart_w, oy), 1)

        # Y ticks
        for i in range(5):
            y = oy - i * chart_h // 4
            val = int(i * self.ylim[1] / 4 + self.ylim[0])
            pygame.draw.line(self.surface, self.axis_color, (ox - 5, y), (ox + 5, y))
            text = self.font.render(str(val), True, self.axis_color)
            self.surface.blit(text, (5, y - 8))

        # X ticks
        for i in range(0, self.history + 1, 25):
            x = ox + i * chart_w // self.history
            pygame.draw.line(self.surface, self.axis_color, (x, oy - 5), (x, oy + 5))
            text = self.font.render(str(i), True, self.axis_color)
            self.surface.blit(text, (x - 10, oy + 8))

        # Draw line
        points = []
        for i, val in enumerate(self.data):
            x = ox + i * chart_w // self.history
            y = oy - int((val - self.ylim[0]) * chart_h / (self.ylim[1] - self.ylim[0]))
            points.append((x, y))
        if len(points) > 1:
            pygame.draw.lines(self.surface, self.line_color, False, points, 2)

        return self.surface

    @property
    def height(self):
        return self.surface.get_height()
    @property
    def width(self):
        return self.surface.get_width()
