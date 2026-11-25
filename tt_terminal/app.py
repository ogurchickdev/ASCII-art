import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pygame


RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
VIDEO_PATH = Path(__file__).resolve().parent / "video.mp4"
MAX_DURATION = 70.0
COLOR_LEVELS = 24
WHITE_LIFT = 0.7
SATURATION = 1.25
CONTRAST = 1.12
BRIGHTNESS = 8
FONT_SIZE = 11
OUTPUT_FPS = 30
MIN_FPS = 1.0


def video_info(capture):
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps < MIN_FPS or frame_count <= 0:
        raise ValueError("Не удалось определить длительность или FPS видео.")
    duration = frame_count / fps
    if duration > MAX_DURATION:
        raise ValueError(
            f"Видео длится {duration:.1f} сек. Максимальная длительность — 60 секунд."
        )
    return fps, frame_count, duration


class AsciiVideo:
    def __init__(self, path):
        pygame.init()
        pygame.display.set_caption("ASCII VIDEO")
        self.screen = pygame.display.set_mode((1200, 800), pygame.RESIZABLE)
        self.fullscreen = False
        self.windowed_size = self.screen.get_size()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", FONT_SIZE, bold=True)
        self.cell_w = max(1, self.font.size("M")[0])
        self.cell_h = max(1, self.font.get_linesize())
        self.glyph_masks = self.build_glyph_masks()
        self.grid_surface = None
        self.capture = None
        self.path = None
        self.fps = 0.0
        self.frame_count = 0
        self.frame_index = 0
        self.next_frame_index = 0
        self.start_time = 0.0
        self.paused_at = 0.0
        self.running = True
        self.playing = True
        self.message = ""
        self.load_video(path)

    def load_video(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Видео не найдено: {path}. Положите его рядом с app.py "
                "или измените VIDEO_PATH в начале файла."
            )
        candidate = cv2.VideoCapture(str(path))
        if not candidate.isOpened():
            candidate.release()
            raise ValueError(f"Не удалось открыть видео: {path}")

        try:
            fps, frame_count, duration = video_info(candidate)
        except ValueError:
            candidate.release()
            raise

        if self.capture is not None:
            self.capture.release()
        self.capture = candidate
        self.path = Path(path)
        self.fps = fps
        self.frame_count = frame_count
        self.frame_index = 0
        self.next_frame_index = 0
        self.start_time = time.perf_counter()
        self.paused_at = 0.0
        self.playing = True
        self.message = f"{self.path.name}  |  {duration:.1f}s"
        pygame.display.set_caption(f"ASCII VIDEO — {self.path.name}")
        self.draw_frame(0)

    def grid_size(self, aspect):
        width, height = self.screen.get_size()
        max_cols = max(1, width // self.cell_w)
        max_rows = max(1, height // self.cell_h)
        cols = min(
            max_cols,
            max(1, int(max_rows * self.cell_h * aspect / self.cell_w)),
        )
        rows = max(1, min(max_rows, int(cols * self.cell_w / self.cell_h / aspect)))
        return cols, rows

    def build_glyph_masks(self):
        masks = np.zeros((len(RAMP), self.cell_h, self.cell_w), dtype=np.uint8)
        for index, character in enumerate(RAMP):
            surface = self.font.render(character, True, (255, 255, 255))
            alpha = pygame.surfarray.array_alpha(surface).T
            height = min(self.cell_h, alpha.shape[0])
            width = min(self.cell_w, alpha.shape[1])
            masks[index, :height, :width] = alpha[:height, :width]
        return masks

    def frame_to_cells(self, frame, cols, rows):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (cols, rows), interpolation=cv2.INTER_AREA)
        small = cv2.convertScaleAbs(small, alpha=CONTRAST, beta=BRIGHTNESS)

        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1].astype(np.float32) * SATURATION, 0, 255
        ).astype(np.uint8)
        small = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        rgb_float = small.astype(np.float32)
        channel_min = rgb_float.min(axis=2)
        channel_max = rgb_float.max(axis=2)
        neutral = 1.0 - (channel_max - channel_min) / 255.0
        white_mask = np.clip((channel_min - 128.0) / 127.0, 0.0, 1.0) ** 2
        white_mask *= neutral
        small = np.clip(
            rgb_float + (255.0 - rgb_float) * white_mask[:, :, None] * WHITE_LIFT,
            0,
            255,
        ).astype(np.uint8)

        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        indices = np.clip(
            np.rint(np.power(gray, 0.72) * (len(RAMP) - 1)),
            0,
            len(RAMP) - 1,
        ).astype(np.uint8)

        levels = COLOR_LEVELS - 1
        colors = (small.astype(np.uint16) * levels // 255 * 255 // levels).astype(
            np.uint8
        )
        return indices, colors

    def draw_frame(self, frame_index):
        if frame_index < self.next_frame_index or frame_index - self.next_frame_index > 30:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self.next_frame_index = frame_index
        while self.next_frame_index < frame_index:
            success, _ = self.capture.read()
            if not success:
                self.message = "Не удалось прочитать кадр видео"
                self.playing = False
                return
            self.next_frame_index += 1
        success, frame = self.capture.read()
        if not success:
            self.message = "Не удалось прочитать кадр видео"
            self.playing = False
            return
        self.next_frame_index = frame_index + 1

        height, width = frame.shape[:2]
        cols, rows = self.grid_size(width / height)
        indices, colors = self.frame_to_cells(frame, cols, rows)

        grid_width = cols * self.cell_w
        grid_height = rows * self.cell_h
        screen_width, screen_height = self.screen.get_size()
        offset_x = (screen_width - grid_width) // 2
        offset_y = (screen_height - grid_height) // 2
        if self.grid_surface is None or self.grid_surface.get_size() != (grid_width, grid_height):
            self.grid_surface = pygame.Surface((grid_width, grid_height))

        alpha = self.glyph_masks[indices].transpose(0, 2, 1, 3).reshape(
            grid_height, grid_width
        )
        rgb = cv2.resize(colors, (grid_width, grid_height), interpolation=cv2.INTER_NEAREST)
        pixels = np.empty_like(rgb)
        for channel in range(3):
            pixels[:, :, channel] = cv2.multiply(
                rgb[:, :, channel],
                alpha,
                scale=1 / 255,
                dtype=cv2.CV_8U,
            )
        pygame.surfarray.blit_array(self.grid_surface, pixels.transpose(1, 0, 2))
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.grid_surface, (offset_x, offset_y))
        pygame.display.flip()

    def toggle_pause(self):
        if self.playing:
            self.playing = False
            self.paused_at = time.perf_counter()
        else:
            self.start_time += time.perf_counter() - self.paused_at
            self.playing = True

    def restart(self):
        self.start_time = time.perf_counter()
        self.frame_index = 0
        self.playing = True
        self.draw_frame(0)

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
            self.fullscreen = False
        else:
            self.windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.fullscreen = True
        self.draw_frame(self.frame_index)

    def run(self):
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            self.running = False
                        elif event.key == pygame.K_F11:
                            self.toggle_fullscreen()
                        elif event.key == pygame.K_SPACE:
                            self.toggle_pause()
                        elif event.key == pygame.K_r:
                            self.restart()
                    elif event.type == pygame.DROPFILE:
                        try:
                            self.load_video(event.file)
                        except (OSError, ValueError) as error:
                            print(f"Ошибка: {error}", file=sys.stderr)

                if not self.running:
                    break

                if self.playing:
                    target_frame = int((time.perf_counter() - self.start_time) * self.fps)
                    if target_frame >= self.frame_count:
                        self.restart()
                    elif target_frame != self.frame_index:
                        self.frame_index = target_frame
                        self.draw_frame(target_frame)

                self.clock.tick_busy_loop(OUTPUT_FPS)
        finally:
            if self.capture is not None:
                self.capture.release()
            pygame.quit()


def get_video_path(argument):
    if argument:
        return Path(argument)
    return VIDEO_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Показывает короткое видео как цветную ASCII-анимацию."
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Путь к видео; если не указан, используется VIDEO_PATH из app.py",
    )
    args = parser.parse_args()

    try:
        path = get_video_path(args.video)
        if not path:
            raise ValueError("Путь к видео не указан.")
        player = AsciiVideo(path)
        player.run()
    except (OSError, ValueError, cv2.error) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        if sys.stdin.isatty():
            input("Нажмите Enter, чтобы закрыть...")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
