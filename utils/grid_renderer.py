import numpy as np
import plotly.colors as colors

COLOR_LIST = [colors.hex_to_rgb(c) for c in colors.qualitative.Light24]


class Grid_Renderer:
    def __init__(self, grid_size, color_map, background_color=(0, 0, 0)):
        self.grid_size = int(grid_size)
        self.background_color = background_color
        self.color_map = color_map

    def render_nd_grid(self, grid, flip=False):
        """
        The grid is supposed to be in this format (h, w, n_objects), the channel idx is supposed to be the object id
        """
        if flip:
            grid = np.swapaxes(grid, 0, 1)
            grid = np.flip(grid, axis=0)
        g_s = self.grid_size
        h, w = grid.shape[0], grid.shape[1]
        img = np.zeros(shape=(int(g_s * h), int(g_s * w), 3))
        for x in range(h):
            for y in range(w):
                if grid[x, y, :].any():
                    object_id = grid[x, y, :].argmax()
                    if object_id in self.color_map:
                        img[x * g_s:(x + 1) * g_s, y * g_s:(y + 1) * g_s, :] = self.color_map[object_id]
        return img.astype(np.uint8)

    def render_2d_grid(self, grid, flip=False):
        if flip:
            grid = np.swapaxes(grid, 0, 1)
            grid = np.flip(grid, axis=0)
        g_s = self.grid_size
        h, w = grid.shape[0], grid.shape[1]
        img = np.zeros(shape=(int(g_s * h), int(g_s * w), 3))
        for x in range(h):
            for y in range(w):
                object_id = grid[x, y]
                if object_id in self.color_map:
                    img[x * g_s:(x + 1) * g_s, y * g_s:(y + 1) * g_s, :] = self.color_map[object_id]
        return img.astype(np.uint8)


def main():
    color_map = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}
    grid = np.zeros(shape=(10, 10, 4), dtype=np.int8)
    grid[1, 1, 2] = 1
    grid[1, 2, 3] = 1
    grid[9, 9, 0] = 1
    grid[5, 6, 1] = 1

    grid_renderer = Grid_Renderer(grid_size=20, color_map=color_map)
    import cv2
    grid_img = grid_renderer.render_nd_grid(grid)
    cv2.imshow('grid render', cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()




