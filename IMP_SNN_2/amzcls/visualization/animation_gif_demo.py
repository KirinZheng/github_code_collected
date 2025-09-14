from typing import Iterable, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class GenGIF(object):
    def __init__(self, data: Union[np.ndarray, Iterable]):
        self.data = data
        self.figure = plt.figure()
        self.frame_handle = plt.imshow(data[0])
        self.animation = animation.FuncAnimation(
            fig=self.figure, func=self.updata, frames=np.arange(0, 16), interval=100)

    def updata(self, index):
        self.frame_handle.set_data(self.data[index])
        return [self.frame_handle]

    def save(self, path='out.gif'):
        self.animation.save(path)

    def show(self):
        plt.show()


if __name__ == '__main__':
    generate_gif = GenGIF(np.random.randint(0, 255, (16, 32, 32, 3)))
    generate_gif.show()
