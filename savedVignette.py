# coding: utf-8
import os

from environment import make_env
import pickle
import lzma

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as img

from PIL import Image, ImageDraw

import colorTest
from vector_util import valueToRGB, checkFormat
from arguments import get_args
import transformFunction
import argparse

saveFormat = '.xz'


@checkFormat(saveFormat)
def loadFromFile(filename, folder="SavedVignette"):
    """
	Returns a saved plot
	"""
    with lzma.open(folder + "/" + filename, 'rb') as handle:
        content = pickle.load(handle)
    return content


class SavedVignette:
    """
	Class storing a Vignette, able to draw it in 3D and 2D
	Useful to serialize in order to be able to change drawing parameters
	"""
    def __init__(self,
                 D,
                 indicesPolicies=None,
                 policyDistance=None,
                 stepalpha=.25,
                 color1=colorTest.color1,
                 color2=colorTest.color2,
                 pixelWidth=10,
                 pixelHeight=10,
                 x_diff=2.,
                 y_diff=2.,
                 colors=None,
                 env="CartPoleContinuous-v0",
                 policy="normal",
                 title="Landscape",
                 maxColor=None,
                 minColor=None):

        # Informations to show
        self.env = env
        self.policy = policy
        self.title = title
        self.maxColor = maxColor
        self.minColor = minColor

        # Content of the Vignette
        self.lines = []  # Upper lines
        self.linesLogProb = []  # log(P(A\S)) for upper lines
        self.directions = D  # All sampled directions
        self.indicesPolicies = indicesPolicies  # Index of directions that go through a policy
        self.policyDistance = policyDistance  # Distance of each policy along its direction

        # 2D plot
        self.stepalpha = stepalpha  # Distance between each model along a direction
        self.color1, self.color2 = color1, color2  # Min color and max color
        self.pixelWidth, self.pixelHeight = pixelWidth, pixelHeight  # Pixels' dimensions
        self.colors = colors

        # 3D plot
        self.fig, self.ax = None, None
        self.x_diff = x_diff  #	Distance between each model along a direction
        self.y_diff = y_diff  #  Distance between each direction

    @checkFormat(saveFormat)
    def saveInFile(self, filename):
        """
		Save the Vignette in a file
		"""
        with lzma.open(filename, 'wb') as handle:
            pickle.dump(self, handle)

    @checkFormat('.png')
    def save2D(self, filename, img=None):
        """
		Save the Vignette as 2D image
		"""
        img = self.image2D() if img is None else img
        img.save(filename, format='png')

    def save2Dplot(self, image, filename, img=None):
        """
		Save the Vignette as 2D image with matplotlib legend
		"""
        img = self.plot2D(filename=image) if img is None else img
        plt.savefig(filename + '.png')
        plt.savefig(filename + '.pdf')
        #plt.show()

    @checkFormat('.pdf')
    def save3D(self, filename, elevs=[30], angles=[0]):
        """
		Save the Vignette as a 3D image
		"""
        self.plot3D()
        for elev in elevs:
            for angle in angles:
                self.ax.view_init(elev, angle)
                plt.draw()
                plt.savefig(filename + '_e{}_a{}.pdf'.format(elev, angle),
                            format='pdf')
        #plt.show()

    def saveAll(self,
                filename,
                saveInFile=False,
                save2D=False,
                save3D=False,
                directoryFile="SavedVignette",
                directory2D="Vignette_output",
                directory3D="Vignette_output",
                computedImg=None,
                angles3D=[0],
                elevs=[0]):
        """
		Centralises the saving process
		"""
        #   Choosing max/min colors
        if self.env == "CartPoleContinuous-v0":
            self.maxColor = 200
            self.minColor = 0
        if self.env == "Pendulum-v0":
            self.maxColor = -100
            self.minColor = -1900

        if saveInFile is True: self.saveInFile(directoryFile + '/' + filename)
        if save2D is True:
            self.save2D(directory2D + '/' + filename + '_2D_image',
                        img=computedImg)
            self.save2Dplot(directory2D + '/' + filename + '_2D_image',
                            directory2D + '/' + filename + '_2D_plot',
                            img=computedImg)
        if save3D is True:
            self.save3D(directory3D + '/' + filename + '_3D_plot',
                        elevs=elevs,
                        angles=angles3D)

    def image2D(self, color1=None, color2=None):
        """
		Compute the 2D image of the Vignette

		Cannot store it as PIL images are non serializable
		"""
        color1, color2 = self.color1 if color1 is None else color1, self.color2 if color2 is None else color2

        width, height = self.pixelWidth * len(
            self.lines[-1]), self.pixelHeight * (len(self.lines))
        newIm = Image.new("RGB", (width, height))
        newDraw = ImageDraw.Draw(newIm)

        #	Adding the results
        y0 = 0
        for l in range(len(self.lines)):
            # 	Drawing the results
            y1 = y0 + self.pixelHeight
            for c in range(len(self.lines[l])):
                x0 = c * self.pixelWidth
                x1 = x0 + self.pixelWidth
                color = valueToRGB(self.lines[l][c],
                                   color1,
                                   color2,
                                   minNorm=self.minColor,
                                   maxNorm=self.maxColor)
                newDraw.rectangle([x0, y0, x1, y1], fill=color)
            y0 += self.pixelHeight

        # 	Adding the policies
        if self.indicesPolicies is not None:
            marginX, marginY = int(self.pixelWidth / 4), int(self.pixelHeight /
                                                             4)
            for k in range(len(self.indicesPolicies)):
                index, distance = self.indicesPolicies[k], round(
                    self.policyDistance[k] / self.stepalpha)
                x0, y0 = (distance + len(self.lines[0]) //
                          2) * self.pixelWidth, index * self.pixelHeight
                x1, y1 = x0 + self.pixelWidth, y0 + self.pixelHeight
                newDraw.ellipse(
                    [x0 + marginX, y0 + marginY, x1 - marginX, y1 - marginY],
                    fill=self.colors[k])

        return newIm

    def plot2D(self, filename):
        """
		Compute the 2D image of the Vignette in a matplotlib plot
		"""
        image = Image.open(filename + '.png')
        width, height = image.size
        plt.title(str(self.env) + " - " + str(self.policy) + "\n" + str(self.title))
        plt.ylabel("Directions")
        plt.xlabel("Distances to central policy")
        plt.imshow(image, cmap='viridis')
        plt.colorbar(label="Reward", orientation="horizontal", aspect=50)
        plt.clim(self.minColor, self.maxColor)
        if (height // 10) > 10:
            if any(color == "#ff7f0e" for color in self.colors):
                plt.plot(0, 0, "-", c="tab:orange", label="PG policies")
            if any(color == "#d62728" for color in self.colors):
                plt.plot(0, 0, "-", c="tab:red", label="CEM policies")

        if (width // 10) < 20:
            xrate = 4
        elif (width // 10) < 40:
            xrate = 8
        else:
            xrate = 16

        if (width // (xrate * 10)) % 2 == 0:
            xnum = (width // (xrate * 10)) + 1
        else:
            xnum = (width // (xrate * 10))
        xlabels = np.linspace(-self.stepalpha * (width // 20),
                              self.stepalpha * (width // 20),
                              xnum)
        [index_zero] = np.where(xlabels == 0)[0]
        halflabels = np.around(xlabels[:(index_zero)], 2)
        xlabels = list(np.abs(halflabels)) + [0] + list(np.abs(halflabels[::-1]))
        plt.xticks(np.linspace(5, width - 5, xnum), xlabels)
        plt.yticks(np.linspace(5, height - 5, height//10), (np.linspace(1, (height//10), height//10, dtype=int))[::-1], fontsize=3)
        plt.tight_layout()
        if (height // 10) > 10:
            plt.legend(loc='upper center',
                       fancybox=True,
                       framealpha=0.15,
                       ncol=2)

    def plot3D(self,
               figsize=(12, 8),
               title="Vignette 3D",
               surfaces=True,
               transparency=1,
               **kwargs):
        """
        Compute the 3D image of the Vignette with surfaces or not, can be shaped by an input function
		"""
        self.fig, self.ax = plt.figure(
            title, figsize=figsize), plt.axes(projection='3d')
        plt.title(str(self.env) + " - " + str(self.policy) + "\n" + str(self.title))
        # Computing the intial 3D Vignette
        if surfaces is True:
            args = [transformFunction.transformIdentity]

            # Default key word arguments
            defaultWidth, defaultLineWidth, defaultCmap = 5, .01, "viridis"
            if "width" not in kwargs.keys(): kwargs["width"] = defaultWidth
            if "linewidth" not in kwargs.keys():
                kwargs["linewidth"] = defaultLineWidth
            if "cmap" not in kwargs.keys(): kwargs["cmap"] = defaultCmap
        else:
            args = [transformFunction.transformIdentity]

        self.computeFunction(*args,
                             transparency=transparency,
                             surfaces=surfaces,
                             **kwargs)
        plt.legend(loc='upper center', fancybox=True, framealpha=0.15, ncol=2)

    def computeFunction(self,
                        function,
                        transparency=0.6,
                        surfaces=True,
                        width=0,
                        linewidth=0,
                        cmap="viridis"):
        if any(color == "#ff7f0e" for color in self.colors):
            plt.plot(0, 0, "-", c="tab:orange", label="PG policies")
        if any(color == "#d62728" for color in self.colors):
            plt.plot(0, 0, "-", c="tab:red", label="CEM policies")
        sm = plt.cm.ScalarMappable(cmap="viridis",
                                   norm=plt.Normalize(vmin=self.minColor,
                                                      vmax=self.maxColor))
        sm._A = []
        plt.colorbar(sm, label="Reward", aspect=50)
        # Iterate over all lines
        for step in range(0, len(self.directions)):
            # Vignette reads from top to bottom
            height = -step
            line = [self.lines[step][k] for k in range(len(self.lines[step]))]

            transformedLine = function.transform(line)

            # We have to iterate over all input policies at each step for an easier retrieval of parameters
            if self.indicesPolicies is not None:
                for k in range(len(self.indicesPolicies)):
                    if self.indicesPolicies[k] == step:
                        self.makePolicy3D(k,
                                          height,
                                          transformedLine,
                                          width=width,
                                          color=self.colors[k])

            if surfaces is True:
                x_line = np.linspace(-len(line) / 2, len(line) / 2, len(line))
                y_line = height * width * np.ones(len(line))

                X = np.array([x_line, x_line])
                Y = np.array([y_line, y_line + width])

                Z = np.array([transformedLine, transformedLine])

                self.ax.plot_surface(self.x_diff * X,
                                     self.y_diff * Y,
                                     Z,
                                     cmap=cmap,
                                     linewidth=linewidth,
                                     alpha=transparency,
                                     norm=plt.Normalize(vmin=self.minColor,
                                                        vmax=self.maxColor))
            else:
                x_line = np.linspace(-len(line) / 2, len(line) / 2, len(line))
                y_line = np.ones(len(line))

                self.ax.plot3D(self.x_diff * x_line,
                               self.y_diff * height * y_line, transformedLine)

            # Plotting user information
            #	Sampled policies
            self.ax.set_xlabel("Distances to central policy")
            posits = [np.ceil(self.x_diff * step) for step in np.linspace(-len(self.lines[0])//2+1, 0, len(self.lines[0])//(len(self.lines[0])//3))] \
                + [np.floor(self.x_diff * step) for step in np.linspace(0, len(self.lines[0])//2+1, len(self.lines[0])//(len(self.lines[0])//3))]
            values = list(np.ceil(np.linspace(int(max(max(self.policyDistance), len(self.lines[0])//2)), 0, len(self.lines[0])//(len(self.lines[0])//3)))) \
                + list(np.ceil(np.linspace(0, int(max(max(self.policyDistance), len(self.lines[0])//2)), len(self.lines[0])//(len(self.lines[0])//3))))
            self.ax.set_xticks(posits)
            self.ax.set_xticklabels(np.around(values, 2))

            #	Sampled directions
            self.ax.set_ylabel("Lines")
        if surfaces is True:
            posits = [self.y_diff * (round(width / 2) - step * width)
                for step in np.linspace(0, len(self.directions)-1, len(self.directions))]
        else:
            posits = [self.y_diff * (-step)
                for step in np.linspace(0, len(self.directions)-1, len(self.directions))]

        values = list(np.linspace(1, len(self.directions), len(self.directions), dtype=int))
        self.ax.set_yticks(posits)
        self.ax.set_yticklabels(values, fontsize=6)

        # 	Reward
        self.ax.set_zlabel("Reward")

    def makePolicy3D(self, index, height, line, width=0, color='white'):
        """
		Plot policies input points on the savedVignette's 3D plot
		"""
        distance = round(self.policyDistance[index] /
                         self.stepalpha)  # Rounding error ?
        dy = width if width != 0 else 1
        x, y, z = self.x_diff * distance, self.y_diff * (
            (height * dy) + round(width / 2)), line[round(len(line) // 2) +
                                                    distance]
        self.ax.scatter(x, y, z, marker='o', color=color, s=700)

    def changeColors(self, color1=None, color2=None):
        self.color1 = color1 if color1 is not None else self.color1
        self.color2 = color2 if color2 is not None else self.color2


if __name__ == "__main__":

    args = get_args()
    print(args)
    directory = os.getcwd() + '/Models/'
    LoadedVignette = loadFromFile(args.filename, folder=args.directoryFile)
    angles3D = [20, 45, 50, 65]  # angles at which to save the plot3D
    elevs = [0, 30, 60]
    LoadedVignette.saveAll(args.filename,
                           saveInFile=False,
                           save2D=True,
                           save3D=True,
                           directoryFile="SavedVignette",
                           directory2D="Vignette_output",
                           directory3D="Vignette_output",
                           computedImg=None,
                           angles3D=angles3D,
                           elevs=elevs)
