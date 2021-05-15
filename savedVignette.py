# coding: utf-8

import pickle
import lzma

import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.image as img

from PIL import Image, ImageDraw

import colorTest
from vector_util import valueToRGB, invertColor, checkFormat

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
                 colors=None):

        # Content of the Vignette
        self.baseLines = []  # Bottom lines
        self.lines = []  # Upper lines
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
        img = self.plot2D() if img is None else img
        img.save(filename, format='png')

    @checkFormat('.pdf')
    def save3D(self, filename, elevs=[30], angles=[0]):
        """
		Save the Vignette as a 3D image
		"""
        for elev in elevs:
            for angle in angles:
                self.ax.view_init(elev, angle)
                plt.draw()
                plt.savefig(filename + '_e{}_a{}.pdf'.format(elev, angle),
                            format='pdf')

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
        if saveInFile is True: self.saveInFile(directoryFile + '/' + filename)
        if save2D is True:
            self.save2D(directory2D + '/' + filename + '_2D', img=computedImg)
            self.plot2Dcleaned(directory2D, filename)
        #if save3D is True: self.save3D(directory3D+'/'+filename+'_3D', elevs=elevs, angles=angles3D)

    def plot2D(self, color1=None, color2=None):
        """
		Compute the 2D image of the Vignette

		Cannot store it as PIL images are non serializable
		"""
        color1, color2 = self.color1 if color1 is None else color1, self.color2 if color2 is None else color2

        width, height = self.pixelWidth * len(
            self.lines[-1]), self.pixelHeight * (len(self.lines))
        #width, height = self.pixelWidth * len(self.lines[-1]), self.pixelHeight * (len(self.lines) + len(self.policyDistance) + len(self.baseLines) + 1)
        newIm = Image.new("RGB", (width, height))
        newDraw = ImageDraw.Draw(newIm)

        maxColor = -100
        minColor = -1900
        for l in range(len(self.lines)):
            for c in range(len(self.lines[l])):
                if self.lines[l][c] > maxColor:
                    maxColor = self.lines[l][c]
                if self.lines[l][c] < minColor:
                    minColor = self.lines[l][c]
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
                                   minNorm=minColor,
                                   maxNorm=maxColor)
                newDraw.rectangle([x0, y0, x1, y1], fill=color)
            y0 += self.pixelHeight

        # 	Adding the separating line
        #y0 += self.pixelHeight
        #y1 = y0 + self.pixelHeight
        #color = valueToRGB(0,
        #                   color1,
        #                   color2,
        #                   minNorm=minColor,
        #                   maxNorm=maxColor)
        #newDraw.rectangle([0, y0, width, y1], fill=color)

        #	Adding the baseLines (bottom lines)
        #for l in range(len(self.baseLines)):
        #    y0 += self.pixelHeight
        #    y1 = y0 + self.pixelHeight
        #    for c in range(len(self.lines[l])):
        #        x0 = c * self.pixelWidth
        #        x1 = x0 + self.pixelWidth
        #        color = valueToRGB(self.baseLines[l][c],
        #                           color1,
        #                           color2,
        #                           minNorm=minColor,
        #                           maxNorm=maxColor)
        #        newDraw.rectangle([x0, y0, x1, y1], fill=color)

        # 	Adding the policies
        if self.indicesPolicies is not None:
            marginX, marginY = int(self.pixelWidth / 4), int(self.pixelHeight /
                                                             4)
            for k in range(len(self.indicesPolicies)):
                #print(self.indicesPolicies)
                #print("nb lines : " + str(self.lines))
                index, distance = self.indicesPolicies[k], round(
                    self.policyDistance[k] / self.stepalpha)
                #print("index : "+ str(index))
                x0, y0 = (distance + len(self.lines[0]) //
                          2) * self.pixelWidth, index * self.pixelHeight
                x1, y1 = x0 + self.pixelWidth, y0 + self.pixelHeight
                #color = invertColor(newIm.getpixel((x0,y0)))
                newDraw.ellipse(
                    [x0 + marginX, y0 + marginY, x1 - marginX, y1 - marginY],
                    fill=self.colors[k + 1])
                #newDraw.text((x0+ int(1.5 * marginX), y0), str(k), fill=invertColor(color))

        return newIm

    def plot2Dcleaned(self, directory2D, filename):
        """
		Clean the 2D image of the Vignette
		"""
        im = img.imread(directory2D + '/' + filename + '_2D' + '.png')
        image = Image.open(directory2D + '/' + filename + '_2D' + '.png')
        width, height = image.size
        #print(width)
        #print(height)
        plt.title(label="Landscape")
        plt.ylabel("Arbitrary directions")
        plt.xlabel("Distances to central policy")
        plt.imshow(image, cmap='viridis')
        #for i in np.linspace(5, height-5, ((height-10)//10)+1,  dtype=int):
        #	plt.text(width//2, i, ' ', bbox={'facecolor': 'tab:orange', 'ec': 'tab:orange'})
        plt.colorbar(label="Reward", orientation="horizontal")
        plt.clim(-1900, -100)
        if (height // 10) > 10:
            if any(color == "#d62728" for color in self.colors):
                plt.plot(0, 0, "-", c="tab:red", label="PG policies")
            if any(color == "#9467bd" for color in self.colors):
                plt.plot(0, 0, "-", c="tab:purple", label="CEM policies")

        if (width // 10) < 20:
            xrate = 2
        elif (width // 10) < 40:
            xrate = 4
        else:
            xrate = 8

        if (width // (xrate * 10)) % 2 == 0:
            xnum = (width // (xrate * 10)) + 1
        else:
            xnum = (width // (xrate * 10))
        xlabels = np.linspace(-self.stepalpha * (width // 20),
                              self.stepalpha * (width // 20),
                              xnum,
                              dtype=int)
        xlabels = np.concatenate(((xlabels[:np.shape(xlabels)[0] // 2] + 1),
                                  xlabels[np.shape(xlabels)[0] // 2:]),
                                 axis=None)
        plt.xticks(np.linspace(5, width - 5, xnum), map(str, xlabels))
        #print((height-10)//10)
        #plt.yticks(np.linspace(5, height-5, ((height-10)//(yrate*10))+1,  dtype=int), map(str, np.arange(1, ((height-10)//(rate*10))+2, dtype=int)))
        plt.yticks([])
        plt.tight_layout()
        #plt.text(5, -25, 'PG Policies : ', bbox={'facecolor': 'white', 'ec': 'white'})
        #plt.text(30, -25, '   ', bbox={'facecolor': 'tab:red', 'ec': 'tab:red'})
        #plt.text(45, -25, 'CEM Policies : ', bbox={'facecolor': 'white', 'ec': 'white'})
        #plt.text(73, -25, '   ', bbox={'facecolor': 'tab:purple', 'ec': 'tab:purple'})
        #plt.text(85, -25, 'Starting Policy : ', bbox={'facecolor': 'white', 'ec': 'white'})
        #plt.text(117, -25, '   ', bbox={'facecolor': 'tab:orange', 'ec': 'tab:orange'})
        if (height // 10) > 10:
            plt.legend(loc='upper center',
                       fancybox=True,
                       framealpha=0.15,
                       ncol=2)
        plt.savefig(directory2D + '/' + filename + '_2D_Cleaned' + '.pdf')

    def plot3D(self,
               function=lambda x: x,
               figsize=(12, 8),
               title="Vignette ligne"):
        """
		Compute the 3D image of the Vignette
		"""
        self.fig, self.ax = plt.figure(
            title, figsize=figsize), plt.axes(projection='3d')
        # Iterate over all lines
        for step in range(0, len(self.directions)):
            # Check if current lines is a baseLine
            #if step == -1:
            #	# baseLines are at the bottom of the image
            #	height = -len(self.directions)-1
            #	line = self.baseLines[0]
            #else:
            # Vignette reads from top to bottom
            height = -step
            line = self.lines[step]

            x_line = np.linspace(-len(line) / 2, len(line) / 2, len(line))
            y_line = np.ones(len(line))

            self.ax.plot3D(self.x_diff * x_line, self.y_diff * height * y_line,
                           function(line))

    def plot3DBand(self,
                   function=lambda x: x,
                   figsize=(12, 8),
                   title="Vignette surface",
                   width=5,
                   linewidth=.01,
                   cmap='coolwarm'):
        """
		Compute the 3D image of the Vignette with surfaces
		"""
        self.fig, self.ax = plt.figure(
            title, figsize=figsize), plt.axes(projection='3d')
        # Iterate over all lines
        for step in range(0, len(self.directions)):
            # Check if current lines is a baseLine
            #if step == -1:
            #	# baseLines are at the bottom of the image
            #	height = -len(self.directions)-1
            #	line = self.baseLines[0]
            #else:
            # Vignette reads from top to bottom
            height = -step
            line = self.lines[step]

            x_line = np.linspace(-len(line) / 2, len(line) / 2, len(line))
            y_line = height * width * np.ones(len(line))

            X = np.array([x_line, x_line])
            Y = np.array([y_line, y_line + width])

            newLine = function(line)
            Z = np.array([newLine, newLine])

            self.ax.plot_surface(self.x_diff * X,
                                 self.y_diff * Y,
                                 Z,
                                 cmap=cmap,
                                 linewidth=linewidth)

    def show2D(self, img=None, color1=None, color2=None):
        color1, color2 = self.color1 if color1 is None else color1, self.color2 if color2 is None else color2
        img = self.plot2D(color1, color2) if img is None else img
        img.show()

    def show3D(self):
        plt.show()

    def changeColors(self, color1=None, color2=None):
        self.color1 = color1 if color1 is not None else self.color1
        self.color2 = color2 if color2 is not None else self.color2


if __name__ == "__main__":
    directoryFile = "SavedVignette"
    directory2D = "Vignette_output"
    SavedVignette.plot2Dcleaned(directory2D, "new_grad_1")
