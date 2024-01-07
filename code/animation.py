import matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator) #minor grid
import numpy as np

import constants

def plotAnimation (xx_ref, xx_star):
    tt_hor = np.linspace(0, constants.TF, constants.TT)
    time = np.arange(len(tt_hor)) * constants.DT
    
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 1), ylim=(-.5, 1.2))
    ax.grid()
    # no labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

  
    line0, = ax.plot([], [], 'o-', lw=2, c='b', label='Optimal')
    line1, = ax.plot([], [], '*-', lw=2, c='g',dashes=[2, 2], label='Reference')

    time_template = 't = %.1f s'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    fig.gca().set_aspect('equal', adjustable='box')

    # Subplot
    left, bottom, width, height = [0.64, 0.13, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.xaxis.set_major_locator(MultipleLocator(2))
    ax2.yaxis.set_major_locator(MultipleLocator(0.25))
    ax2.set_xticklabels([])
  

    ax2.grid(which='both')
    ax2.plot(time, xx_star[0],c='b')
    ax2.plot(time, xx_ref[0], color='g', dashes=[2, 1])

    point1, = ax2.plot([], [], 'o', lw=2, c='b')


    def init():
        line0.set_data([], [])
        line1.set_data([], [])

        point1.set_data([], [])

        time_text.set_text('')
        return line0,line1, time_text, point1


    def animate(i):
      # Trajectory
      thisx0 = [0, np.sin(xx_star[0, i])]
      thisy0 = [0, np.cos(xx_star[0, i])]
      line0.set_data(thisx0, thisy0)

      # Reference
      thisx1 = [0, np.sin(xx_ref[0, -1])]
      thisy1 = [0, np.cos(xx_ref[0, -1])]
      line1.set_data(thisx1, thisy1)

      point1.set_data(i*constants.DT, xx_star[0, i])

      time_text.set_text(time_template % (i*constants.DT))
      return line0, line1, time_text, point1


    ani = animation.FuncAnimation(fig, animate, constants.TT, interval=1, blit=True, init_func=init)
    ax.legend(loc="lower left")

    plt.show()