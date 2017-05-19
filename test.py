from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from visdom import Visdom
import numpy as np
import math
import os.path
import getpass

# a = np.array([1,1])
# # a = np.squeeze(a)
# print(a.shape)

viz = Visdom()
win = None
# line updates
win = viz.line(
    X=np.array([[0]]),
    Y=np.array([[1]]),
    win=win,
    update=None
)
# viz.line(
#     X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
#     Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
#     win=win,
#     update='append'
# )
# viz.updateTrace(
#     X=np.arange(21, 30),
#     Y=np.arange(1, 10),
#     win=win,
#     name='2'
# )
