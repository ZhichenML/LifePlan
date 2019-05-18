#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# use mesh result rather than scanning by two if.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SGD:
    @classmethod
    def __init__(self, threshold, max_iter):
        self.threshold = threshold
        self.max_iter = max_iter
        self.ret_list = np.array([]) #np.zeros((max_iter, 2))
        self.A = np.array([[3, 2], [2, 6]])
        self.b = np.array([2, -8])
        self.Iter_num = 0# number of iterations before stop

    @classmethod
    def clear_history(self):
        self.Iter_num = 0
        self.ret_list = np.array([])

    @classmethod
    def Eval(self,x):
        ret = 0.5 * np.dot(np.dot(x.T, self.A),x) - np.dot(self.b.T, x)
        return ret

    @classmethod
    def gredient(self, x):
        deriv = np.dot(self.A, x) - self.b
        return deriv

    @classmethod
    def Steepest_search(self, x):
        x1 = x
        x2 = np.zeros_like(x)
        self.ret_list = x1 # the first location
        for i in range(self.max_iter):

            r = self.b - np.dot(self.A, x1)
            alpha = np.dot(r.T,r)/ np.dot(r.T,np.dot(self.A,r))
            x2 = x1 + alpha*r

            self.Iter_num += 1
            self.ret_list = np.vstack((self.ret_list, x2))

            if abs(np.sum(x2-x1)) < self.threshold:
                return x2

            x1 = x2

        return x2


    @classmethod
    def Jacobi_search(self, x):
        # for ill-conditioned optimization, Jacobi optimization seems better than SGD
        # this may be achieved by trasforming the optimization axis into a new one (B)
        # the spectral radius of B has more significant effect on convergence
        # x(i+1) = Bx + z
        D_inv = np.diag(1/np.diag(self.A))
        E = self.A - np.diag(np.diag(self.A))
        B = - np.dot(D_inv, E)
        z = np.dot(D_inv, self.b)

        # begin iteration
        x1 = x
        x2 = np.zeros_like(x)
        self.ret_list = x1
        for _ in range(self.max_iter):
            x2 = np.dot(B, x1) + z
            self.ret_list = np.vstack((self.ret_list, x2))
            self.Iter_num += 1

            if abs(sum(x2-x1)) < self.threshold:
                return x2
            x1 = x2

        return x2

    @classmethod
    def CG_search(self, x):
        x1 = x
        x2 = np.zeros_like(x)
        self.ret_list = x1
        r1 = self.b - np.dot(self.A, x1)
        d = r1
        r2 = r1

        for i in range(self.max_iter):
            alpha = np.dot(r2.T,r2) / np.dot(np.dot(d.T, self.A), d)
            x2 = x1 + alpha * d
            self.ret_list = np.vstack((self.ret_list, x2)) # tuple parameter in np.vstack
            self.Iter_num += 1
            if abs(sum(x1-x2)) < self.threshold:
                return x2

            r1 = r2
            r2 = r1 - alpha * np.dot(self.A, d)

            beta = np.dot(r2.T, r2) / np.dot(r1.T, r1)
            d = r2 + beta * d
            x1 = x2

        return x2




def example():
    # generate grid data
    n = 100
    tmp = np.linspace(-4,6,n)
    x,y = np.meshgrid(tmp,tmp)

    # function evaluations for the samples
    f1 = np.zeros_like(x)
    x_len, y_len = x.shape
    deriv1 = np.zeros((x_len, y_len, 2))
    SGD(10e-15, 100)

    for i in range(x_len):
        for j in range(y_len):
            f1[i,j] = SGD.Eval(np.array([x[i,j],y[i,j]]))
            deriv1[i, j] = SGD.gredient([x[i,j],y[i,j]])
#            print(deriv1[i,j,1])

    # optimization
    result = SGD.Steepest_search(np.array([-.4, 4]))
    print('Best solution:', result)
    print('Terminates in %d iteration\n' % SGD.Iter_num)


    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(2, 2, 4)
    f_val = np.array([])
    #print(SGD.ret_list)
    for i in range(SGD.Iter_num):
        f_val = np.append(f_val, SGD.Eval(SGD.ret_list[i, :]))

    plt.plot(range(len(f_val)), f_val)
    plt.xlabel('#iteration')
    plt.ylabel('Loss function')

    #Axes3D(fig) # build a 3D axis
    ax = fig.add_subplot(2,2,1, projection='3d')

    #ax.plot_surface(x,y,f1, rstride = 5, cstride =5, cmap='gray')
    ax.plot_wireframe(x, y, f1, rstride = 2, cstride = 2, colors = 'black')
    ax.view_init(45,240)
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.grid()


    ax = fig.add_subplot(2,2,2)
    #plt.contourf(x, y, f1,20, alpha=.75)
    C = plt.contour(x,y,f1,50, colors = 'black')
    plt.clabel(C, inline=True, fontsize=10)
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.grid()

    for i in range(SGD.Iter_num):
        ax.arrow(SGD.ret_list[i,0], SGD.ret_list[i,1], SGD.ret_list[i+1,0] - SGD.ret_list[i,0], SGD.ret_list[i+1,1] - SGD.ret_list[i,1],
                 length_includes_head = False, head_width = 0.25, head_length = .21, fc = 'k', ec = 'k')



    ax = fig.add_subplot(2,2,3)
    #print(deriv1)
    max_v = np.max(deriv1,axis=1)
    max_v = np.max(max_v,axis=0)
    deriv1[:,:,0] = deriv1[:,:,0]/abs(max_v[0])
    deriv1[:, :, 1] = deriv1[:, :, 1] / abs(max_v[1])
    #Q = plt.quiver(x,y,deriv1[:,:,0],deriv1[:,:,1])
    #print(deriv1)

    for i in range(1,x_len,5):
        for j in range(1,y_len,5):
            ax.arrow(x[i,j], y[i,j], deriv1[i,j,0], deriv1[i,j,1],
                    length_includes_head=False,  # 增加的长度包含箭头部分
                    head_width=0.25, head_length=0.1, fc='k', ec='k')
            # ax.annotate("", xy=(x[i,j]+deriv1[i,j,0]*.09, y[i,j]+deriv1[i,j,1]*.09), xytext=(x[i,j], y[i,j]), arrowprops=dict(arrowstyle="->"))

    ax.set_xlim(-4, 6)
    ax.set_ylim(-4, 6)
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.grid()
    ax.set_aspect('equal')  # x轴y轴等比例




    fig.tight_layout()
    plt.show()

    # 保存图片，通过pad_inches控制多余边缘空白
    plt.savefig('arrow.png', transparent=True, bbox_inches='tight', pad_inches=0.25)

def drawArrow1(A, B):
    '''
    Draws arrow on specified axis from (x, y) to (x + dx, y + dy).
    Uses FancyArrow patch to construct the arrow.

    The resulting arrow is affected by the axes aspect ratio and limits.
    This may produce an arrow whose head is not square with its stem.
    To create an arrow whose head is square with its stem, use annotate() for example:
    Example:
        ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->"))
    '''
    fig = plt.figure()
    ax = fig.add_subplot(121)
    # fc: filling color
    # ec: edge color
    ax.arrow(A[0], A[1], B[0]-A[0], B[1]-A[1],
             length_includes_head=True,# 增加的长度包含箭头部分
             head_width=0.25, head_length=0.5, fc='r', ec='b')
    # 注意： 默认显示范围[0,1][0,1],需要单独设置图形范围，以便显示箭头
    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)
    ax.grid()
    ax.set_aspect('equal') #x轴y轴等比例
    # Example:
    ax = fig.add_subplot(122)
    ax.annotate("", xy=(B[0], B[1]), xytext=(A[0], A[1]),arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.grid()
    ax.set_aspect('equal') #x轴y轴等比例
    plt.show()
    plt.tight_layout()
    #保存图片，通过pad_inches控制多余边缘空白
    #plt.savefig('arrow.png', transparent = True, bbox_inches = 'tight', pad_inches = 0.25)

#%%
def draw_example():
    a = np.array([1,2])
    b = np.array([3,4])
    drawArrow1(a,b)


if __name__ == "__main__":
    example()
